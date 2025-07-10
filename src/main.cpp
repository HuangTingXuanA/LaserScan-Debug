#include "main.h"
#include "twopass.h"
#include "laser.h"
#include <iostream>
#include <fstream>


int main() {

    loadImgs();
    // loadTrackImgs();

    bool is_load = loadStereoCalibInfo();
    if (!is_load) throw std::logic_error("can't load stereo calib info");
    is_load = loadQuadSurfaceInfo();
    if (!is_load) throw std::logic_error("can't load planes calib info");

    auto calib_info = ConfigManager::getInstance().getCalibInfo();

    for (size_t img_idx = 0; img_idx < laser_imgs_.size(); ++img_idx) {
        // if (img_idx != 11) continue;
        
        cv::Mat img_l, img_r;
        std::tie(img_l, img_r) = laser_imgs_[img_idx];
        auto rectify_imgs_have_laser = getEpipolarRectifyImage(
            img_l,
            img_r
        );
        calib_info = ConfigManager::getInstance().getCalibInfo();

        // 生成连通区域
        cv::Mat label_img_l, label_img_r;
        cv::Mat color_label_img_l, color_label_img_r;
        Two_PassNew(rectify_imgs_have_laser[0], label_img_l);
        Two_PassNew(rectify_imgs_have_laser[1], label_img_r);
	    LabelColor(label_img_l, color_label_img_l);
        LabelColor(label_img_r, color_label_img_r);
        cv::imwrite(debug_img_dir / ("labelImg_l" + std::to_string(img_idx) + ".jpg"), color_label_img_l);
        cv::imwrite(debug_img_dir / ("labelImg_r" + std::to_string(img_idx) + ".jpg"), color_label_img_r);

        // ROI框选
        cv::Mat roi_img_l = rectify_imgs_have_laser[0].clone();
        cv::Mat roi_img_r = rectify_imgs_have_laser[1].clone();
        cv::cvtColor(roi_img_l, roi_img_l, cv::COLOR_GRAY2BGR);
        cv::cvtColor(roi_img_r, roi_img_r, cv::COLOR_GRAY2BGR);
        auto roi_l = DetectLaserRegions(label_img_l);
        auto roi_r = DetectLaserRegions(label_img_r);
        for (const auto& rect : roi_l) {
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; ++i) {
                cv::line(roi_img_l, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
            }
        }
        for (const auto& rect : roi_r) {
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; ++i) {
                cv::line(roi_img_r, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
            }

        }
        cv::imwrite(debug_img_dir / ("roiImg_l" + std::to_string(img_idx) + ".jpg"), roi_img_l);
        cv::imwrite(debug_img_dir / ("roiImg_r" + std::to_string(img_idx) + ".jpg"), roi_img_r);


        // 激光线中心点提取
        rectify_imgs_have_laser[0] = processImg(rectify_imgs_have_laser[0], 0, true);
        rectify_imgs_have_laser[1] = processImg(rectify_imgs_have_laser[1], 1, true);
        LaserProcessor laser_processor;
        cv::Mat laser_img_l = rectify_imgs_have_laser[0].clone();
        cv::Mat laser_img_r = rectify_imgs_have_laser[1].clone();
        auto laser_l = laser_processor.extractLine(roi_l, laser_img_l);
        auto laser_r = laser_processor.extractLine(roi_r, laser_img_r);
        cv::cvtColor(laser_img_l, laser_img_l, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laser_img_r, laser_img_r, cv::COLOR_GRAY2BGR);
        for (const auto& l : laser_l)
            for (const auto& [y, p] : l.points) laser_img_l.at<cv::Vec3b>(y, std::round(p.x)) = cv::Vec3b(255, 0, 255);
        for (const auto& l : laser_r)
            for (const auto& [y, p] : l.points) laser_img_r.at<cv::Vec3b>(y, std::round(p.x)) = cv::Vec3b(255, 0, 255);
        cv::imwrite(debug_img_dir / ("laser_img_l" + std::to_string(img_idx) + ".jpg"), laser_img_l);
        cv::imwrite(debug_img_dir / ("laser_img_r" + std::to_string(img_idx) + ".jpg"), laser_img_r);

        // 左线均匀采样
        std::vector<std::map<float, float>> sample_points_l(laser_l.size());
        // const int sample_n = 50;
        // auto sample = [&](const LaserLine& line) -> std::map<float, float> {
        //     std::map<float, float> result;
        //     if (line.points.empty()) throw std::logic_error("can not sample");

        //     // 将map中的点按y排序（本身已排序）
        //     std::vector<const LaserPoint*> pts;
        //     for (const auto& kv : line.points) {
        //         pts.push_back(&kv.second);
        //     }
        //     if (pts.size() <= sample_n) {
        //         // 点数不足，全部采样
        //         for (const auto* p : pts) result[p->y] = p->x;
        //         return result;
        //     }
        //     // 均匀采样
        //     for (int sample_idx = 0; sample_idx < sample_n; ++sample_idx) {
        //         size_t idx = static_cast<size_t>(sample_idx * (pts.size() - 1) / (sample_n - 1));
        //         result[pts[idx]->y] = pts[idx]->x;
        //     }
        //     return result;
        // };


        sort(laser_l.begin(), laser_l.end(), [](const auto& l1, const auto& l2){
            return l1.points.size() > l2.points.size();
        });

        for (size_t idx = 0; idx < laser_l.size(); ++idx) {
            std::map<float, float> mmp;
            for (const auto& p : laser_l[idx].points)
                mmp[p.second.y] = p.second.x;
            sample_points_l[idx] = mmp;
        }

        // 重投影到右图（l_idx, plane_idx, r_idx）
        auto match_vec_tuple = laser_processor.match4(sample_points_l, laser_r, rectify_imgs_have_laser[0], rectify_imgs_have_laser[1]);

        // 同名点匹配
        auto cloud_points = laser_processor.generateCloudPoints(match_vec_tuple, laser_l, laser_r);

        std::string txt_file = std::to_string(1 + img_idx);
        std::ofstream ofs(txt_file + ".txt");
        for (const auto& pt : cloud_points) {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }
        ofs.close();
    }

    return 0;
}