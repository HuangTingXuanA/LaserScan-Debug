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
        // if (img_idx <= 8) continue;
        printf("idx: %d\n", (int)img_idx);
        
        //  极线校正
        cv::Mat img_l, img_r;
        std::tie(img_l, img_r) = laser_imgs_[img_idx];
        auto rectify_imgs_have_laser = getEpipolarRectifyImage(
            img_l,
            img_r
        );
        calib_info = ConfigManager::getInstance().getCalibInfo();

        // 预处理图像
        cv::Mat bin_img_l, bin_img_r;
        bin_img_l = processImg2(rectify_imgs_have_laser[0], 0, true);
        bin_img_r = processImg2(rectify_imgs_have_laser[1], 1, true);

        // 生成连通区域和二值化图
        std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours_l, contours_r;
        Two_PassNew3(bin_img_l, contours_l, img_idx);
        Two_PassNew3(bin_img_r, contours_r, img_idx);

        // 只保留激光区域的原始像素
        cv::Mat laser_img_l = cv::Mat::zeros(rectify_imgs_have_laser[0].size(), CV_8UC1);
        cv::Mat laser_img_r = cv::Mat::zeros(rectify_imgs_have_laser[1].size(), CV_8UC1);
        rectify_imgs_have_laser[0].copyTo(laser_img_l, bin_img_l);
        rectify_imgs_have_laser[1].copyTo(laser_img_r, bin_img_r);

        // 激光线中心点提取
        LaserProcessor laser_processor;
        auto laser_l = laser_processor.extractLine2(laser_img_l, contours_l, img_idx);
        auto laser_r = laser_processor.extractLine2(laser_img_r, contours_r, img_idx);
        cv::Mat vis_img_l = rectify_imgs_have_laser[0].clone();
        cv::Mat vis_img_r = rectify_imgs_have_laser[1].clone();
        cv::cvtColor(vis_img_l, vis_img_l, cv::COLOR_GRAY2BGR);
        cv::cvtColor(vis_img_r, vis_img_r, cv::COLOR_GRAY2BGR);
        for (const auto& l : laser_l)
            for (const auto& [y, p] : l.points) vis_img_l.at<cv::Vec3b>(y, p.x) = cv::Vec3b(255, 0, 255);
        for (const auto& l : laser_r)
            for (const auto& [y, p] : l.points) vis_img_r.at<cv::Vec3b>(y, p.x) = cv::Vec3b(255, 0, 255);
        cv::imwrite(debug_img_dir / ("laser_img_l" + std::to_string(img_idx) + ".jpg"), vis_img_l);
        cv::imwrite(debug_img_dir / ("laser_img_r" + std::to_string(img_idx) + ".jpg"), vis_img_r);

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
        // auto match_vec_tuple = laser_processor.match4(sample_points_l, laser_r, rectify_imgs_have_laser[0], rectify_imgs_have_laser[1]);

        // 同名点匹配
        // auto cloud_points = laser_processor.generateCloudPoints(match_vec_tuple, laser_l, laser_r);

        auto match_res = laser_processor.match5(sample_points_l, laser_r, rectify_imgs_have_laser[0], rectify_imgs_have_laser[1]);
        auto cloud_points = laser_processor.generateCloudPoints2(match_res, laser_l, laser_r);

        std::string txt_file = std::to_string(img_idx);
        std::ofstream ofs(txt_file + ".txt");
        for (const auto& pt : cloud_points) {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }
        ofs.close();
    }

    return 0;
}
