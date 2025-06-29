#include "main.h"
#include "twopass.h"
#include "laser.h"
#include <iostream>
#include <fstream>

// std::vector<cv::Point3f> loadVecPoint3f(std::string filename)
// {
//     std::vector<cv::Point3f> vec_points;
// std::fstream fsread(filename);
//  std::string buf_str;
//  while (std::getline(fsread, buf_str))
//  {
//   std::stringstream ss(buf_str);
//   if (!ss.eof())
//   {
//    float x, y, z;
//    ss >> x >> y >> z;
//    vec_points.emplace_back(x, y, z);
//   }
//   buf_str.clear();
//   ss.clear();
//  }
//  const bool bOk = !fsread.bad();
//  fsread.close();
//  return vec_points;
// }

int main() {

    loadImgs();
    
    bool is_load = loadStereoCalibInfo();
    if (!is_load) throw std::logic_error("can't load stereo calib info");
    std::string yml_file = (calib_info_dir_ / "laser_calib.yml").string();
    std::vector<Plane> planes_info;
    is_load = loadPlaneCalibInfo(yml_file, planes_info);
    ConfigManager::getInstance().setPlane(planes_info);
    if (!is_load) throw std::logic_error("can't load planes calib info");

    auto calib_info = ConfigManager::getInstance().getCalibInfo();

    for (size_t img_idx = 0; img_idx < laser_imgs_.size(); ++img_idx) {
        cv::Mat img_l, img_r;
        std::tie(img_l, img_r) = laser_imgs_[img_idx];
        auto rectify_imgs_have_laser = getEpipolarRectifyImage(
            processImg(img_l, 0, true),
            processImg(img_r, 1, true)
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
        LaserProcessor laser_processor;
        cv::Mat laser_img_l = rectify_imgs_have_laser[0].clone();
        cv::Mat laser_img_r = rectify_imgs_have_laser[1].clone();
        auto laser_l = laser_processor.extractLine(roi_l, laser_img_l);
        auto laser_r = laser_processor.extractLine(roi_r, laser_img_r);
        cv::cvtColor(laser_img_l, laser_img_l, cv::COLOR_GRAY2BGR);
        cv::cvtColor(laser_img_r, laser_img_r, cv::COLOR_GRAY2BGR);
        for (const auto& l : laser_l)
            for (const auto& [y, p] : l.points) cv::circle(laser_img_l, cv::Point2f(p.x, y), 1, cv::Scalar(255, 0, 255), -1);
        for (const auto& l : laser_r)
            for (const auto& [y, p] : l.points) cv::circle(laser_img_r, cv::Point2f(p.x, y), 1, cv::Scalar(255, 0, 255), -1);
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
        for (size_t idx = 0; idx < laser_l.size(); ++idx) {
            std::map<float, float> mmp;
            for (const auto& p : laser_l[idx].points)
                mmp[p.second.y] = p.second.x;
            sample_points_l[idx] = mmp;
        }


        // 重投影到右图
        laser_processor.match(sample_points_l, laser_r, rectify_imgs_have_laser[0], rectify_imgs_have_laser[1]);
    }

    return 0;
}