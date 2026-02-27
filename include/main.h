#pragma once
#include "configer.h"
#include <atomic>
#include <numeric>
#include <regex>
#include <opencv2/core/ocl.hpp>

// ============================================================
// 加载图像对
// 要求 dir 下有 left/ 和 right/ 子目录，文件名相同的视为配对
// ============================================================
void loadImgs() {
    std::vector<fs::path> left_imgs, right_imgs;
    for (const auto& e : fs::directory_iterator(laser_imgs_dir_ / "left"))
        if (e.is_regular_file()) left_imgs.push_back(e.path());
    for (const auto& e : fs::directory_iterator(laser_imgs_dir_ / "right"))
        if (e.is_regular_file()) right_imgs.push_back(e.path());

    if (left_imgs.size() != right_imgs.size() || left_imgs.empty())
        throw std::logic_error("左右图像数量不匹配或为空");

    // 按文件名中的数字排序（兼容 image_1.bmp 等命名）
    auto extract_index = [](const fs::path& p) -> int {
        std::smatch m;
        std::string stem = p.stem().string();
        if (std::regex_search(stem, m, std::regex(R"(\d+)")))
            return std::stoi(m[0]);
        return -1;
    };
    auto cmp = [&](const fs::path& a, const fs::path& b) {
        return extract_index(a) < extract_index(b);
    };
    std::sort(left_imgs.begin(),  left_imgs.end(),  cmp);
    std::sort(right_imgs.begin(), right_imgs.end(), cmp);

    laser_imgs_.clear();
    image_names_.clear();
    for (size_t i = 0; i < left_imgs.size(); ++i) {
        cv::Mat img_l = cv::imread(left_imgs[i].string(),  cv::IMREAD_GRAYSCALE);
        cv::Mat img_r = cv::imread(right_imgs[i].string(), cv::IMREAD_GRAYSCALE);
        if (img_l.empty() || img_r.empty())
            throw std::logic_error("Failed to load image: " + left_imgs[i].string());
        laser_imgs_.emplace_back(img_l, img_r);
        image_names_.push_back(left_imgs[i].stem().string()); // 取文件名（无扩展名）
    }
}

// ============================================================
// 加载双目标定信息
// ============================================================
bool loadStereoCalibInfo() {
    fs::path filepath = calib_info_dir_ / "stereo_calib.yml";
    cv::FileStorage fs_file(filepath, cv::FileStorage::READ);
    if (!fs_file.isOpened()) return false;

    CalibrationResult calib;
    fs_file["camera_cameraMatrix"]    >> calib.camera_matrix[0];
    fs_file["camera_distCoeffs"]      >> calib.dist_coeffs[0];
    fs_file["projector_cameraMatrix"] >> calib.camera_matrix[1];
    fs_file["projector_distCoeffs"]   >> calib.dist_coeffs[1];

    int image_count = 0;
    fs_file["camera_poses_num"] >> image_count;
    for (int i = 0; i < image_count; ++i) {
        cv::Mat rvec, tvec;
        fs_file["camera_rvec_"    + std::to_string(i)] >> rvec;
        fs_file["camera_tvec_"    + std::to_string(i)] >> tvec;
        calib.rvecs[0].emplace_back(rvec); calib.tvecs[0].emplace_back(tvec);

        fs_file["projector_rvec_" + std::to_string(i)] >> rvec;
        fs_file["projector_tvec_" + std::to_string(i)] >> tvec;
        calib.rvecs[1].emplace_back(rvec); calib.tvecs[1].emplace_back(tvec);
    }

    fs_file["R"] >> calib.R; fs_file["T"] >> calib.T;
    fs_file["E"] >> calib.E; fs_file["F"] >> calib.F;
    fs_file["camera_rms_error"]    >> calib.l_rmse;
    fs_file["projector_rms_error"] >> calib.r_rmse;
    fs_file["stereo_rms_error"]    >> calib.rmse;
    fs_file["l_reproj_avge"]       >> calib.l_reproj_avge;
    fs_file["r_reproj_avge"]       >> calib.r_reproj_avge;
    fs_file.release();

    ConfigManager::getInstance().updateCalibInfo([&](auto& c){ c = calib; });
    return true;
}

// ============================================================
// 加载二次曲面标定信息
// ============================================================
bool loadQuadSurfaceInfo() {
    fs::path filepath = calib_info_dir_ / "quad_surface.yml";
    cv::FileStorage fs_file(filepath, cv::FileStorage::READ);
    if (!fs_file.isOpened()) return false;

    std::vector<QuadSurface> surfaces;
    int surface_count = 0;
    fs_file["surface_count"] >> surface_count;
    cv::FileNode node = fs_file["surfaces"];
    if (node.type() != cv::FileNode::SEQ) return false;
    for (auto it = node.begin(); it != node.end(); ++it) {
        QuadSurface s;
        (*it)["coefficients"] >> s.coefficients;
        (*it)["rmse"]         >> s.rmse;
        surfaces.push_back(s);
    }
    fs_file.release();
    ConfigManager::getInstance().setQuadSurfaces(surfaces);
    return true;
}

// ============================================================
// 生成极线校正重映射表（仅初始化一次）
// ============================================================
void generateRemapMaps(CalibrationResult& calib, cv::Size img_size) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat cm0 = calib.camera_matrix[0].clone();
    cv::Mat cm1 = calib.camera_matrix[1].clone();
    cv::Mat R   = calib.R.clone();
    cv::Mat T   = calib.T.clone();
    cm0.convertTo(cm0, CV_64F); cm1.convertTo(cm1, CV_64F);
    R.convertTo(R, CV_64F);     T.convertTo(T, CV_64F);

    cv::stereoRectify(cm0, calib.dist_coeffs[0], cm1, calib.dist_coeffs[1],
                      img_size, R, T, R1, R2, P1, P2, Q, 0, -1);

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(cm0, calib.dist_coeffs[0], R1, P1, img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(cm1, calib.dist_coeffs[1], R2, P2, img_size, CV_16SC2, map21, map22);

    calib.remap = {{map11.clone(), map12.clone()}, {map21.clone(), map22.clone()}};
    calib.rectify_matrix[0] = P1(cv::Rect(0,0,3,3)).clone();
    calib.rectify_matrix[1] = P2(cv::Rect(0,0,3,3)).clone();
    calib.P[0] = P1.clone(); calib.P[1] = P2.clone();
    calib.rectify_R[0] = R1; calib.rectify_R[1] = R2;
    calib.Q = Q.clone();
}

// ============================================================
// 极线校正（懒初始化重映射表）
// ============================================================
std::array<cv::Mat, 2> getEpipolarRectifyImage(const cv::Mat& img_l, const cv::Mat& img_r) {
    static std::once_flag remap_init_flag;
    cv::Size img_size = img_l.size();
    std::call_once(remap_init_flag, [&img_size]{
        ConfigManager::getInstance().updateCalibInfo([&img_size](auto& c){
            if (c.remap.empty()) generateRemapMaps(c, img_size);
        });
    });

    const auto& calib = ConfigManager::getInstance().getCalibInfo();
    cv::Mat rectify_l, rectify_r;
    cv::remap(img_l, rectify_l, calib.remap[0][0], calib.remap[0][1], cv::INTER_LINEAR);
    cv::remap(img_r, rectify_r, calib.remap[1][0], calib.remap[1][1], cv::INTER_LINEAR);
    return {rectify_l, rectify_r};
}

// ============================================================
// 图像预处理（自适应阈值二值化）
// ============================================================
cv::Mat processImg2(const cv::Mat& img_origin, int /*is_right*/, bool /*have_laser*/) {
    cv::Mat gray;
    if (img_origin.channels() != 1)
        cv::cvtColor(img_origin, gray, cv::COLOR_RGB2GRAY);
    else
        gray = img_origin.clone();

    // 降采样 -> 中值降噪 -> 自适应阈值 -> 上采样
    cv::Mat resized, denoised, binary;
    cv::resize(gray, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::medianBlur(resized, denoised, 3);
    cv::adaptiveThreshold(denoised, binary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          21, -11);
    cv::resize(binary, binary, gray.size(), 0, 0, cv::INTER_NEAREST);
    return binary;
}