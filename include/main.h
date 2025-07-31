#pragma once
#include "configer.h"
#include <atomic>
#include <numeric>
#include <regex>

void loadImgs() {
    // 读取点云检测图片
    std::vector<fs::path> left_imgs, right_imgs;
    for (const auto& entry : fs::directory_iterator(laser_imgs_dir_ / "left"))
        if (entry.is_regular_file()) left_imgs.push_back(entry.path());
    for (const auto& entry : fs::directory_iterator(laser_imgs_dir_ / "right"))
        if (entry.is_regular_file()) right_imgs.push_back(entry.path());
    if (left_imgs.size() != right_imgs.size() || left_imgs.empty())
        throw std::logic_error("Left and right images count mismatch or empty");

    // 文件排序
    auto extract_index = [](const std::filesystem::path& p) -> int {
        std::smatch m;
        std::regex re(R"(image_(\d+))");
        std::string stem = p.stem().string();
        if (std::regex_search(stem, m, re)) {
            return std::stoi(m[1]);
        }
        return -1; // 未匹配到数字，排在前面
    };
    std::sort(left_imgs.begin(), left_imgs.end(), [&](const auto& a, const auto& b) {
        return extract_index(a) < extract_index(b);
    });
    std::sort(right_imgs.begin(), right_imgs.end(), [&](const auto& a, const auto& b) {
        return extract_index(a) < extract_index(b);
    });

    for (size_t i = 0; i < left_imgs.size(); ++i) {
        cv::Mat img_l = cv::imread(left_imgs[i].string(), cv::IMREAD_GRAYSCALE);
        cv::Mat img_r = cv::imread(right_imgs[i].string(), cv::IMREAD_GRAYSCALE);
        if (img_l.empty() || img_r.empty()) {
            throw std::logic_error("Failed to load laser images");
        }
        laser_imgs_.emplace_back(img_l, img_r);
    }
}

void loadTrackImgs() {
    laser_imgs_.clear();  // 清空原有数据

    // 1. 获取所有子文件夹并排序
    std::vector<fs::path> subdirs;
    for (const auto& entry : fs::directory_iterator(laser_imgs_dir_)) {
        if (entry.is_directory()) {
            subdirs.push_back(entry.path());
        }
    }
    
    // 按子文件夹数字名称排序
    std::sort(subdirs.begin(), subdirs.end(), [](const auto& a, const auto& b) {
        try {
            return std::stoi(a.filename()) < std::stoi(b.filename());
        } catch (...) {
            return a.filename() < b.filename(); // 非数字名称按字典序
        }
    });

    // 2. 遍历每个子文件夹处理指定图片
    for (const auto& subdir : subdirs) {
        fs::path img1_path, img2_path;
        bool found_img1 = false;
        bool found_img2 = false;
        
        // 只扫描文件，不递归子目录
        for (const auto& file : fs::directory_iterator(subdir)) {
            if (!file.is_regular_file()) continue;
            
            std::string stem = file.path().stem().string();
            if (stem == "image1") {
                img1_path = file.path();
                found_img1 = true;
            } else if (stem == "image2") {
                img2_path = file.path();
                found_img2 = true;
            }
            
            // 已找到两图则提前退出扫描
            if (found_img1 && found_img2) break;
        }
        
        // 3. 读取找到的图像对
        if (found_img1 && found_img2) {
            cv::Mat img_l = cv::imread(img1_path.string(), cv::IMREAD_GRAYSCALE);
            cv::Mat img_r = cv::imread(img2_path.string(), cv::IMREAD_GRAYSCALE);
            
            if (!img_l.empty() && !img_r.empty()) {
                laser_imgs_.emplace_back(img_l, img_r);
                std::cout << "Loaded image pair from: " << subdir.filename() << std::endl;
            } else {
                std::cerr << "Warning: Failed to load images in " << subdir 
                          << " (" << img1_path << " or " << img2_path << ")" << std::endl;
            }
        } else {
            std::cerr << "Warning: Missing image1 or image2 in " << subdir << std::endl;
        }
    }
    
    // 4. 最终检查
    if (laser_imgs_.empty()) {
        throw std::logic_error("No valid image pairs found in any subdirectory");
    }
    std::cout << "Successfully loaded " << laser_imgs_.size() 
              << " image pairs from " << subdirs.size() << " subdirectories\n";
}

bool loadPlaneCalibInfo() {
    fs::path filepath = calib_info_dir_ / "laser_calib.yml";
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    std::vector<Plane> planes;
    int plane_count = 0;
    fs["plane_count"] >> plane_count;
    cv::FileNode planes_node = fs["planes"];
    if (planes_node.type() != cv::FileNode::SEQ) return false;

    planes.clear();
    for (auto it = planes_node.begin(); it != planes_node.end(); ++it) {
        Plane plane;
        (*it)["normal"] >> plane.normal;
        (*it)["d"] >> plane.d;
        (*it)["mean_err"] >> plane.mean_error;
        (*it)["max_err"] >> plane.max_error;
        planes.push_back(plane);
    }
    fs.release();
    
    ConfigManager::getInstance().setPlane(planes);

    return true;
}

bool loadStereoCalibInfo() {
    fs::path filepath = calib_info_dir_ / "stereo_calib.yml";
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) { return false; }

    CalibrationResult calib_info;
    fs["camera_cameraMatrix"] >> calib_info.camera_matrix[0];
    fs["camera_distCoeffs"] >> calib_info.dist_coeffs[0];
    fs["projector_cameraMatrix"] >> calib_info.camera_matrix[1];
    fs["projector_distCoeffs"] >> calib_info.dist_coeffs[1];

    // 读取相机姿态参数
    int image_count = 0;
    fs["camera_poses_num"] >> image_count;
    for (size_t i = 0; i < image_count; ++i) {
        cv::Mat rvec, tvec;
        fs["camera_rvec_" + std::to_string(i)] >> rvec;
        fs["camera_tvec_" + std::to_string(i)] >> tvec;
        calib_info.rvecs[0].emplace_back(rvec);
        calib_info.tvecs[0].emplace_back(tvec);

        fs["projector_rvec_" + std::to_string(i)] >> rvec;
        fs["projector_tvec_" + std::to_string(i)] >> tvec;
        calib_info.rvecs[1].emplace_back(rvec);
        calib_info.tvecs[1].emplace_back(tvec);
    }

    // 读取外参
    fs["R"] >> calib_info.R;
    fs["T"] >> calib_info.T;
    fs["E"] >> calib_info.E;
    fs["F"] >> calib_info.F;

    // 读取标定质量指标
    fs["camera_rms_error"] >> calib_info.l_rmse;
    fs["projector_rms_error"] >> calib_info.r_rmse;
    fs["stereo_rms_error"] >> calib_info.rmse;
    fs["l_reproj_avge"] >> calib_info.l_reproj_avge;
    fs["r_reproj_avge"] >> calib_info.r_reproj_avge;

    fs.release();
    ConfigManager::getInstance().updateCalibInfo([&](auto& calib) {
        calib = calib_info;
    });

    return true;
}

bool loadQuadSurfaceInfo() {
    fs::path filepath = calib_info_dir_ / "quad_surface.yml";
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    std::vector<QuadSurface> surfaces;
    int surface_count = 0;
    fs["surface_count"] >> surface_count;
    
    cv::FileNode surfaces_node = fs["surfaces"];
    if (surfaces_node.type() != cv::FileNode::SEQ) return false;
    
    for (auto it = surfaces_node.begin(); it != surfaces_node.end(); ++it) {
        QuadSurface surface;
        (*it)["coefficients"] >> surface.coefficients;
        (*it)["rmse"] >> surface.rmse;
        surfaces.push_back(surface);
    }
    fs.release();
    
    ConfigManager::getInstance().setQuadSurfaces(surfaces);

    return true;
}


cv::Mat getGaussImgOnly(const cv::Mat& img, const int& n,
    const double& gsigma, const int& border) {
    int w = img.cols;
    int h = img.rows;

    // 创建边界扩展图像
    cv::Mat gauss_border_img;
    cv::copyMakeBorder(img, gauss_border_img, border, border, border, border, cv::BORDER_REFLECT);

    // 自定义高斯核
    const int small_gaussian_size = 7;
    static const std::vector<std::vector<float>> small_gaussian_tab = {
        { 1.f },
        { 0.25f, 0.5f, 0.25f },
        { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
        { 0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f }
    };

    // 选择固定的高斯核
    std::vector<float> fixed_kernel = {};
    if (n % 2 == 1 && n <= small_gaussian_size && gsigma <= 0) {
        const auto& kernel = small_gaussian_tab[n >> 1];
        fixed_kernel.assign(kernel.begin(), kernel.end());
    }

    // 计算高斯核的长度
    const int length = fixed_kernel.empty() ? small_gaussian_tab[3].size() : fixed_kernel.size();
    std::vector<float> kernel(length);
    double sigmaX = gsigma > 0 ? gsigma : ((length - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;

    // 生成高斯核
    for (int i = 0; i < length; i++) {
        double x = i - (length - 1) * 0.5;
        double t = fixed_kernel.empty() ? exp(scale2X * x * x) : (double)fixed_kernel[i];
        kernel[i] = (float)t;
        sum += kernel[i];
    }

    // 归一化高斯核，保证图像整体亮度不变
    sum = 1.0 / sum;
    for (int i = 0; i < length; i++) {
        kernel[i] *= sum;
    }

    // 应用高斯滤波
    cv::Mat gauss_image = gauss_border_img.clone();
    cv::Mat gauss_result = cv::Mat::zeros(h, w, CV_32F);

    // 水平卷积
    for (int y = 0; y < h + 2 * border; y++) {
        for (int x = 0; x < w + 2 * border; x++) {
            float sum = 0.0f;
            for (int k = -length / 2; k <= length / 2; k++) {
                if (x + k >= 0 && x + k < w + 2 * border) sum += gauss_border_img.at<float>(y, x + k) * kernel[k + length / 2];
            }
            gauss_image.at<float>(y, x) = sum;
        }
    }

    // 垂直卷积
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int k = -length / 2; k <= length / 2; k++) {
                if (y + k >= 0 && y + k < h + 2 * border) sum += gauss_image.at<float>(y + k, x) * kernel[k + length / 2];
            }
            gauss_result.at<float>(y, x) = sum;
        }
    }

    return gauss_result;

}

cv::Mat processImg(const cv::Mat& img_origin, int is_right, bool have_laser) {
    
    // 灰度转换
    cv::Mat gray_img;
    if (img_origin.channels() != 1) {
        cv::cvtColor(img_origin, gray_img, cv::COLOR_RGB2GRAY);
    } else {
        gray_img = img_origin.clone();
    }

    // 数据类型转换
    cv::Mat img_float;
    gray_img.convertTo(img_float, CV_32F);


    // 使用自定义的高斯处理方法
    const int small_gaussian_size = 7;
    const int border_size = (small_gaussian_size - 1) / 2;
    cv::Mat gauss_result = getGaussImgOnly(img_float, 
    small_gaussian_size,  // n=7对应预定义核 
    -1,                   // 强制使用small_gaussian_tab 
    border_size);                // 自动计算边界扩展量

    // 转换精度
    cv::Mat processed_img;
    gauss_result.convertTo(processed_img, CV_8UC1);

    // 形态学降噪（核尺寸自适应图像尺寸）
    const int min_dim = std::min(processed_img.rows,  processed_img.cols); 
    int morph_size = min_dim > 2000 ? 5 :
                        min_dim > 1000 ? 3 : 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(morph_size, morph_size));
    cv::morphologyEx(processed_img, processed_img, cv::MORPH_OPEN, kernel);

    return processed_img;
}

cv::Mat processImg2(const cv::Mat& img_origin, int is_right, bool have_laser) {
    // 1. 灰度转换
    cv::Mat gray_img;
    if (img_origin.channels() != 1) {
        cv::cvtColor(img_origin, gray_img, cv::COLOR_RGB2GRAY);
    } else {
        gray_img = img_origin.clone();
    }

    // 2. 中值滤波，去除反光和斑点噪声，保护边缘
    cv::Mat denoised_img;
    cv::medianBlur(gray_img, denoised_img, 5); // 5x5核，强力去除小噪声

    // 3. 自适应阈值，增强激光线结构
    cv::Mat binary_img;
    cv::adaptiveThreshold(denoised_img, binary_img, 255,
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                         31, -17); // 21为窗口，10为偏置，可根据实际调整

    // 4. 形态学开运算，去除小噪声，保护细线
    int morph_size = std::min(binary_img.rows, binary_img.cols) > 2000 ? 3 : 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_size, morph_size));
    cv::morphologyEx(binary_img, binary_img, cv::MORPH_OPEN, kernel);

    // 5. 返回处理结果
    return binary_img;
}


std::array<cv::Mat, 2> getEpipolarRectifyImage(
    const cv::Mat& img_l, 
    const cv::Mat& img_r
) {
    static std::once_flag remap_init_flag;
    cv::Size img_size = img_l.size();
    std::call_once(remap_init_flag, [&img_size]{ // 无捕获，避免潜在的数据竞争
        ConfigManager::getInstance().updateCalibInfo([&img_size](auto& mutable_calib){
            if (mutable_calib.remap.empty()) {
                // 直接通过写接口操作最新数据
                generateRemapMaps(mutable_calib, img_size);
            }
        });
    });

    // 获取标定数据
    const auto& safe_calib = ConfigManager::getInstance().getCalibInfo();
    cv::Mat rectify_img_l, rectify_img_r;
    cv::remap(img_l, rectify_img_l, safe_calib.remap[0][0], safe_calib.remap[0][1], cv::INTER_LINEAR);
    cv::remap(img_r, rectify_img_r, safe_calib.remap[1][0], safe_calib.remap[1][1], cv::INTER_LINEAR);

    return {rectify_img_l, rectify_img_r};
}

void generateRemapMaps(CalibrationResult& calib, cv::Size img_size) {
    cv::Mat R1, R2, P1, P2, Q;
    
    // 类型转换（创建临时副本）
    cv::Mat camera_matrix[2] = {
        calib.camera_matrix[0].clone(),
        calib.camera_matrix[1].clone()
    };
    cv::Mat R = calib.R.clone();
    cv::Mat T = calib.T.clone();

    // 确保64F精度
    camera_matrix[0].convertTo(camera_matrix[0], CV_64F);
    camera_matrix[1].convertTo(camera_matrix[1], CV_64F);
    R.convertTo(R, CV_64F);
    T.convertTo(T, CV_64F);

    // 计算校正参数
    cv::stereoRectify(camera_matrix[0], calib.dist_coeffs[0],
                    camera_matrix[1], calib.dist_coeffs[1],
                    img_size, R, T, R1, R2, P1, P2, Q, 0, -1);

    // 生成映射表
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(camera_matrix[0], calib.dist_coeffs[0],
                              R1, P1, img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(camera_matrix[1], calib.dist_coeffs[1],
                              R2, P2, img_size, CV_16SC2, map21, map22);

    // 原子化更新
    calib.remap = {
        {map11.clone(), map12.clone()},
        {map21.clone(), map22.clone()}
    };
    calib.rectify_matrix[0] = P1(cv::Rect(0, 0, 3, 3)).clone();
    calib.rectify_matrix[1] = P2(cv::Rect(0, 0, 3, 3)).clone();
    calib.P[0] = P1.clone();
    calib.P[1] = P2.clone();
    calib.rectify_R[0] = R1, calib.rectify_R[1] = R2;
    calib.Q = Q.clone();
}


std::pair<cv::Point2f, cv::Point2f> getAxisEndpoints(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    // 计算所有相邻边的长度并识别最短边
    float min_len = FLT_MAX;
    int short_edge_idx = 0;
    for (int i = 0; i < 4; ++i) {
        float len = cv::norm(vertices[i] - vertices[(i+1)%4]);
        if (len < min_len) {
            min_len = len;
            short_edge_idx = i;
        }
    }

    // 确定对应的两条短边（对边）
    int opposite_idx = (short_edge_idx + 2) % 4;
    
    // 计算两条短边的中点
    cv::Point2f mid1 = 0.5f * (vertices[short_edge_idx] + vertices[(short_edge_idx+1)%4]);
    cv::Point2f mid2 = 0.5f * (vertices[opposite_idx] + vertices[(opposite_idx+1)%4]);

    return std::make_pair(mid1, mid2);
}

Plane fitPlane(std::vector<cv::Point3f> points) {
    Plane plane;

    cv::Point3f mean = std::accumulate(points.begin(), points.end(), cv::Point3f(0.0f,0.0f,0.0f));
    mean *= (1.0f / points.size());

    // 2. 构建协方差矩阵
    cv::Matx33f cov = cv::Matx33f::zeros();
    for (const auto& p : points) {
        cv::Vec3f d = cv::Vec3f(p.x - mean.x, p.y - mean.y, p.z - mean.z);
        cov += d * d.t();
    }

    // 3. SVD 求解最小特征值对应的特征向量（法向量）
    cv::Matx33f cov_mat = cov / static_cast<float>(points.size());
    cv::Mat w, u, vt;
    cv::SVD::compute(cv::Mat(cov_mat), w, u, vt);
    cv::Vec3f normal(u.at<float>(0,2), u.at<float>(1,2), u.at<float>(2,2));
    normal = cv::normalize(normal);

    // 4. 计算 d
    float d = -normal.dot(cv::Vec3f(mean.x, mean.y, mean.z));

    // 5. 计算误差
    float sum_err = 0, max_err = 0;
    for (const auto& p : points) {
        float err = std::abs(normal[0]*p.x + normal[1]*p.y + normal[2]*p.z + d);
        sum_err += err;
        if (err > max_err) max_err = err;
    }
    plane.normal = normal;
    plane.d = d;
    plane.mean_error = sum_err / points.size();
    plane.max_error = max_err;

    return plane;

}