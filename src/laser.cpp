#include "laser.h"
#include <fstream>
#define DEBUG_PLANE_MATCH

/************************************** Extract Centers **************************************/
int LaserProcessor::convert_to_odd_number(float num) {
    int rounded = static_cast<int>(std::round(num));  // 四舍五入到整数 
    
    if (rounded % 2 != 0) {
        return rounded;  // 若结果本身是奇数，直接返回 
    } else {
        float dist_prev = std::abs(num - (rounded - 1));  // 与下一个较小奇数的距离 
        float dist_next = std::abs(num - (rounded + 1));  // 与下一个较大奇数的距离 
        
        // 选择更近的奇数（距离相同则取较大的）
        if (dist_prev < dist_next) {
            return rounded - 1;
        } else if (dist_next < dist_prev) {
            return rounded + 1;
        } else {
            return rounded + 1;  // 距离相等时优先较大值 
        }
    }
}

std::tuple<cv::Mat, cv::Mat, int> LaserProcessor::computeGaussianDerivatives(float sigma, float angle_rad, bool h_is_long_edge) {
    int ksize = cvRound(sigma * 3 * 2) + 1;
    ksize = std::max(ksize, 3) > 31 ? 31 : std::max(ksize, 3); // 限制最大核尺寸
    ksize = ksize % 2 == 0 ? ksize + 1 : ksize; // 确保奇数
    
    // 创建标准高斯导数核
    cv::Mat dx_std, dy_std;
    cv::getDerivKernels(dx_std, dy_std, 1, 0, ksize, true, CV_32F);
    cv::multiply(dx_std, -1.0/(sigma*sigma), dx_std);
    
    cv::Mat dx_temp, dy_temp;
    cv::getDerivKernels(dx_temp, dy_temp, 0, 1, ksize, true, CV_32F);
    cv::multiply(dy_temp, -1.0/(sigma*sigma), dy_temp);
    
    // 计算旋转角度的正弦和余弦
    float vx, vy;
    if (h_is_long_edge) {
        // θ 是从 y 轴到长边 ⇒ 主轴方向为 (sinθ, -cosθ)
        vx = std::sin(angle_rad);
        vy = -std::cos(angle_rad);
    } else {
        // θ 是从 y 轴到短边 ⇒ 主轴方向为 (cosθ, sinθ)
        vx = std::cos(angle_rad);
        vy = std::sin(angle_rad);
    }
    
    // 创建旋转后的核
    cv::Mat dx_kernel = dx_std * vx + dy_temp * vy;
    cv::Mat dy_kernel = -dx_std * vy + dy_temp * vx;

    return {dx_kernel, dy_kernel ,ksize};
}

float LaserProcessor::interpolateChannel(const cv::Mat& img, float x, float y) {
    int xi = std::clamp(static_cast<int>(x), 0, img.cols-2);
    int yi = std::clamp(static_cast<int>(y), 0, img.rows-2);
    
    float dx = x - xi;
    float dy = y - yi;
    
    const float* row0 = img.ptr<float>(yi);
    const float* row1 = img.ptr<float>(yi+1);
    
    float a = row0[xi];
    float b = row0[xi+1];
    float c = row1[xi];
    float d = row1[xi+1];
    
    return a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy;
}

float LaserProcessor::findSymmetricCenter(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range) {
    // 阶段1：梯度预检测快速定位
    const float coarse_step = 0.2f;
    float max_grad = -FLT_MAX;
    float peak_t = 0;
    
    // 梯度检测（中心差分法）
    for (float t = -search_range; t <= search_range; t += coarse_step) {
        float val_prev = interpolateChannel(img, x + (t-0.1f)*dir[0], y + (t-0.1f)*dir[1]);
        float val_next = interpolateChannel(img, x + (t+0.1f)*dir[0], y + (t+0.1f)*dir[1]);
        float grad = val_next - val_prev;
        if (grad > max_grad) {
            max_grad = grad;
            peak_t = t;
        }
    }

    // 阶段2：精细采样
    const float fine_range = search_range * 0.6f; // 根据线宽自适应
    std::vector<std::pair<float, float>> profile;
    profile.reserve(static_cast<int>(2*fine_range/0.1f) + 2);
    
    // 获取当前剖面数据并计算max_val
    float max_val = -FLT_MAX;
    for (float t = peak_t - fine_range; t <= peak_t + fine_range; t += 0.02f) {
        float val = interpolateChannel(img, x + t*dir[0], y + t*dir[1]);
        profile.emplace_back(t, val);
        if (val > max_val) max_val = val;
    }

    // 方法A：梯度对称点检测（一阶导数过零点）
    float sym_center = FLT_MAX;
    {
        std::vector<float> deriv1;
        for (size_t i = 1; i < profile.size()-1; ++i) {
            deriv1.push_back( (profile[i+1].second - profile[i-1].second)/0.2f ); // 中心差分
        }
        
        // 寻找正到负的过零点
        for (size_t i = 1; i < deriv1.size(); ++i) {
            if (deriv1[i-1] > 0 && deriv1[i] < 0) {
                // 线性插值求精确过零点
                float alpha = deriv1[i] / (deriv1[i] - deriv1[i-1]);
                sym_center = profile[i].first - 0.1f*alpha;
                break;
            }
        }
    }

    // 方法B：加权质心法（动态阈值）
    float centroid = FLT_MAX;
    {
        const float threshold = max_val * 0.7f; // 自适应阈值
        float sum_wt = 0.0f, sum_w = 0.0f;
        
        for (const auto& p : profile) {
            if (p.second < threshold) continue;
            float w = p.second - threshold;
            sum_wt += w * p.first;
            sum_w += w;
        }
        
        if (sum_w > 1e-6f) {
            centroid = sum_wt / sum_w;
        }
    }

    // 结果融合策略
    float final_center = FLT_MAX;
    
    // 情况1：两种方法均有效时取加权平均
    if (sym_center != FLT_MAX && centroid != FLT_MAX) {
        if (std::abs(sym_center - centroid) < 0.2f) { // 结果一致性检查
            final_center = 0.4f*sym_center + 0.6f*centroid;
        } else {
            final_center = centroid; // 优先质心法
        }
    }
    // 情况2：仅一种方法有效
    else if (centroid != FLT_MAX) {
        final_center = centroid;
    } else if (sym_center != FLT_MAX) {
        final_center = sym_center;
    }

    return final_center;
}

std::vector<cv::Point2f> LaserProcessor::processCenters(const std::map<float, float>& orign_centers) {
    std::vector<cv::Point2f> new_centers;
    if (orign_centers.empty()) return new_centers;

    static auto isInteger = [](float y) {
        return std::abs(y - std::round(y)) < 1e-5;
    };

    auto itA = orign_centers.begin();
    // 先处理第一个点 A 是否整数 y
    if (isInteger(itA->first)) {
        new_centers.emplace_back(itA->second, itA->first);
    }

    // 遍历相邻对 (A, B)
    for (auto itB = std::next(orign_centers.begin()); itB != orign_centers.end(); ++itB, ++itA) {
        float yA = itA->first, xA = itA->second;
        float yB = itB->first, xB = itB->second;

        // 确保 A 在前，B 在后
        if (yB <= yA) continue;

        // 向上取整得到第一个可能的整数 y 值
        float yI = std::ceil(yA);
        // 检查这个整数是否在 (yA, yB] 区间内
        if (yI > yA && yI <= yB) {
            // 如果 yI 恰好等于 B.y 并且 B.y 本身就是整数，直接使用 B 点
            if (isInteger(yB) && std::fabs(yI - yB) < 1e-6f) {
                new_centers.emplace_back(xB, yB);
            } else {
                // 否则进行线性插值
                float t = (yI - yA) / (yB - yA);
                float xI = xA + t * (xB - xA);
                new_centers.emplace_back(xI, yI);
            }
        }
        // 如果 yI > yB，则说明区间内无整数；此时若 B.y 本身是整数，也可以直接加入
        else if (isInteger(yB)) {
            new_centers.emplace_back(xB, yB);
        }
    }

    return new_centers;
}

std::pair<cv::Point2f, cv::Point2f> LaserProcessor::getAxisEndpoints(const cv::RotatedRect& rect) {
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

std::vector<LaserLine> LaserProcessor::extractLine(const std::vector<cv::RotatedRect>& rois, const cv::Mat& rectify_img) {
    std::vector<LaserLine> laser_lines;
    cv::Mat rectify_img_float;
    rectify_img.convertTo(rectify_img_float, CV_32F, 1.0f / 255.0f);
    
    // 1. 整体二值化
    cv::Mat bin;
    cv::threshold(rectify_img, bin, 100, 255, cv::THRESH_BINARY);
    
    for (size_t i = 0; i < rois.size(); ++i) {
        const auto& roi = rois[i];
        const auto& roi_w = std::min(roi.size.width, roi.size.height);

        // 2. 获取ROI四个顶点
        cv::Point2f vertices[4];
        roi.points(vertices);

        // 3. 创建mask
        cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8UC1);
        std::vector<cv::Point> roi_poly;
        for (int i = 0; i < 4; ++i)
            roi_poly.push_back(vertices[i]);
        std::vector<std::vector<cv::Point>> polys = {roi_poly};
        cv::fillPoly(mask, polys, cv::Scalar(255));

        // 4. 提取ROI区域
        cv::Mat roi_bin;
        bin.copyTo(roi_bin, mask);

        // 5. 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        // 可视化contour
        cv::Mat contour_vis;
        cv::cvtColor(rectify_img, contour_vis, cv::COLOR_GRAY2BGR);
        cv::drawContours(contour_vis, contours, -1, cv::Scalar(0, 0, 255), 2); // 红色轮廓

        // 6. 收集轮廓点
        if (contours.size() != 1) throw std::logic_error("have many contours");
        std::vector<cv::Point2f> edge_pixels;
        auto [p1, p2] = getAxisEndpoints(roi);
        cv::Point2f axis = p2 - p1;
        float axis_len = cv::norm(axis);
        cv::Point2f axis_dir = axis / axis_len;
        float min_proj = 0.02f * axis_len;
        float max_proj = 0.98f * axis_len;
        for (const auto& pt : contours[0]) {
            cv::Point2f pt_f(pt.x, pt.y);
            cv::Point2f vec = pt_f - p1;
            float proj = vec.dot(axis_dir);
            if (proj >= min_proj && proj <= max_proj) {
                edge_pixels.push_back(pt);
            }
        }
        // 可视化方向边缘
        cv::Mat fix_contour_vis;
        cv::cvtColor(rectify_img, fix_contour_vis, cv::COLOR_GRAY2BGR);
        for (const auto& p : edge_pixels) cv::circle(fix_contour_vis, p, 1, cv::Scalar(0, 0, 255), -1);

        // 7. 垂直剖解激光线
        auto is_toin = [&](const cv::Point2f& pt) -> bool {
            int x = cvRound(pt.x);
            int y = cvRound(pt.y);

            // 边界检查
            if (y - 2 < 0 || y + 2 >= bin.rows) return false;

            int val_up   = bin.at<uchar>(y - 2, x);
            int val_down = bin.at<uchar>(y + 2, x);

            // 如果往255区域为向里，往0区域为向外
            return val_down > val_up;
        };
        cv::Mat direct_vis;
        cv::cvtColor(rectify_img, direct_vis, cv::COLOR_GRAY2BGR);
        for (const auto& p : edge_pixels)
            if (is_toin(p)) cv::circle(direct_vis, p, 1, cv::Scalar(255, 0, 0), -1);
            else cv::circle(direct_vis, p, 1, cv::Scalar(0, 255, 0), -1);

        cv::Mat laser_center_vis;
        cv::cvtColor(rectify_img, laser_center_vis, cv::COLOR_GRAY2BGR);
        std::map<float, float> orign_centers;
        for (const auto& p : edge_pixels) {
            if (!is_toin(p)) continue;
            cv::Vec2f dir(0, 1);
            float t_peak = findSymmetricCenter(rectify_img_float, p.x, p.y, dir, roi_w);
            // float t_peak = findSymmetricCenter2(rectify_img_float, p.x, p.y, dir, roi_w, 0);
            if (t_peak == FLT_MAX) continue;    // 说明抛物线拟合失败，可能是search_range极端边缘点
            float center_x = p.x + t_peak*dir[0], center_y = p.y + t_peak * dir[1];
            orign_centers[center_y] = center_x;

            // 可视化激光线中心点
            cv::Point2f center(center_x, center_y);
            cv::circle(laser_center_vis, center, 1, cv::Scalar(0, 255, 0), -1);
        }

        // 8. 同一条线相邻中心点插值为整数
        cv::Mat new_centers_vis;
        cv::cvtColor(rectify_img, new_centers_vis, cv::COLOR_GRAY2BGR);
        std::vector<cv::Point2f> new_centers = processCenters(orign_centers);
        for (const auto& p : new_centers) cv::circle(new_centers_vis, p, 1, cv::Scalar(0, 255, 0), -1);

        // 9. 存储结果
        const float laser_width = std::min(roi.size.width, roi.size.height) / roi_scale_;
        const float sigma_val = convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f)));
        const float roi_angle = roi.angle * CV_PI / 180.0f;
        const bool h_is_long_edge = roi.size.height >= roi.size.width;
        auto [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val, roi_angle, h_is_long_edge);
        cv::Mat dx, dy;
        cv::filter2D(rectify_img_float, dx, CV_32F, dx_kernel);
        cv::filter2D(rectify_img_float, dy, CV_32F, dy_kernel);
        std::map<float, LaserPoint> best_points;
        for (const auto& p : new_centers) {
            float gx = interpolateChannel(dx, p.x, p.y);
            float gy = interpolateChannel(dy, p.x, p.y);
            best_points[p.y] = {p.x, p.y, gx, gy};
        }
        LaserLine best_line;
        best_line.addPoints(best_points);
        laser_lines.emplace_back(best_line);
    }

    return laser_lines;
}
/*********************************************************************************** */


/************************************** Match Laser **************************************/
void LaserProcessor::match(
    const std::vector<std::map<float, float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l, const cv::Mat& rectify_r) {

    const auto calib_info = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getPlane();

    const double fx_l = calib_info.P[0].at<double>(0, 0);
    const double fy_l = calib_info.P[0].at<double>(1, 1);
    const double cx_l = calib_info.P[0].at<double>(0, 2);
    const double cy_l = calib_info.P[0].at<double>(1, 2);
    const double fx_r = calib_info.P[1].at<double>(0, 0);
    const double fy_r = calib_info.P[1].at<double>(1, 1);
    const double cx_r = calib_info.P[1].at<double>(0, 2);
    const double cy_r = calib_info.P[1].at<double>(1, 2);
    const double baseline = -calib_info.P[1].at<double>(0, 3) / fx_r;

    for (int l_laser_idx = 0; l_laser_idx < static_cast<int>(sample_points.size()); ++l_laser_idx) {
        const auto& l_points = sample_points[l_laser_idx];

#ifdef DEBUG_PLANE_MATCH
        cv::Mat vis_img;
        cv::hconcat(rectify_l, rectify_r, vis_img);
        if (vis_img.channels() == 1)
            cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
        const int cols = rectify_l.cols;
        cv::circle(vis_img, cv::Point(cx_l, cy_l), 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(vis_img, cv::Point(cx_r + cols, cy_r), 4, cv::Scalar(0, 0, 255), -1);
#endif
        if (l_laser_idx == 3)
            puts("");

        MatchMap match_map;
        for (const auto& [y, x] : l_points) {
#ifdef DEBUG_PLANE_MATCH
            cv::circle(vis_img, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
#endif
            cv::Point3f ray_dir((x - cx_l) / fx_l, (y - cy_l) / fy_l, 1.0f);
            ray_dir *= 1.0f / cv::norm(ray_dir);

            for (int plane_idx = 0; plane_idx < static_cast<int>(planes.size()); ++plane_idx) {
                const auto& plane = planes[plane_idx];
                if (!plane.isValid()) continue;

                const cv::Vec3f& normal = plane.normal;
                float d = plane.d;
                float cos_theta = ray_dir.dot(normal);
                if (std::abs(cos_theta) < 1e-6f) continue;

                // t = ((P_s - P_c) · N_s) / (D_c · N_s)，nP_s + d = 0， P_c = 0
                float t = -d / cos_theta;

                cv::Point3f pt = ray_dir * t;
                float err = pt.x*normal[0] + pt.y*normal[1] + pt.z * normal[2] + d;
                float err2 = 135.565002*normal[0] + -11.041595*normal[1] + 311.500031*normal[2] + d;

                cv::Point3f pt_r(pt.x - baseline, pt.y, pt.z);

                // cv::Mat pt_hom = (cv::Mat_<double>(4,1) << pt.x, pt.y, pt.z, 1.0);
                // cv::Mat pt_img = calib_info.P[0] * pt_hom;
                // cv::Point2f pt_rr(pt_img.at<double>(0,0) / pt_img.at<double>(2,0),
                //                   pt_img.at<double>(1,0) / pt_img.at<double>(2,0));

                cv::Point3f test_dir = pt * (1.0f / cv::norm(pt));

                 if (plane_idx == 3 && l_laser_idx == 3) {
                    std::ofstream ofs("point3d.txt", std::ios::app);
                    if (ofs.is_open()) {
                        ofs << pt.x << " " << pt.y << " " << pt.z << std::endl;
                        ofs.close();
                    }
                }

                float x_r = fx_r * pt_r.x / pt_r.z + cx_r;
                float y_r = fy_r * pt_r.y / pt_r.z + cy_r;

                if (x_r < 0 || x_r >= rectify_r.cols || y_r < 0 || y_r >= rectify_r.rows) continue;

#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(cvRound(x_r) + cols, cvRound(y_r)), 2, cv::Scalar(0, 0, 255), -1);
#endif

                for (int r_laser_idx = 0; r_laser_idx < static_cast<int>(laser_r.size()); ++r_laser_idx) {
                    const auto& r_line = laser_r[r_laser_idx].points;
                    if (r_line.empty()) continue;

                    auto it = r_line.lower_bound(y_r);
                    float min_dist = FLT_MAX;

                    for (int j = 0; j < 3; ++j) {
                        auto check_it = it;
                        if (j == 0 && it != r_line.begin()) check_it = std::prev(it);
                        else if (j == 2 && it != r_line.end()) check_it = std::next(it);
                        if (check_it == r_line.end()) continue;

                        const auto& pt_check = check_it->second;
                        float dist = std::hypot(pt_check.x - x_r, check_it->first - y_r);
                        min_dist = std::min(min_dist, dist);
                    }

                    if (min_dist < 3.0f) {
                        MatchKey key{l_laser_idx, plane_idx, r_laser_idx};
                        float score = 1.0f / (min_dist + 1.0f);
                        match_map[key].score_sum += score;
                        match_map[key].count += 1;
                    }
                }
            }
        }

        // 寻找最佳匹配
        int best_r_idx = -1;
        int best_plane_idx = -1;
        float best_score = -1.0f;
        const int n_samples = static_cast<int>(l_points.size());

        for (const auto& [key, result] : match_map) {
            if (key.l_laser_idx != l_laser_idx) continue;
            if (result.count < n_samples / 10) continue;

            const auto& r_points = laser_r[key.r_laser_idx].points;
            float unit_score = result.score_sum / std::max(1, static_cast<int>(r_points.size()));
            float coverage = result.count / float(n_samples);
            float combined = unit_score * coverage;
            // float f1 = 2.0f * result.count / (n_samples + float(static_cast<int>(r_points.size())));

            if (combined > best_score) {
                best_score = combined;
                best_plane_idx = key.plane_idx;
                best_r_idx = key.r_laser_idx;
            }
        }

#ifdef DEBUG_PLANE_MATCH
        if (best_r_idx >= 0 && best_r_idx < static_cast<int>(laser_r.size())) {
            const auto& best_line = laser_r[best_r_idx].points;
            for (const auto& [y, pt] : best_line) {
                cv::circle(vis_img, cv::Point(cvRound(pt.x + cols), cvRound(y)), 2, cv::Scalar(0, 255, 0), -1);
            }
            if (!l_points.empty() && !best_line.empty()) {
                auto it_l = l_points.begin();
                std::advance(it_l, l_points.size() / 2);
                auto it_r = best_line.begin();
                std::advance(it_r, best_line.size() / 2);
                cv::line(vis_img, 
                    cv::Point(cvRound(it_l->second), cvRound(it_l->first)),
                    cv::Point(cvRound(it_r->second.x + cols), cvRound(it_r->first)),
                    cv::Scalar(0, 255, 255), 2);
            }
        }

        std::string text = (best_r_idx >= 0) ?
            "Best Match: L" + std::to_string(l_laser_idx) +
            " - R" + std::to_string(best_r_idx) +
            " (Plane " + std::to_string(best_plane_idx) + ")" :
            "Best Match: None";

        cv::putText(vis_img, text,
            cv::Point((vis_img.cols - 400) / 2, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        cv::namedWindow("Sample Points Projection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Sample Points Projection", vis_img.cols, vis_img.rows);
        cv::imshow("Sample Points Projection", vis_img);
        if (cv::waitKey(0) == 27) break;  // ESC 跳出所有可视化
#endif
    }
}

/*************************************************************************************** */