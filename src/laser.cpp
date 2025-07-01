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
                new_centers.emplace_back(xB, std::round(yB));
            } else {
                // 否则进行线性插值
                float t = (yI - yA) / (yB - yA);
                float xI = xA + t * (xB - xA);
                new_centers.emplace_back(xI, std::round(yI));
            }
        }
        // 如果 yI > yB，则说明区间内无整数；此时若 B.y 本身是整数，也可以直接加入
        else if (isInteger(yB)) {
            new_centers.emplace_back(xB, std::round(yB));
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
    cv::threshold(rectify_img, bin, 80, 255, cv::THRESH_BINARY);
    
    for (size_t i = 0; i < rois.size(); ++i) {
        const auto& roi = rois[i];
        const auto& roi_w = std::min(roi.size.width, roi.size.height);

        // 2. 获取ROI四个顶点
        cv::Point2f vertices[4];
        roi.points(vertices);

        // 3. 创建mask
        CV_Assert(bin.type() == CV_8UC1);
        cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8UC1);
        std::vector<cv::Point> roi_poly;
        for (int i = 0; i < 4; ++i) {
            roi_poly.push_back(cv::Point(
                cvRound(vertices[i].x), 
                cvRound(vertices[i].y)
            ));
        }
        // cv::Mat bin_with_roi = bin.clone();
        // cv::polylines(bin_with_roi, roi_poly, true, cv::Scalar(0, 255, 0), 2);
        // for (int j = 0; j < roi_poly.size(); j++) {
        //     cv::circle(bin_with_roi, roi_poly[j], 3, cv::Scalar(0, 0, 255), -1);
        // }
        // cv::imwrite("debug_img/roi_" + std::to_string(i) + "_on_bin.jpg", bin_with_roi);
        std::vector<std::vector<cv::Point>> polys = {roi_poly};
        cv::fillPoly(mask, polys, cv::Scalar(255));

        // 4. 提取ROI区域
        cv::Mat roi_bin = cv::Mat::zeros(bin.size(), bin.type());
        bin.copyTo(roi_bin, mask);

        // cv::Mat label_img, color_img;
        // Two_PassNew(roi_bin, label_img);
        // LabelColor(label_img, color_img);
        // cv::Mat copy_rectify_img = rectify_img.clone();
        
        // // 收集所有连通区域点集
        // int max_size_idx = -1, region_max_size = 0;
        // std::map<int, std::vector<cv::Point>> regionPoints;
        // for (int y = 0; y < label_img.rows; ++y) {
        //     const int* row = label_img.ptr<int>(y);
        //     for (int x = 0; x < label_img.cols; ++x) {
        //         int label = row[x];
        //         if (label > 1) // 忽略背景(0)和未处理区域(1)
        //             regionPoints[label].emplace_back(x, y);
        //     }
        // }
        // for (const auto& [key, val] : regionPoints) {
        //     if (val.size() > region_max_size) {
        //         region_max_size = val.size();
        //         max_size_idx = key;
        //     }
        // }

        // // 覆盖原图
        // cv::Mat copy_rectify_img_float;
        // if (regionPoints.size() != 1) {
        //     for (const auto& [key, val] : regionPoints) {
        //         if (key == max_size_idx && key > 1) continue;
        //         for (const auto& p : val) {
        //             copy_rectify_img.at<uchar>(p.y, p.x) = 0;
        //         }
        //     }
        //     copy_rectify_img.convertTo(copy_rectify_img_float, CV_32F, 1.0f / 255.0f);
        // }
        // else copy_rectify_img_float = rectify_img_float.clone();

        

        // 5. 查找轮廓 
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roi_bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        // 可视化contour
        cv::Mat contour_vis;
        cv::cvtColor(rectify_img, contour_vis, cv::COLOR_GRAY2BGR);
        cv::drawContours(contour_vis, contours, -1, cv::Scalar(0, 0, 255), 2); // 红色轮廓

        // 6. 收集轮廓点
        if (contours.empty()) throw std::logic_error("no contours found");
        auto max_it = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return a.size() < b.size();
            });
        std::vector<std::vector<cv::Point>> max_contour{*max_it};
        std::vector<cv::Point2f> edge_pixels;
        auto [p1, p2] = getAxisEndpoints(roi);
        cv::Point2f axis = p2 - p1;
        float axis_len = cv::norm(axis);
        cv::Point2f axis_dir = axis / axis_len;
        float min_proj = 0.06f * axis_len;
        float max_proj = 0.94f * axis_len;
        for (const auto& pt : max_contour[0]) {
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
        cv::Mat rectify_r_img = rectify_r.clone();
        cv::cvtColor(rectify_r_img, rectify_r_img, cv::COLOR_GRAY2BGR);
        cv::Mat vis_img;
        cv::hconcat(rectify_l, rectify_r, vis_img);
        if (vis_img.channels() == 1)
            cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
        const int cols = rectify_l.cols;
        cv::circle(vis_img, cv::Point(cx_l, cy_l), 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(vis_img, cv::Point(cx_r + cols, cy_r), 4, cv::Scalar(0, 0, 255), -1);
#endif

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

                cv::Point3f pt_r(pt.x - baseline, pt.y, pt.z);
                float x_r = fx_r * pt_r.x / pt_r.z + cx_r;
                float y_r = fy_r * pt_r.y / pt_r.z + cy_r;

                
                if (x_r < 0 || x_r >= rectify_r.cols || y_r < 0 || y_r >= rectify_r.rows) continue;

#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(cvRound(x_r) + cols, cvRound(y_r)), 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(rectify_r_img, cv::Point(x_r, y_r), 1, cv::Scalar(0, 0, 255), -1);
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
                        float dist = std::hypot(pt_check.x - x_r, pt_check.y - y_r);
                        min_dist = std::min(min_dist, dist);
                    }
                    if (min_dist == FLT_MAX) continue;

                    MatchKey key{l_laser_idx, plane_idx, r_laser_idx};
                    float score = 1 / (min_dist + 1.0f);
                    match_map[key].score_sum += score*score;
                    match_map[key].count += 1;

                    // if (l_laser_idx == 6 && (r_laser_idx == 8 || r_laser_idx == 10))
                    //     printf("point(%f, %f) - r_idx_%d - p_idx_%d - score: %f\n", x, y, r_laser_idx, plane_idx, score);
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
            if (result.count < n_samples / 10) continue;  // 最小覆盖阈值

            // 计算左激光线上所有采样点的平均得分
            float avg_score = sqrt(result.score_sum / n_samples);

            if (avg_score > best_score) {
                best_score = avg_score;
                best_plane_idx = key.plane_idx;
                best_r_idx = key.r_laser_idx;
            }

            if (l_laser_idx == 6 && (key.r_laser_idx == 8 || key.r_laser_idx == 10) && (key.plane_idx == 4 || key.plane_idx == 3))
                printf("L%d - R%d - P%d - Score %f\n", l_laser_idx, key.r_laser_idx, key.plane_idx, avg_score);
        }

#ifdef DEBUG_PLANE_MATCH
        // 1. 可视化所有候选匹配线
        for (const auto& [key, result] : match_map) {
            if (key.l_laser_idx != l_laser_idx) continue;
            if (result.count < n_samples / 10) continue;

            const auto& r_points = laser_r[key.r_laser_idx].points;
            float unit_score = result.score_sum / std::max(1, static_cast<int>(r_points.size()));
            float coverage = result.count / float(n_samples);
            float combined = unit_score * coverage;

            // 匹配线颜色：最佳匹配高亮，其他为淡绿色
            cv::Scalar line_color = (key.r_laser_idx == best_r_idx) ? cv::Scalar(128, 0, 128) : cv::Scalar(0, 180, 0);

            // 画点
            for (const auto& [y, pt] : r_points) {
                cv::circle(vis_img, cv::Point(cvRound(pt.x + cols), cvRound(y)), 2, line_color, -1);
            }

            // 计算中心点
            if (!r_points.empty()) {
                float sum_x = 0, sum_y = 0;
                for (const auto& [y, pt] : r_points) {
                    sum_x += pt.x + cols;
                    sum_y += y;
                }
                int center_x = cvRound(sum_x / r_points.size());
                int center_y = cvRound(sum_y / r_points.size());

                // 标注r_laser_idx
                cv::putText(vis_img, "R" + std::to_string(key.r_laser_idx),
                    cv::Point(center_x, center_y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.4, line_color, 2);
            }
        }

        // 2. 可视化最佳匹配连线
        if (best_r_idx >= 0 && best_r_idx < static_cast<int>(laser_r.size())) {
            const auto& best_line = laser_r[best_r_idx].points;
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

        // 3. 显示最佳匹配文本
        std::string text = (best_r_idx >= 0) ?
            "Best Match: L" + std::to_string(l_laser_idx) +
            " - R" + std::to_string(best_r_idx) +
            " (P " + std::to_string(best_plane_idx) + " / S " + std::to_string(best_score) + ")" :
            "Best Match: None";

        cv::putText(vis_img, text,
            cv::Point((vis_img.cols - 500) / 2, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        cv::namedWindow("Sample Points Projection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Sample Points Projection", vis_img.cols, vis_img.rows);
        cv::imshow("Sample Points Projection", vis_img);
        if (cv::waitKey(0) == 27) break;  // ESC 跳出所有可视化
#endif
    }
}



std::vector<cv::Point3f> LaserProcessor::findIntersection(const cv::Point3f &point, const cv::Point3f &normal,
                                          const cv::Mat &Coeff6x1)
{
  float a = Coeff6x1.at<float>(0);
  float b = Coeff6x1.at<float>(1);
  float c = Coeff6x1.at<float>(2);
  float d = Coeff6x1.at<float>(3);
  float e = Coeff6x1.at<float>(4);
  float f = Coeff6x1.at<float>(5);

  std::vector<cv::Point3f> intersections;

  // 计算二次方程的系数
  float A = a * normal.y * normal.y + b * normal.y * normal.z + c * normal.z * normal.z;

  float B = 2 * a * point.y * normal.y + b * (point.y * normal.z + point.z * normal.y)
            + 2 * c * point.z * normal.z + d * normal.y + e * normal.z - normal.x;

  float C = a * point.y * point.y + b * point.y * point.z + c * point.z * point.z + d * point.y
            + e * point.z + f - point.x;

  // 解二次方程 A*t² + B*t + C = 0
  float discriminant = B * B - 4 * A * C;

  if (std::abs(A) < 1e-6)
  {  // 线性情况
    if (std::abs(B) > 1e-6)
    {
      float t = -C / B;
      intersections.push_back(point + t * normal);
    }
  }
  else if (discriminant > 0)
  {  // 两个实数解
    float sqrt_discriminant = std::sqrt(discriminant);
    float t1                = (-B + sqrt_discriminant) / (2 * A);
    float t2                = (-B - sqrt_discriminant) / (2 * A);
    intersections.push_back(point + t1 * normal);
    intersections.push_back(point + t2 * normal);
  }
  else if (std::abs(discriminant) < 1e-6)
  {  // 一个实数解
    float t = -B / (2 * A);
    intersections.push_back(point + t * normal);
  }
  // 判别式小于0无实数解，返回空vector

  return intersections;
}

double LaserProcessor::evaluateQuadSurf(const cv::Mat &Coeff6x1, const cv::Point3f &p)
{
  float a      = Coeff6x1.at<float>(0);
  float b      = Coeff6x1.at<float>(1);
  float c      = Coeff6x1.at<float>(2);
  float d      = Coeff6x1.at<float>(3);
  float e      = Coeff6x1.at<float>(4);
  float f      = Coeff6x1.at<float>(5);
  float x_eval = a * p.y * p.y + b * p.y * p.z + c * p.z * p.z + d * p.y + e * p.z + f;
  float res    = abs(x_eval - p.x);

  cv::Point3f norm(1, -2.0 * a * p.y - b * p.z - d, -b * p.y - 2.0 * c * p.z - e);
  auto        vec_intersec = findIntersection(p, norm, Coeff6x1);
  for (const auto &p_inter : vec_intersec)
  {
    float res_i = cv::norm(p_inter - p);
    res         = MIN(res, res_i);
  }
  return res;
}

double LaserProcessor::evaluateQuadSurf(const cv::Mat& Coeff6x1, const std::vector<cv::Point3f>& points)
{
  std::vector<float> vec_res;
  for (const auto &p : points) vec_res.push_back(evaluateQuadSurf(Coeff6x1, p));
  float sum_of_squares = 0.0f;
  for (float value : vec_res)
  {
    sum_of_squares += std::pow(value, 2.0f);  // 或者 value * value
  }
  float mean_square = sum_of_squares / (float)vec_res.size();

  // 3. 计算平方根
  float rmse = std::sqrt(mean_square);
  return rmse;
}


void LaserProcessor::match2(
    const std::vector<std::map<float, float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l, const cv::Mat& rectify_r) {

    const auto calib_info = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();

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
        
        // 为当前左激光线创建输出文件
        std::ofstream data_ofs("points_3d/L" + std::to_string(l_laser_idx) + "_results.txt");
        data_ofs << "左激光线 " << l_laser_idx << " 重投影与匹配结果\n";
        data_ofs << "平面索引 | 最佳右激光线 | 平均得分 | 覆盖点数 | 总点数 | 点坐标 (左) | 重投影坐标 (右)\n";
        
#ifdef DEBUG_PLANE_MATCH
        cv::Mat vis_img;
        cv::hconcat(rectify_l, rectify_r, vis_img);
        if (vis_img.channels() == 1)
            cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
        const int cols = rectify_l.cols;
        cv::circle(vis_img, cv::Point(cx_l, cy_l), 4, cv::Scalar(0, 0, 255), -1);
        cv::circle(vis_img, cv::Point(cx_r + cols, cy_r), 4, cv::Scalar(0, 0, 255), -1);
#endif


        MatchMap match_map;
        std::map<int, PlaneMatchResult> plane_results;
        
        // 光平面循环：针对当前左激光线，遍历每个光平面
        for (int plane_idx = 0; plane_idx < static_cast<int>(planes.size()); ++plane_idx) {
            const auto& plane = planes[plane_idx];
            PlaneMatchResult& result = plane_results[plane_idx];
            result.plane_idx = plane_idx;
            
            // 为当前平面输出分数
            std::string score_filename = "points_3d/L" + 
                            std::to_string(l_laser_idx) + "_P" +
                            std::to_string(plane_idx) + "_scores.txt";
            std::ofstream score_ofs(score_filename);

            // 点集循环：针对当前平面，遍历左激光线上的每个点
            std::vector<cv::Point2f> vec_points;
            for (const auto& [y, x] : l_points) {
#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
#endif
                // 创建重投影信息（先初始化）
                ReprojectionInfo info(x, y, -1, -1);
                
                cv::Point3f ray_dir((x - cx_l) / fx_l, (y - cy_l) / fy_l, 1.0f);
                ray_dir *= 1.0f / cv::norm(ray_dir);
                cv::Point3f ray_origin(0, 0, 0);

                cv::Point3f pt;
                if (plane.coefficients.rows == 4) { // 平面求交
                    cv::Vec3f normal(
                        plane.coefficients.at<float>(0, 0),
                        plane.coefficients.at<float>(1, 0),
                        plane.coefficients.at<float>(2, 0)
                    );
                    float d = plane.coefficients.at<float>(3, 0);
                    float cos_theta = ray_dir.dot(normal);
                    if (std::abs(cos_theta) < 1e-6f) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无法求交的点
                    }
                    float t = -d / cos_theta;
                    pt = ray_dir * t;
                }
                else {  // 曲面求交
                    auto intersections = findIntersection(ray_origin, ray_dir, plane.coefficients);
                    if (intersections.empty()) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无交点的点
                    }
                    bool valid_pt_found = false;
                    for (const auto& p : intersections) {
                        float err = evaluateQuadSurf(plane.coefficients, p);
                        if (p.z <= 100 || p.z >= 1200) continue;
                        pt = p;
                        valid_pt_found = true;
                    }
                    if (!valid_pt_found) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无有效点的点
                    }
                }

                // 只在左激光线的第一条平面输出3D点
                // if (plane_idx == 0 && l_laser_idx == 0) {
                //     std::ofstream ofs("points3d.txt", std::ios::app);
                //     ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
                //     ofs.close();
                // }

                cv::Point3f pt_r(pt.x - baseline, pt.y, pt.z);
                float x_r = fx_r * pt_r.x / pt_r.z + cx_r;
                float y_r = std::round(fy_r * pt_r.y / pt_r.z + cy_r);

                // 更新重投影信息
                info.x_right = x_r;
                info.y_right = y_r;

                // 如果点不在右图像内
                if (x_r < 0 || x_r >= rectify_r.cols || y_r < 0 || y_r >= rectify_r.rows) {
                    continue;
                }

#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(cvRound(x_r) + cols, cvRound(y_r)), 1, cv::Scalar(0, 0, 255), -1);
#endif


                // 右激光线循环：针对当前点和平面结果，遍历所有右激光线
                bool has_candidates = false;
                for (int r_laser_idx = 0; r_laser_idx < static_cast<int>(laser_r.size()); ++r_laser_idx) {
                    const auto& r_line = laser_r[r_laser_idx].points;
                    if (r_line.empty()) continue;

                    if (l_laser_idx == 1 && plane_idx == 1 && r_laser_idx == 2)
                        puts("");

                    float min_dist = FLT_MAX;
                    auto it = r_line.find(y_r);
                    if (it != r_line.end()) min_dist = std::hypot(it->second.x - x_r, it->second.y - y_r);
                    if (min_dist == FLT_MAX) continue;

                    // 记录该点在该右激光线上的得分
                    has_candidates = true;
                    info.r_scores[r_laser_idx] = min_dist;
                    
                    // 更新匹配映射
                    MatchKey key{l_laser_idx, plane_idx, r_laser_idx};
                    match_map[key].score_sum -= min_dist;
                    match_map[key].count += 1;
                    
                    // 更新该右激光线的统计信息
                    result.r_line_scores[r_laser_idx].score_sum -= min_dist;
                    result.r_line_scores[r_laser_idx].count += 1;
                    
                    // 调试输出（根据需要启用）
                    // if (l_laser_idx == 6 && (r_laser_idx == 8 || r_laser_idx == 10))
                    //     printf("point(%f, %f) - r_idx_%d - p_idx_%d - score: %f\n", x, y, r_laser_idx, plane_idx, min_dist);
                }
                if (!has_candidates) continue;
                // 保存重投影信息
                result.reprojected_points.push_back(info);
                result.point_count++;  // 仅计数有效点
            } // end 点集循环
            
            // 处理完当前平面所有点后，计算该平面的匹配结果
            if (result.point_count > 0) {
                // 寻找该平面的最佳右激光线
                float best_score = -FLT_MAX;
                int best_r_idx = -1;
                
                for (auto& [r_idx, score_acc] : result.r_line_scores) {
                    if (score_acc.count > 0) {
                        float avg_score = score_acc.score_sum / score_acc.count;
                        
                        // 找出最佳右激光线
                        if (avg_score > best_score) {
                            best_score = avg_score;
                            best_r_idx = r_idx;
                        }
                    }
                }
                
                // 更新平面结果
                result.best_r_idx = best_r_idx;
                result.avg_score = best_score;
                
                // 输出当前平面的匹配摘要到文件
                data_ofs << plane_idx << " | " << result.best_r_idx << " | " 
                         << result.avg_score << " | " << result.point_count << " | " 
                         << l_points.size() << "\n";
                
                // 输出所有重投影点的详细信息
                for (const auto& info : result.reprojected_points) {
                    data_ofs << "    P: (" << info.x_left << ", " << info.y_left << ") -> (" 
                             << info.x_right << ", " << info.y_right << ") | ";      
                    if (!info.r_scores.empty()) {
                        data_ofs << "R Scores: [";
                        for (const auto& [r_idx, score] : info.r_scores) {
                            data_ofs << r_idx << ":" << score << " ";
                        }
                        data_ofs << "] | Best R: " << result.best_r_idx;
                        if (result.best_r_idx >= 0 && info.r_scores.count(result.best_r_idx)) {
                            data_ofs << " (Score: " << info.r_scores.at(result.best_r_idx) << ")";
                        }
                    } else if (info.x_right >= 0 && info.y_right >= 0) {
                        data_ofs << "坐标在右图像范围内但无匹配";
                    } else {
                        data_ofs << "重投影到右图像外";
                    }
                    data_ofs << "\n";


                    float output_score = -1;  // 默认无效分数
                    // 检查当前平面最佳右激光线的分数是否存在
                    if (info.r_scores.count(best_r_idx)) {
                        score_ofs << info.r_scores.at(best_r_idx) << "\n";  // 输出分数值
                    }
                }
                score_ofs.close();  // 关闭分数文件
                
                // 输出当前平面的匹配摘要到控制台
                printf("左激光线 %d | 平面 %d | 最佳右线: %d | 平均得分: %.3f | 有效点: %d/%d\n",
                       l_laser_idx, plane_idx, result.best_r_idx, result.avg_score, 
                       result.point_count, static_cast<int>(l_points.size()));
            }
        } // end 光平面循环
        
        // 寻找最佳匹配
        int best_r_idx = -1;
        int best_plane_idx = -1;
        float best_score = -FLT_MAX;
        const int n_samples = static_cast<int>(l_points.size());

        for (const auto& [key, result] : match_map) {
            if (key.l_laser_idx != l_laser_idx) continue;
            // if (result.count < n_samples / 10) continue;  // 最小覆盖阈值

            // 计算左激光线上所有采样点的平均得分
            float avg_score = result.score_sum / result.count;

            if (avg_score > best_score) {
                best_score = avg_score;
                best_plane_idx = key.plane_idx;
                best_r_idx = key.r_laser_idx;
            }
        }

        // 输出整体匹配结果
        if (best_r_idx >= 0) {
            printf("左激光线 %d 最佳匹配: 平面 %d | 右线 %d | 得分 %.3f\n", 
                   l_laser_idx, best_plane_idx, best_r_idx, best_score);
        } else {
            printf("左激光线 %d 没有找到有效匹配\n", l_laser_idx);
        }   

#ifdef DEBUG_PLANE_MATCH
        // 1. 可视化所有候选匹配线
        for (const auto& [key, result] : match_map) {
            if (key.l_laser_idx != l_laser_idx) continue;
            if (result.count < n_samples / 10) continue;

            const auto& r_points = laser_r[key.r_laser_idx].points;
            float unit_score = result.score_sum / std::max(1, static_cast<int>(r_points.size()));
            float coverage = result.count / float(n_samples);
            float combined = unit_score * coverage;

            // 匹配线颜色：最佳匹配高亮，其他为淡绿色
            cv::Scalar line_color = (key.r_laser_idx == best_r_idx) ? cv::Scalar(128, 0, 128) : cv::Scalar(0, 180, 0);

            // 画点
            for (const auto& [y, pt] : r_points) {
                cv::circle(vis_img, cv::Point(cvRound(pt.x + cols), cvRound(y)), 2, line_color, -1);
            }

            // 计算中心点
            if (!r_points.empty()) {
                float sum_x = 0, sum_y = 0;
                for (const auto& [y, pt] : r_points) {
                    sum_x += pt.x + cols;
                    sum_y += y;
                }
                int center_x = cvRound(sum_x / r_points.size());
                int center_y = cvRound(sum_y / r_points.size());

                // 标注r_laser_idx
                cv::putText(vis_img, "R" + std::to_string(key.r_laser_idx),
                    cv::Point(center_x, center_y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.4, line_color, 2);
            }
        }

        // 2. 可视化最佳匹配连线
        if (best_r_idx >= 0 && best_r_idx < static_cast<int>(laser_r.size())) {
            const auto& best_line = laser_r[best_r_idx].points;
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

        // 3. 显示最佳匹配文本
        std::string text = (best_r_idx >= 0) ?
            "Best Match: L" + std::to_string(l_laser_idx) +
            " - R" + std::to_string(best_r_idx) +
            " (P " + std::to_string(best_plane_idx) + " / S " + std::to_string(best_score) + ")" :
            "Best Match: None";

        cv::putText(vis_img, text,
            cv::Point((vis_img.cols - 500) / 2, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        cv::namedWindow("Sample Points Projection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Sample Points Projection", vis_img.cols, vis_img.rows);
        cv::imshow("Sample Points Projection", vis_img);
        if (cv::waitKey(0) == 27) break;  // ESC 跳出所有可视化
#endif
    
    }

}


/*************************************************************************************** */

void LaserProcessor::LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	int num = 0;
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
					num++;
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

void LaserProcessor::Two_PassNew(const cv::Mat &img, cv::Mat &labImg)
{
    cv::Mat bwImg;
    cv::threshold(img, bwImg, 100, 255, cv::THRESH_BINARY);
	assert(bwImg.type() == CV_8UC1);
	labImg.create(bwImg.size(), CV_32SC1);   //bwImg.convertTo( labImg, CV_32SC1 );
	labImg = cv::Scalar(0);
	labImg.setTo(cv::Scalar(1), bwImg);
	assert(labImg.isContinuous());
	const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
	int label = 1;
	std::vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);
	//the first pass
	int *data_prev = (int*)labImg.data;   //0-th row : int* data_prev = labImg.ptr<int>(i-1);
	int *data_cur = (int*)(labImg.data + labImg.step); //1-st row : int* data_cur = labImg.ptr<int>(i);
	for (int i = 1; i < Rows; i++)
	{
		data_cur++;
		data_prev++;
		for (int j = 1; j<Cols; j++, data_cur++, data_prev++)
		{
			if (*data_cur != 1)
				continue;
			int left = *(data_cur - 1);
			int up = *data_prev;
			int neighborLabels[2];
			int cnt = 0;
			if (left>1)
				neighborLabels[cnt++] = left;
			if (up > 1)
				neighborLabels[cnt++] = up;
			if (!cnt)
			{
				labelSet.push_back(++label);
				labelSet[label] = label;
				*data_cur = label;
				continue;
			}
			int smallestLabel = neighborLabels[0];
			if (cnt == 2 && neighborLabels[1]<smallestLabel)
				smallestLabel = neighborLabels[1];
			*data_cur = smallestLabel;
			// 保存最小等价表
			for (int k = 0; k<cnt; k++)
			{
				int tempLabel = neighborLabels[k];
				int& oldSmallestLabel = labelSet[tempLabel];  //这里的&不是取地址符号,而是引用符号
				if (oldSmallestLabel > smallestLabel)
				{
					labelSet[oldSmallestLabel] = smallestLabel;
					oldSmallestLabel = smallestLabel;
				}
				else if (oldSmallestLabel<smallestLabel)
					labelSet[smallestLabel] = oldSmallestLabel;
			}
		}
		data_cur++;
		data_prev++;
	}
	//更新等价队列表,将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}
	//second pass
	data_cur = (int*)labImg.data;
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < bwImg.cols - 1; j++, data_cur++)
			*data_cur = labelSet[*data_cur];
		data_cur++;
	}
}

cv::Scalar LaserProcessor::GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}



/************************************** Match Third **************************************
// 计算点集的主方向（2D PCA）
cv::Vec2f LaserProcessor::computePrincipalDirection(const std::vector<cv::Point2f>& pts) {
    cv::Mat data(pts.size(), 2, CV_32F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    auto dir = pca.eigenvectors.row(0);
    return cv::Vec2f(dir[0], dir[1]);
}

// 将角度映射为 [0,1] 相似度：夹角越小，相似度越高
float LaserProcessor::orientationScore(const cv::Vec2f& v1, const cv::Vec2f& v2) {
    float dot = v1.dot(v2) / (cv::norm(v1) * cv::norm(v2) + 1e-6f);
    dot = std::clamp(dot, -1.0f, 1.0f);
    float angle = std::acos(dot); // [0, pi]
    // 5° 阈值下映射： angle=0->1, angle=5°->0
    float cutoff = CV_PI * 5.0f / 180.0f;
    return std::max(0.0f, 1.0f - angle / cutoff);
}

// 简易 DTW 计算两序列 x(y) 形状相似度, 返回 [0,1]
float LaserProcessor::dtwScore(const std::vector<float>& seq1, const std::vector<float>& seq2) {
    int n = seq1.size(), m = seq2.size();
    const float INF = 1e9;
    std::vector<std::vector<float>> D(n+1, std::vector<float>(m+1, INF));
    D[0][0] = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            float cost = std::abs(seq1[i-1] - seq2[j-1]);
            D[i][j] = cost + std::min({D[i-1][j], D[i][j-1], D[i-1][j-1]});
        }
    }
    float dist = D[n][m];
    // 用指数衰减映射到 [0,1]
    return std::exp(-dist / (float)(n+m+1));
}


void LaserProcessor::match3(
    const std::vector<std::map<float, float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l, const cv::Mat& rectify_r) {

    // 基本参数
    auto calib_info = ConfigManager::getInstance().getCalibInfo();
    auto planes     = ConfigManager::getInstance().getQuadSurfaces();
    double fx_l = calib_info.P[0].at<double>(0,0), fy_l = calib_info.P[0].at<double>(1,1);
    double cx_l = calib_info.P[0].at<double>(0,2), cy_l = calib_info.P[0].at<double>(1,2);
    double fx_r = calib_info.P[1].at<double>(0,0), fy_r = calib_info.P[1].at<double>(1,1);
    double cx_r = calib_info.P[1].at<double>(0,2), cy_r = calib_info.P[1].at<double>(1,2);
    double baseline = -calib_info.P[1].at<double>(0,3) / fx_r;
    
    for (int l_idx = 0; l_idx < (int)sample_points.size(); ++l_idx) {
        const auto& l_map = sample_points[l_idx];
        std::vector<cv::Point2f> l_pts;
        for (auto& [y,x]: l_map) l_pts.emplace_back(x,y);
        if (l_pts.empty()) continue;

// 可视化
#ifdef DEBUG_PLANE_MATCH
        cv::Mat vis;
        cv::hconcat(rectify_l, rectify_r, vis);
        if (vis.channels()==1) cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
        int offset = rectify_l.cols;
        cv::circle(vis, {cvRound(cx_l),cvRound(cy_l)},4,{0,0,255},-1);
        cv::circle(vis, {cvRound(cx_r)+offset,cvRound(cy_r)},4,{0,0,255},-1);
#endif

        // 左侧主方向
        cv::Vec2f dir_l = computePrincipalDirection(l_pts);

        struct Candidate {int plane, rline; float score;};
        std::vector<Candidate> cands;

        // 平面循环
        for (int p_idx = 0; p_idx < (int)planes.size(); ++p_idx) {
            std::vector<cv::Point2f> reproj;
            const auto& plane = planes[p_idx];
            // 重投影点
            for (auto& [y,x]: l_map) {
#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
#endif
                // 创建重投影信息（先初始化）
                ReprojectionInfo info(x, y, -1, -1);
                
                cv::Point3f ray_dir((x - cx_l) / fx_l, (y - cy_l) / fy_l, 1.0f);
                ray_dir *= 1.0f / cv::norm(ray_dir);
                cv::Point3f ray_origin(0, 0, 0);

                cv::Point3f pt;
                if (plane.coefficients.rows == 4) { // 平面求交
                    cv::Vec3f normal(
                        plane.coefficients.at<float>(0, 0),
                        plane.coefficients.at<float>(1, 0),
                        plane.coefficients.at<float>(2, 0)
                    );
                    float d = plane.coefficients.at<float>(3, 0);
                    float cos_theta = ray_dir.dot(normal);
                    if (std::abs(cos_theta) < 1e-6f) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无法求交的点
                    }
                    float t = -d / cos_theta;
                    pt = ray_dir * t;
                }
                else {  // 曲面求交
                    auto intersections = findIntersection(ray_origin, ray_dir, plane.coefficients);
                    if (intersections.empty()) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无交点的点
                    }
                    bool valid_pt_found = false;
                    for (const auto& p : intersections) {
                        float err = evaluateQuadSurf(plane.coefficients, p);
                        if (p.z <= 100 || p.z >= 1200) continue;
                        pt = p;
                        valid_pt_found = true;
                    }
                    if (!valid_pt_found) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无有效点的点
                    }
                }

                // 只在左激光线的第一条平面输出3D点
                // if (plane_idx == 0 && l_laser_idx == 0) {
                //     std::ofstream ofs("points3d.txt", std::ios::app);
                //     ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
                //     ofs.close();
                // }

                cv::Point3f pt_r(pt.x - baseline, pt.y, pt.z);
                float x_r = fx_r * pt_r.x / pt_r.z + cx_r;
                float y_r = std::round(fy_r * pt_r.y / pt_r.z + cy_r);

                // 更新重投影信息
                info.x_right = x_r;
                info.y_right = y_r;

                // 如果点不在右图像内
                if (x_r >= 0 && x_r < rectify_r.cols && y_r >= 0 && y_r < rectify_r.rows)
                    reproj.emplace_back(x_r,y_r);
#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis, {cvRound(x_r)+offset,cvRound(y_r)},1,{0,255,0},-1);
#endif

            }
            if (reproj.empty()) continue;

            // 方向评分
            cv::Vec2f dir_p = computePrincipalDirection(reproj);
            float ori_s = orientationScore(dir_p, dir_l);
            
            // 横坐标序列
            std::vector<float> seq_p;
            for (auto& pt: reproj) seq_p.push_back(pt.x);

            // 对每条右线打分
            for (int r_idx = 0; r_idx < (int)laser_r.size(); ++r_idx) {
                const auto& line = laser_r[r_idx].points;
                if (line.empty()) continue;
                // 右侧点与序列
                std::vector<float> seq_r;
                for (auto& [y,pt]: line) seq_r.push_back(pt.x);
                float dtw_s = dtwScore(seq_p, seq_r);

                // 距离得分
                float sumd=0;
                for (auto& pt: reproj) {
                    // 找y最接近
                    auto it = std::min_element(line.begin(), line.end(), [&](auto&a,auto&b){
                        return std::abs(a.first-pt.y)<std::abs(b.first-pt.y);
                    });
                    sumd += cv::norm(cv::Point2f(it->second.x,it->first)-pt);
                }
                float dist_s = std::exp(-sumd/reproj.size()/50);

                // 综合
                float alpha=0.5f;
                float final_s = alpha*dist_s + (1-alpha)*(ori_s*0.5f + dtw_s*0.5f);
                cands.push_back({p_idx,r_idx,final_s});

#ifdef DEBUG_PLANE_MATCH
                // 可视化不同候选颜色深浅
                cv::Scalar col = (final_s>0.5?cv::Scalar(255,0,255):cv::Scalar(0,180,0));
                for (auto& [y,pt]: line)
                    cv::circle(vis, {cvRound(pt.x)+offset,cvRound(y)},1,col,1);
#endif
            }
        }

        // 选择最优
        if (!cands.empty()) {
            auto best = *std::max_element(cands.begin(),cands.end(),[](auto&a,auto&b){return a.score<b.score;});
            printf("L%d -> Plane %d, R%d, Score=%.3f\n",l_idx,best.plane,best.rline,best.score);
#ifdef DEBUG_PLANE_MATCH
            cv::putText(vis, "Best: P"+std::to_string(best.plane)+" R"+std::to_string(best.rline),
                {50,50},cv::FONT_HERSHEY_SIMPLEX,1,{0,255,255},2);
            cv::imshow("Match2 Debug", vis);
            cv::waitKey(1);
#endif
        }
    }
}



/*************************************************************************************** */