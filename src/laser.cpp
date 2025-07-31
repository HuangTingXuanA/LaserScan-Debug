#include "laser.h"
#include <fstream>
#define DEBUG_PLANE_MATCH
#define DEBUG_PLANE_MATCH_FINAL
// #define DEBUG_CENTER_FIND


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

std::tuple<cv::Mat, cv::Mat, int> LaserProcessor::computeGaussianDerivatives(float sigma_val) 
{
    // 计算核尺寸（确保为奇数）
    int ksize = cvRound(sigma_val * 3 * 2) + 1;
    ksize = std::max(ksize, 3) > 31 ? 31 : std::max(ksize, 3);
    ksize = ksize % 2 == 0 ? ksize + 1 : ksize;
    
    // 创建标准高斯导数核（不旋转）
    cv::Mat dx_kernel, dy_kernel;
    cv::getDerivKernels(dx_kernel, dy_kernel, 1, 1, ksize, false, CV_32F);
    
    // 应用尺度缩放（符合高斯导数定义）
    float scale_factor = -1.0f / (sigma_val * sigma_val);
    cv::multiply(dx_kernel, scale_factor, dx_kernel);
    cv::multiply(dy_kernel, scale_factor, dy_kernel);
    
    return {dx_kernel, dy_kernel, ksize};
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

float LaserProcessor::findSymmetricCenter2(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range, size_t is_right) {
#ifdef DEBUG_CENTER_FIND
    cv::namedWindow("Gray Profile Visualization", cv::WINDOW_NORMAL);
#endif
    // 1. 搜索区间
    std::vector<std::pair<float, float>> profile; // (位置t, 灰度值val)

    // 更细致的步长
    float step = 0.1f;
    
    // 在法向量方向采样灰度剖面
    for (float t = -search_range; t <= search_range; t += step) {
        float px = x + t*dir[0];
        float py = y + t*dir[1];
        float val = interpolateChannel(img, px, py);
        profile.emplace_back(t, val);
    }
    
    // 2. 找到灰度峰值区域
    float max_val = -FLT_MAX;
    for (const auto& p : profile) {
        max_val = std::max(max_val, p.second);
    }
    
    // 提取接近峰值的所有点（考虑到灰度相同的情况）
    const float tolerance = 0.01f; // 灰度值容差
    std::vector<float> peak_positions;
    
    for (const auto& p : profile) {
        if (std::fabs(p.second - max_val) <= tolerance) {
            peak_positions.push_back(p.first);
        }
    }
    
    float res1 = FLT_MAX, res2 = FLT_MAX, res3 = FLT_MAX;
    std::vector<float> fit_t, fit_val;
    fit_t.resize(7), fit_val.resize(7);

    // 3. 如果有多个灰度相同的点，找到峰值区域的中心
    if (peak_positions.size() > 1) {
        // 连续同值区域分组
        std::vector<std::vector<float>> regions;
        std::vector<float> current_region = {peak_positions[0]};
        
        for (size_t i = 1; i < peak_positions.size(); ++i) {
            // 如果与前一个点连续（考虑采样步长），归为同一区域
            if (std::fabs(peak_positions[i] - peak_positions[i-1]) <= 1.5f * step) {
                current_region.push_back(peak_positions[i]);
            } else {
                regions.push_back(current_region);
                current_region = {peak_positions[i]};
            }
        }
        
        if (!current_region.empty()) {
            regions.push_back(current_region);
        }
        
        // 选择最长的连续区域（可能是线条的中心平台）
        auto longest_region = std::max_element(regions.begin(), regions.end(),
            [](const std::vector<float>& a, const std::vector<float>& b) {
                return a.size() < b.size();
            });
        
        if (longest_region != regions.end()) {
            // 返回该区域的中心位置
            float sum = 0;
            for (float pos : *longest_region) {
                sum += pos;
            }
            res1 = sum / longest_region->size();
            
#ifdef DEBUG_CENTER_FIND
            
            fit_t[0] = res1;
            fit_val[0] = 0.9f;
#endif
        }
    }
    
    // 4. 如果没有明显的平台区域，使用所有峰值的中心
    if (!peak_positions.empty()) {
        float sum = 0;
        for (float pos : peak_positions) {
            sum += pos;
        }
        res2 = sum / peak_positions.size();

#ifdef DEBUG_CENTER_FIND
        fit_t[1] = res2;
        fit_val[1] = 0.9f;
#endif
    }
    
    // 5. 抛物线拟合
    auto max_it = std::max_element(profile.begin(), profile.end(),
        [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            return a.second < b.second;
        });
    
    size_t max_idx = std::distance(profile.begin(), max_it);
    
    // 抛物线五点拟合 - 改进版
    float a_coef, b_coef, c_coef;
    if (profile.size() >= 5) {
        cv::Mat A(5, 3, CV_32F);
        cv::Mat b(5, 1, CV_32F);
        
        // 使用基于分布的采样点选择策略
        std::vector<size_t> selected_indices;
        
        // 确保包含最大值点
        selected_indices.push_back(max_idx);
        
        // 计算合适的最小间隔（基于profile的总跨度）
        float total_range = profile.back().first - profile.front().first;
        // 将最小间隔从1/10改为1/30，使采样更密集
        float min_distance = total_range / 30.0f; // 最小间隔为总范围的三十分之一
        
        // 创建非对称间隔策略，使峰值附近采样更密集
        std::vector<float> distance_factors = {0.5f, 0.75f, 1.5f, 2.5f}; // 不同距离因子
        
        // 向左右两侧各选择两个点，峰值附近间隔更小
        for (int direction = -1; direction <= 1; direction += 2) {
            float last_t = profile[max_idx].first;
            int idx = max_idx;
            
            for (int count = 0; count < 2; count++) {
                bool found = false;
                // 根据离峰值的距离动态调整最小间隔
                float current_min_distance = min_distance * distance_factors[count];
                
                while (true) {
                    idx += direction;
                    if (idx < 0 || idx >= profile.size()) break;
                    
                    if (std::abs(profile[idx].first - last_t) >= current_min_distance) {
                        selected_indices.push_back(idx);
                        last_t = profile[idx].first;
                        found = true;
                        break;
                    }
                }
                if (!found) break;
            }
        }
        
        // 如果没有选出足够的点，使用备选策略
        if (selected_indices.size() < 5) {
            selected_indices.clear();
            
            // 备选策略：均匀分布选择
            for (int i = 0; i < 5; i++) {
                size_t idx = static_cast<size_t>(i * (profile.size() - 1) / 4); // 均匀分布
                selected_indices.push_back(idx);
            }
            
            // 确保包含最大值点
            bool has_max = false;
            for (size_t idx : selected_indices) {
                if (idx == max_idx) {
                    has_max = true;
                    break;
                }
            }
            
            if (!has_max) {
                // 用中间点替换为max_idx
                selected_indices[2] = max_idx;
            }
        }
        
        // 排序确保索引有序
        std::sort(selected_indices.begin(), selected_indices.end());
        
        // 使用选择的点进行拟合
        for (int i = 0; i < selected_indices.size(); i++) {
            const auto& p = profile[selected_indices[i]];
            A.at<float>(i, 0) = p.first * p.first; 
            A.at<float>(i, 1) = p.first; 
            A.at<float>(i, 2) = 1;
            b.at<float>(i, 0) = p.second; 

#ifdef DEBUG_CENTER_FIND
            if (i + 2 < fit_t.size()) {
                fit_t[i + 2] = p.first;
                fit_val[i + 2] = p.second;
            }
#endif
        }
    
        cv::Mat coeffs;
        if (cv::solve(A, b, coeffs, cv::DECOMP_SVD)) {
            a_coef = coeffs.at<float>(0); 
            b_coef = coeffs.at<float>(1);
            c_coef = coeffs.at<float>(2);

            if (a_coef < 0) { // 仅当抛物线开口向下时才接受拟合结果 
                res3 = -b_coef / (2 * a_coef);

                // 计算拟合点两侧灰度值
                float left_intensity = interpolateChannel(img, x + (res3 - 3)*dir[0], y + (res3 - 3)*dir[1]);
                
                float right_intensity = interpolateChannel(img, x + (res3 + search_range/2)*dir[0], y + (res3 + search_range/2)*dir[1]);
                float intensity_change = std::fabs(left_intensity - right_intensity);
                
                // 验证梯度幅值是否达标
                if (intensity_change < 0.1) {
                    res3 = FLT_MAX; // 拒绝平缓区域
                }
            }
        }
    }

    /** 三点拟合
    if (max_idx > 0 && max_idx < profile.size() - 1) {
        // 取三个点进行抛物线拟合
        float t0 = profile[max_idx-1].first;
        float t1 = profile[max_idx].first;
        float t2 = profile[max_idx+1].first;
        
        float y0 = profile[max_idx-1].second;
        float y1 = profile[max_idx].second;
        float y2 = profile[max_idx+1].second;
        
        // 抛物线拟合公式: y = a*t^2 + b*t + c
        float denom = (t0-t1)*(t0-t2)*(t1-t2);
        if (std::fabs(denom) > 1e-6f) {
            float a = ((t2*(y1-y0) + t1*(y0-y2) + t0*(y2-y1)) / denom);
            float b = ((t2*t2*(y0-y1) + t1*t1*(y2-y0) + t0*t0*(y1-y2)) / denom);
            
            // 计算抛物线峰值位置: t = -b/(2a)
            if (std::fabs(a) > 1e-6f) {
                return -b/(2*a);
            }
        }
    }
    */
    
#ifdef DEBUG_CENTER_FIND
    // 可视化灰度剖面和抛物线拟合（每隔一定步长进行一次，避免过多）
    // if (is_right == 0 && x > 720 && x < 815 && y > 300 && y < 370) {
        // 创建FreeType2对象并加载中文字体
        cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData("./simsun.ttc", 0);
        
        // 创建灰度剖面图，增大宽度以容纳图例
        cv::Mat profile_img(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // 绘制坐标轴
        cv::line(profile_img, cv::Point(50, 350), cv::Point(550, 350), cv::Scalar(0, 0, 0), 1);
        cv::line(profile_img, cv::Point(50, 50), cv::Point(50, 350), cv::Scalar(0, 0, 0), 1);
        
        // 添加水平轴刻度和标签
        int num_ticks = 5; // t轴上的刻度数量(每边)
        for (int i = -num_ticks; i <= num_ticks; i++) {
            float tick_t = i * search_range / num_ticks;
            int tick_x = 50 + static_cast<int>((tick_t + search_range) / (2 * search_range) * 500);
            
            // 绘制刻度线
            cv::line(profile_img, cv::Point(tick_x, 350), cv::Point(tick_x, 355), cv::Scalar(0, 0, 0), 1);
            
            // 添加刻度值
            std::string tick_label = cv::format("%.1f", tick_t);
            ft2->putText(profile_img, tick_label, cv::Point(tick_x - 10, 370), 14, 
                        cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        }
        
        // 添加垂直轴刻度和标签
        int v_num_ticks = 5; // intensity轴上的刻度数量
        for (int i = 0; i <= v_num_ticks; i++) {
            float tick_val = static_cast<float>(i) / v_num_ticks;
            int tick_y = 350 - static_cast<int>(tick_val * 300);
            
            // 绘制刻度线
            cv::line(profile_img, cv::Point(45, tick_y), cv::Point(50, tick_y), cv::Scalar(0, 0, 0), 1);
            
            // 添加刻度值
            std::string tick_label = cv::format("%.1f", tick_val);
            ft2->putText(profile_img, tick_label, cv::Point(25, tick_y + 5), 14, 
                        cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        }
        
        // 添加坐标轴标签
        ft2->putText(profile_img, "t", cv::Point(550, 365), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
        ft2->putText(profile_img, "intensity", cv::Point(30, 40), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
        
        // 映射函数将灰度剖面值映射到图像坐标
        auto mapToImg = [&](float t, float val) -> cv::Point {
            int img_x = 50 + static_cast<int>((t + search_range) / (2 * search_range) * 500);
            int img_y = 350 - static_cast<int>(val * 300);
            return cv::Point(img_x, img_y);
        };
        
        // 绘制灰度剖面点
        for (size_t i = 0; i < profile.size(); ++i) {
            cv::Point pt = mapToImg(profile[i].first, profile[i].second);
            cv::circle(profile_img, pt, 2, cv::Scalar(0, 0, 255), -1);
            
            // 连接相邻点
            if (i > 0) {
                cv::Point prev_pt = mapToImg(profile[i-1].first, profile[i-1].second);
                cv::line(profile_img, prev_pt, pt, cv::Scalar(0, 0, 255), 1);
            }
        }
        
        // 绘制抛物线拟合结果
        for (float t = -search_range; t <= search_range; t += 0.1f) {
            float val = a_coef * t * t + b_coef * t + c_coef;
            if (val >= 0 && val <= 1) {
                cv::Point pt = mapToImg(t, val);
                cv::circle(profile_img, pt, 1, cv::Scalar(0, 255, 0), -1);
            }
        }
        
        // 标记拟合使用的点
        for (size_t i = 2; i < fit_t.size(); ++i) {
            cv::Point pt = mapToImg(fit_t[i], fit_val[i]);
            cv::circle(profile_img, pt, 4, cv::Scalar(255, 0, 0), 2);
        }
        
        // 标记计算的峰值位置
        if (res1 != FLT_MAX) {
            cv::Point pt_res1 = mapToImg(fit_t[0], fit_val[0]);
            cv::circle(profile_img, pt_res1, 5, cv::Scalar(255, 255, 0), -1);  // 青色
        }
        if (res2 != FLT_MAX) {
            cv::Point pt_res2 = mapToImg(fit_t[1], fit_val[1]);
            cv::circle(profile_img, pt_res2, 5, cv::Scalar(30, 105, 210), -1);  // 巧克力色
        }
        if (res3 != FLT_MAX) {
            cv::Point peak_pt = mapToImg(res3, a_coef * res3 * res3 + b_coef * res3 + c_coef);
            cv::circle(profile_img, peak_pt, 5, cv::Scalar(0, 255, 255), -1);   // 黄色
        }
        
        // 添加文字说明
        ft2->putText(profile_img, "Profile at (" + std::to_string(x) + "," + std::to_string((int)y) + ")", 
                    cv::Point(50, 30), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
        ft2->putText(profile_img, "a=" + std::to_string(a_coef) + ", b=" + std::to_string(b_coef) + ", c=" + std::to_string(c_coef)
                    + ", res1=" + (res1 == FLT_MAX ? "no" : std::to_string(res1))
                    + ", res2=" + (res2 == FLT_MAX ? "no" : std::to_string(res2)) + ", res3=t_peak", 
                    cv::Point(50, 390), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
        
        // 添加右侧图例
        int legend_x = 590;
        int legend_y = 80;
        int legend_spacing = 40;
        
        // 图例标题
        ft2->putText(profile_img, "图例说明", cv::Point(legend_x, legend_y), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        legend_y += 30;
        
        // 原始灰度剖面
        cv::circle(profile_img, cv::Point(legend_x, legend_y), 2, cv::Scalar(0, 0, 255), -1);
        cv::line(profile_img, cv::Point(legend_x-10, legend_y), cv::Point(legend_x+10, legend_y), cv::Scalar(0, 0, 255), 1);
        ft2->putText(profile_img, "灰度剖面采样点", cv::Point(legend_x+20, legend_y+5), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        
        // 抛物线拟合
        legend_y += legend_spacing;
        cv::circle(profile_img, cv::Point(legend_x, legend_y), 2, cv::Scalar(0, 255, 0), -1);
        cv::line(profile_img, cv::Point(legend_x-10, legend_y), cv::Point(legend_x+10, legend_y), cv::Scalar(0, 255, 0), 1);
        ft2->putText(profile_img, "抛物线拟合曲线", cv::Point(legend_x+20, legend_y+5), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        
        // 拟合使用的点
        legend_y += legend_spacing;
        cv::circle(profile_img, cv::Point(legend_x, legend_y), 4, cv::Scalar(255, 0, 0), 2);
        ft2->putText(profile_img, "拟合使用的采样点", cv::Point(legend_x+20, legend_y+5), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        
        // 计算的峰值位置
        legend_y += legend_spacing;
        cv::circle(profile_img, cv::Point(legend_x, legend_y), 5, cv::Scalar(0, 255, 255), -1);
        ft2->putText(profile_img, "计算的线中心位置", cv::Point(legend_x+20, legend_y+5), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        
        // 添加计算方法说明
        legend_y += legend_spacing;
        ft2->putText(profile_img, "中心点计算方法:", cv::Point(legend_x, legend_y), 20, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        legend_y += 25;
        ft2->putText(profile_img, "1. y = at²+bt+c", cv::Point(legend_x+10, legend_y), 18, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        legend_y += 20;
        ft2->putText(profile_img, "2. t_peak = -b/(2a)", cv::Point(legend_x+10, legend_y), 18, 
                    cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        legend_y += 20;
        if (res3 == FLT_MAX)
            ft2->putText(profile_img, "3. t_peak = 无" , cv::Point(legend_x+10, legend_y), 20, 
                        cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        else
            ft2->putText(profile_img, "3. t_peak = " + cv::format("%.4f", res3), cv::Point(legend_x+10, legend_y), 18, 
                        cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
        
        // 显示图像
        const std::string profile_name = "./gray_profile/(" + std::to_string(x) + "_" + std::to_string(y) + ").png";
        //cv::imwrite(profile_name, profile_img);
        while(true){
            cv::imshow("Gray Profile Visualization", profile_img);
            if (cv::waitKey(0) == 27) break;
        }
    // }
#endif

    // 如果所有方法都失败，说明是极端边缘点
    float final_res = FLT_MAX;
    if (res1 != FLT_MAX) final_res = res1;
    else if (res2 != FLT_MAX) final_res = res2;
    else if (res3 != FLT_MAX) final_res = res3;
    return final_res;
}

float LaserProcessor::findSymmetricCenter3(
    const cv::Mat& img,
    float x, float y,
    cv::Vec2f dir,
    float R)
{
    // 采样与拟合参数
    const float step    = 0.05f;   // t 方向采样步长
#ifdef DEBUG_CENTER_FIND
    // 准备可视化窗口
    cv::namedWindow("GrayProfile", cv::WINDOW_NORMAL);
#endif

    float extend_range = R + 2.0f;
    // if (R < 5) extend_range += extend_range * 0.8f;
    // else if (R < 10) extend_range += extend_range * 0.7f;
    // else if (R > 10) extend_range += 5.0f;

    // 1. 全范围采样
    std::vector<float> Ts, Vs;
    for (float t = -3; t <= extend_range; t += step) {
        Ts.push_back(t);
        Vs.push_back(interpolateChannel(img, x + t*dir[0], y + t*dir[1]));
    }
    int N = (int)Ts.size();
    if (N < 3) return FLT_MAX;

    // 2. 找到第一个峰值
    float maxVal = -FLT_MAX;
    int peakIdx = -1;
    bool isRising = false;
    for (int i = 0; i < N; ++i) {
        // 只考虑 t <= R 的点
        if (Vs[i] > maxVal && Ts[i] <= R) {
            maxVal = Vs[i];
            peakIdx = i;
        }
    }
    float t_peak = Ts[peakIdx];
    if (Vs[peakIdx] < 0.5f) return FLT_MAX;

    // 确定第一个最大峰的下降沿边界（向右搜索）
    int rightBound = peakIdx + 1;  // 初始化为峰顶位置
    while (rightBound < N) {
        // 检测灰度值上升：表示下降沿结束
        float diff = Vs[rightBound] - Vs[rightBound-1];
        if ((diff <= 1e-3f && Vs[rightBound] <= 0.05) || Ts[rightBound] > R+1) break;
        rightBound++;
    }
    rightBound -= 2;
 
    // 3. 平台检测 → res1
    float res1 = FLT_MAX;
    {
        float v0 = Vs[peakIdx];
        int L = peakIdx, Rg = peakIdx;
        while (L>0 && std::fabs(Vs[L-1]-v0)<1e-6f) --L;
        while (Rg+1<rightBound && std::fabs(Vs[Rg+1]-v0)<1e-6f) ++Rg;
        if (Rg - L + 1 > 30) {
            res1 = 0.5f*(Ts[L] + Ts[Rg]);
        }
    }

    // 4. 抛物线拟合 → res2
    float res2 = FLT_MAX;
    {
        // 确定左右两侧长度
        int leftLen = peakIdx;        // 左侧点个数（不包括峰值）
        int rightLen = rightBound - 1 - peakIdx; // 右侧点个数（不包括峰值）

        // 判断哪边更短，并获取短边末端索引和灰度值
        int shortEnd;
        float targetGray;
        int searchStart, searchEnd;
        
        if (leftLen < rightLen) {
            // 左侧较短
            shortEnd = 0;
            targetGray = Vs[0];
            searchStart = peakIdx + 1;
            searchEnd = rightBound;
        } else {
            // 右侧较短
            shortEnd = rightBound;
            targetGray = Vs[rightBound];
            searchStart = 0;
            searchEnd = peakIdx - 1;
        }

        // 在长边寻找灰度最接近的点
        int matchIdx = -1;
        float minDiff = FLT_MAX;
        for (int i = searchStart; i <= searchEnd; i++) {
            float diff = std::fabs(Vs[i] - targetGray);
            if (diff < minDiff) {
                minDiff = diff;
                matchIdx = i;
            }
        }

        // 确定拟合点集范围
        int startIdx, endIdx;
        if (leftLen < rightLen) {
            startIdx = shortEnd;    // 左侧末端 (0)
            endIdx = matchIdx;      // 右侧匹配点
        } else {
            startIdx = matchIdx;    // 左侧匹配点
            endIdx = shortEnd;      // 右侧末端 (N-1)
        }
        int numPoints = endIdx - startIdx + 1;

        // 至少需要3个点才能拟合抛物线
        if (matchIdx != -1 && numPoints >= 3) {
            cv::Mat A(numPoints, 3, CV_32F), Y(numPoints, 1, CV_32F);
            for (int i = 0; i < numPoints; ++i) {
                int idx = startIdx + i;
                float t = Ts[idx];
                A.at<float>(i, 0) = t * t;
                A.at<float>(i, 1) = t;
                A.at<float>(i, 2) = 1.0f;
                Y.at<float>(i, 0) = Vs[idx];
            }

            cv::Mat coeff;
            if (cv::solve(A, Y, coeff, cv::DECOMP_SVD)) {
                float a = coeff.at<float>(0, 0);
                float b = coeff.at<float>(1, 0);
                if (a < 0) {  // 确保是开口向下的抛物线
                    float tp = -b / (2 * a);
                    if (tp >= -extend_range && tp <= extend_range) {
                        res2 = tp;
                    }
                }
            }
        }
    }
#ifndef DEBUG_CENTER_FIND
    if (res2 != FLT_MAX) {
        return res2;
    }
#endif

    float right_bound = Ts[rightBound];
    float res3 = t_peak;


#ifdef DEBUG_CENTER_FIND
    //
    // —— 可视化部分 —— 
    //
    int W = 800, H = 500;
    cv::Mat prof_img(H, W, CV_8UC3, cv::Scalar(255,255,255));
    int x0 = 60, y0 = H - 60;
    int x1 = W - 200, y1 = 60;
    // 0) 目前的点
    cv::putText(prof_img, cv::format("%s%.3f%s%.3f%s", "Profile at (", x, ",", y,")"), 
                cv::Point(50, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1);

    // 1) 坐标轴
    cv::line(prof_img, {x0, y0}, {x1, y0}, {0,0,0}, 1);
    cv::line(prof_img, {x0, y0}, {x0, y1}, {0,0,0}, 1);

    // 2) 刻度与标签
    int n_xt = 5;
    for (int i = 0; i <= n_xt; ++i) {
        float t_tick = -extend_range + 2*extend_range * i / n_xt;
        int xt = x0 + int((t_tick + extend_range)/(2*extend_range)*(x1 - x0));
        cv::line(prof_img, {xt, y0-5}, {xt, y0+5}, {0,0,0},1);
        cv::putText(prof_img, cv::format("%.2f", t_tick),
                    {xt-20, y0+25}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0},1);
    }
    int n_yt = 5;
    for (int i = 0; i <= n_yt; ++i) {
        float v_tick = float(i)/n_yt;
        int yt = y0 - int(v_tick*(y0 - y1));
        cv::line(prof_img, {x0-5, yt}, {x0+5, yt}, {0,0,0},1);
        cv::putText(prof_img, cv::format("%.2f", v_tick),
                    {x0-60, yt+5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,0,0},1);
    }

    // 3) 全采样点 & 连线（灰）
    for (int i = 0; i < N; ++i) {
        int xx = x0 + int((Ts[i]+extend_range)/(2*extend_range)*(x1 - x0));
        int yy = y0 - int(Vs[i]*(y0 - y1));
        cv::circle(prof_img, {xx,yy}, 3, {180,180,180}, -1);
        if (i>0) {
            int xx0 = x0 + int((Ts[i-1]+extend_range)/(2*extend_range)*(x1 - x0));
            int yy0 = y0 - int(Vs[i-1]*(y0 - y1));
            cv::line(prof_img, {xx0,yy0}, {xx,yy}, {200,200,200},1);
        }
    }


    // 4) 四条垂直标线 & 图例
    struct Rst{ float t; cv::Scalar c; const char* lbl; };
    std::vector<Rst> rst = {
        {res1, {0,255,0}, "res1"},
        {res2, {255,0,0}, "res2"},
        {res3, {255,255,0}, "res3"},
        {right_bound, {128, 0, 128}, "r_bound"}
    };
    int lx = W - 180, ly = 80, dy = 30;
    for (auto &r : rst) {
        if (r.t==FLT_MAX) continue;
        int xx = x0 + int((r.t+extend_range)/(2*extend_range)*(x1 - x0));
        cv::line(prof_img, {xx, y1}, {xx, y0}, r.c, 1);
        cv::putText(prof_img,
            cv::format("%s: %.3f", r.lbl, r.t),
            {lx, ly}, cv::FONT_HERSHEY_SIMPLEX, 0.6, r.c, 2);
        ly += dy;
    }

    cv::imshow("GrayProfile", prof_img);
    cv::waitKey(0);
#endif

    return res3;
}

std::vector<cv::Point2f> LaserProcessor::processCenters(const std::map<float,float>& orig)
{
    std::vector<cv::Point2f> out;
    if (orig.empty() || precision <= 0) 
        return out;

    // k -> x 映射，保证有序且去重
    std::map<int, float> k2x;

    // 阶段1：收集原始对齐点
    for (auto const& [y, x] : orig) {
        int k = int(std::lround(y / precision));
        if (std::abs(y - k*precision) < EPS) {
            k2x.emplace(k, x);
        }
    }

    // 阶段2：相邻对插值
    auto prev = orig.begin();
    for (auto it = std::next(orig.begin()); it != orig.end(); ++it, ++prev) {
        float yA = prev->first, xA = prev->second;
        float yB = it->first,  xB = it->second;
        if (yB <= yA + EPS) continue;
        // if ((yB - yA) > 2*precision + EPS) continue;

        int k_start = int(std::ceil((yA + EPS) / precision));
        int k_end   = int(std::floor((yB + EPS) / precision));

        for (int k = k_start; k <= k_end; ++k) {
            // 已有原始或已插值过，则跳过
            if (k2x.count(k)) 
                continue;
            float yi = k * precision;
            float t  = (yi - yA) / (yB - yA);
            float xi = xA + t * (xB - xA);
            k2x.emplace(k, xi);
        }
    }

    // 阶段3：按 k 有序输出
    out.reserve(k2x.size());
    for (auto const& [k, x] : k2x) {
        out.emplace_back(x, k * precision);
    }
    return out;
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

std::vector<LaserLine> LaserProcessor::extractLine(
    const std::vector<cv::RotatedRect>& rois,
    const cv::Mat& rectify_img,
    const cv::Mat& label_img,
    int img_idx) {
    std::vector<LaserLine> laser_lines;
    cv::Mat rectify_img_float;
    rectify_img.convertTo(rectify_img_float, CV_32F, 1.0f / 255.0f);
    
    // 保留全局二值化用于后续方向判断（非轮廓提取）
    cv::Mat bin;
    cv::threshold(rectify_img, bin, 80, 255, cv::THRESH_BINARY);
    
    // 可视化边缘
    cv::Mat direct_vis;
    cv::cvtColor(rectify_img, direct_vis, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < rois.size(); ++i) {
        const auto& roi = rois[i];
        const auto& roi_w = std::min(roi.size.width, roi.size.height);

        // 1. 获取ROI四个顶点
        cv::Point2f vertices[4];
        roi.points(vertices);

        // 2. 创建ROI区域mask
        cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8UC1);
        std::vector<cv::Point> roi_poly;
        for (int i = 0; i < 4; ++i) {
            roi_poly.push_back(cv::Point(
                cvRound(vertices[i].x), 
                cvRound(vertices[i].y)
            ));
        }
        std::vector<std::vector<cv::Point>> polys = {roi_poly};
        cv::fillPoly(mask, polys, cv::Scalar(255));

        // 3. 提取ROI区域内的标签图
        cv::Mat roi_labels(label_img.size(), CV_32SC1, cv::Scalar(0));
        label_img.copyTo(roi_labels, mask);  // 仅复制ROI区域的标签

        // 4. 在ROI区域内选择面积最大的连通域
        int max_label = 0;
        double max_area = 0.0;
        
        // 统计ROI内各标签面积
        std::map<int, double> label_areas;
        for (int r = 0; r < roi_labels.rows; ++r) {
            const int* ptr = roi_labels.ptr<int>(r);
            for (int c = 0; c < roi_labels.cols; ++c) {
                int label_val = ptr[c];
                if (label_val > 1) {  // 忽略背景(0)和未标记(1)
                    label_areas[label_val] += 1.0;
                }
            }
        }
        
        // 寻找最大面积的标签
        for (const auto& [label, area] : label_areas) {
            if (area > max_area) {
                max_area = area;
                max_label = label;
            }
        }
        
        // 5. 创建最大连通域的二值图像并提取轮廓
        cv::Mat max_blob = (roi_labels == max_label);
        max_blob.convertTo(max_blob, CV_8U, 255);  // 转换为8UC1格式
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(max_blob, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        
        // 6. 收集轮廓点（后续逻辑保持不变）
        if (contours.empty()) throw std::logic_error("no contours found in max blob");
        auto max_it = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return a.size() < b.size();
            });
        
        // 7. 主轴投影区域
        std::vector<std::vector<cv::Point>> max_contour{*max_it};
        std::unordered_map<int, int> edge_maxY_map;
        std::unordered_map<int, int> edge_minY_map;
        auto [p1, p2] = getAxisEndpoints(roi);
        cv::Point2f axis = p2 - p1;
        float axis_len = cv::norm(axis);
        cv::Point2f axis_dir = axis / axis_len;
        float min_proj = 0.02f * axis_len;
        float max_proj = 0.98f * axis_len;
        for (const auto& pt : max_contour[0]) {
            cv::Point2f pt_f(pt.x, pt.y);
            cv::Point2f vec = pt_f - p1;
            float proj = vec.dot(axis_dir);
            if (proj >= min_proj && proj <= max_proj) {
                // 找Y最大值
                if (edge_maxY_map.find(pt.x) == edge_maxY_map.end())
                    edge_maxY_map[pt.x] = pt.y;
                else if (pt.y > edge_maxY_map[pt.x])
                    edge_maxY_map[pt.x] = pt.y;
                
                // 找Y最小值
                if (edge_minY_map.find(pt.x) == edge_minY_map.end())
                    edge_minY_map[pt.x] = pt.y;
                else if (pt.y < edge_minY_map[pt.x])
                    edge_minY_map[pt.x] = pt.y;
            }
        }

        // 8. 可视化边缘方向
        for (const auto& [x, y] : edge_minY_map)
            direct_vis.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(0, 0, 255); // 红色表示上边沿
        for (const auto& [x, y] : edge_maxY_map)
            direct_vis.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255, 0, 0); // 蓝色表示下边沿

        cv::Mat laser_center_vis;
        cv::cvtColor(rectify_img, laser_center_vis, cv::COLOR_GRAY2BGR);
        std::map<float, float> orign_centers;
        std::unordered_map<int ,bool> x_used;
        float max_search_range = -FLT_MAX;
        for (const auto& [x, y] : edge_minY_map) {
            if (x_used[x]) continue;

            auto it = edge_maxY_map.find(x);
            if (it == edge_maxY_map.end()) continue;
            float search_range = (it->second - y + 1);
            if (search_range < 3 || search_range > 17) continue;
            cv::Vec2f dir(0, 1);
            x_used[x] = true;
            if (search_range > max_search_range) max_search_range = search_range;

            // if (x == 1755 && y == 647)
                // puts("");
         
            // float t_peak = FLT_MAX;
            // if ((x >= 992 && x <= 1037) && (y >= 610 && y <= 640))
                // t_peak = findSymmetricCenter3(rectify_img_float, x, y, dir, search_range);

            float t_peak = findSymmetricCenter3(rectify_img_float, x, y, dir, search_range);
            // float t_peak = findSymmetricCenter2(rectify_img_float, x, y, dir, roi_w, 0);
            if (t_peak == FLT_MAX || t_peak <= 0) continue;
            float center_x = x + t_peak*dir[0], center_y = y + t_peak * dir[1];

            orign_centers[center_y] = center_x;

            // 可视化激光线中心点
            cv::Point2f center(center_x, center_y);
            laser_center_vis.at<cv::Vec3b>(cv::Point(center.x, center.y)) = cv::Vec3b(0, 255, 0); // 绿色表示中心点
        }



        // 9. 同一条线相邻中心点插值为整数
        cv::Mat new_centers_vis;
        cv::cvtColor(rectify_img, new_centers_vis, cv::COLOR_GRAY2BGR);
        std::vector<cv::Point2f> new_centers = processCenters(orign_centers);
        for (const auto& p : new_centers) new_centers_vis.at<cv::Vec3b>(cv::Point(std::round(p.x), p.y)) = cv::Vec3b(0, 255, 0); // 绿色表示中心点
    

        // 10. 存储结果
        // const float roi_angle = roi.angle * CV_PI / 180.0f;
        // const float laser_width = max_search_range * sin(roi_angle);
        // const float sigma_val = convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f)));
        // const bool h_is_long_edge = roi.size.height >= roi.size.width;
        // auto [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val, roi_angle, h_is_long_edge);
        const float laser_width = max_search_range;
        const float sigma_val = convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f)));
        auto [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val);
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

    static int vis_img_cnt = 0;
    if (vis_img_cnt % 2 == 0) cv::imwrite(debug_img_dir / ("direct_l_" + std::to_string(img_idx) + ".jpg"), direct_vis);
    else cv::imwrite(debug_img_dir / ("direct_r_" + std::to_string(img_idx) + ".jpg"), direct_vis);
    vis_img_cnt++;
    
    return laser_lines;
}


std::vector<LaserLine> LaserProcessor::extractLine2(
    const cv::Mat& rectify_img,
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
    int img_idx) {
    std::vector<LaserLine> laser_lines;
    cv::Mat rectify_img_float;
    rectify_img.convertTo(rectify_img_float, CV_32F, 1.0f / 255.0f);
    
    for (size_t i = 0; i < contours.size(); ++i) {
        const auto& edge_pair = contours[i];

        // 1. 求最小外接矩形 ROI
        std::vector<cv::Point> contour_points;
        for (const auto& p : edge_pair) {
            contour_points.push_back(p.first);
            contour_points.push_back(p.second);
        }
        cv::RotatedRect roi = cv::minAreaRect(contour_points);
        
        // 2. 主轴投影区域
        auto [p1, p2] = getAxisEndpoints(roi);
        cv::Point2f axis = p2 - p1;
        float axis_len = cv::norm(axis);
        cv::Point2f axis_dir = axis / axis_len;
        float min_proj = 0.03f * axis_len;
        float max_proj = 0.97f * axis_len;

        // 3. 激光线中心点提取
        cv::Mat laser_center_vis;
        cv::cvtColor(rectify_img, laser_center_vis, cv::COLOR_GRAY2BGR);
        std::map<float, float> orign_centers;
        float max_search_range = -FLT_MAX;
        for (const auto& p : edge_pair) {
            cv::Point2f vec = cv::Point2f(p.first.x, p.first.y) - p1;
            float proj = vec.dot(axis_dir);
            if (proj <= min_proj || proj >= max_proj) continue;

            float search_range = (p.second.y - p.first.y + 1);
            cv::Vec2f dir(0, 1);
            if (search_range > max_search_range) max_search_range = search_range;

            // if (x == 1755 && y == 647)
                // puts("");
         
            // float t_peak = FLT_MAX;
            // if ((p.first.x >= 1226 && p.first.x <= 1239) && (p.first.y >= 32 && p.first.y <= 48))
                // t_peak = findSymmetricCenter3(rectify_img_float, p.first.x, p.first.y, dir, search_range);

            float t_peak = findSymmetricCenter3(rectify_img_float, p.first.x, p.first.y, dir, search_range);
            if (t_peak == FLT_MAX) continue;
            float center_x = p.first.x + t_peak*dir[0], center_y = p.first.y + t_peak * dir[1];

            orign_centers[center_y] = center_x;

            // 可视化激光线中心点
            cv::Point2f center(center_x, center_y);
            laser_center_vis.at<cv::Vec3b>(cv::Point(center.x, center.y)) = cv::Vec3b(0, 255, 0); // 绿色表示中心点
        }



        // 4. 同一条线相邻中心点插值为整数
        cv::Mat new_centers_vis;
        cv::cvtColor(rectify_img, new_centers_vis, cv::COLOR_GRAY2BGR);
        std::vector<cv::Point2f> new_centers = processCenters(orign_centers);
        for (const auto& p : new_centers) new_centers_vis.at<cv::Vec3b>(cv::Point(std::round(p.x), p.y)) = cv::Vec3b(0, 255, 0); // 绿色表示中心点
    

        // 5. 存储结果
        // const float roi_angle = roi.angle * CV_PI / 180.0f;
        // const float laser_width = max_search_range * sin(roi_angle);
        // const float sigma_val = convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f)));
        // const bool h_is_long_edge = roi.size.height >= roi.size.width;
        // auto [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val, roi_angle, h_is_long_edge);
        const float laser_width = max_search_range;
        const float sigma_val = convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f)));
        auto [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val);
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


/************************************** Quad Surface Operation **************************************/
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


/************************************** Match Three ***************************************/

float LaserProcessor::computeCompScore3(float avg_dist, float coverage) {
    const float DIS_WEIGHT = 2.0f;
    const float COVER_WEIGHT = 1.0f;

    // 分数越低越好
    return (avg_dist * DIS_WEIGHT) + ((1.0f - coverage) * COVER_WEIGHT * 10.0f);
}

std::vector<std::tuple<int, int, int>> LaserProcessor::match3(
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

    // 匹配结果
    std::vector<std::tuple<int, int, int>> match_results;

    // 匹配策略数据结构
    std::set<int> used_r_lines; // 已使用的右激光线索引
    std::vector<std::tuple<int, int, float>> final_matches(sample_points.size(), 
                                                 std::make_tuple(-1, -1, -FLT_MAX));
    std::vector<UnmatchedInfo> unmatched_info;

    // 第一轮匹配：处理每条左激光线
    for (int l_laser_idx = 0; l_laser_idx < static_cast<int>(sample_points.size()); ++l_laser_idx) {
        const auto& l_points = sample_points[l_laser_idx];
        if (l_points.empty()) continue;  // 跳过空点集
        
        // 准备可视化
#ifdef DEBUG_PLANE_MATCH
        std::ofstream data_ofs("points_3d/L" + std::to_string(l_laser_idx) + "_results.txt");
        data_ofs << "左激光线 " << l_laser_idx << " 重投影与匹配结果\n";
        data_ofs << "平面索引 | 最佳右激光线 | 平均得分 | 覆盖点数 | 总点数 | 点坐标 (左) | 重投影坐标 (右)\n";

        cv::Mat vis_img;
        cv::hconcat(rectify_l, rectify_r, vis_img);
        if (vis_img.channels() == 1)
            cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
        const int cols = rectify_l.cols;
        for (int r_idx : used_r_lines) {
        if (r_idx < laser_r.size()) {
            const auto& r_points = laser_r[r_idx].points;
            for (const auto& [y, pt] : r_points) {
                cv::circle(vis_img, cv::Point(cvRound(pt.x) + cols, y), 
                           3, cv::Scalar(0, 255, 255), -1);
            }
        }
    }

#endif

        // 匹配处理数据结构
        std::map<MatchKey, ScoreAccumulator> match_map;
        std::map<int, PlaneMatchResult> plane_results;
        UnmatchedInfo current_unmatched;
        bool has_match = false;

        // 处理所有光平面
        for (int plane_idx = 0; plane_idx < static_cast<int>(planes.size()); ++plane_idx) {
            const auto& plane = planes[plane_idx];
            PlaneMatchResult& result = plane_results[plane_idx];
            result.plane_idx = plane_idx;

#ifdef DEBUG_PLANE_MATCH            
            std::string score_filename = "points_3d/L" + 
                            std::to_string(l_laser_idx) + "_P" +
                            std::to_string(plane_idx) + "_scores.txt";
            std::ofstream score_ofs(score_filename);
#endif

            // 处理当前激光线上的每个点
            for (const auto& [y, x] : l_points) {
#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
#endif
                ReprojectionInfo info(x, y, -1, -1);
                
                // 光线方向计算
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
                else {  // 二次曲面求交
                    auto intersections = findIntersection(ray_origin, ray_dir, plane.coefficients);
                    if (intersections.empty()) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无交点的点
                    }
                    
                    // 找到有效交点
                    bool valid_pt_found = false;
                    for (const auto& p : intersections) {
                        // float err = evaluateQuadSurf(plane.coefficients, p);
                        if (p.z > 100 && p.z < 1200) {
                            pt = p;
                            valid_pt_found = true;
                        }
                    }
                    
                    if (!valid_pt_found) {
                        result.reprojected_points.push_back(info);
                        continue; // 跳过无有效点的点
                    }
                }

                // 重投影到右图像
                cv::Point3f pt_r(pt.x - baseline, pt.y, pt.z);
                float x_r = fx_r * pt_r.x / pt_r.z + cx_r;
                float y_r = std::round(fy_r * pt_r.y / pt_r.z + cy_r);

                // 更新重投影信息
                info.x_right = x_r;
                info.y_right = y_r;

                // 跳过图像范围外的点
                if (x_r < 0 || x_r >= rectify_r.cols || y_r < 0 || y_r >= rectify_r.rows) {
                    continue;
                }

#ifdef DEBUG_PLANE_MATCH
                cv::circle(vis_img, cv::Point(cvRound(x_r) + cols, cvRound(y_r)), 1, cv::Scalar(0, 0, 255), -1);
#endif

                // 在右激光线中寻找匹配
                bool found_candidate = false;
                for (int r_laser_idx = 0; r_laser_idx < static_cast<int>(laser_r.size()); ++r_laser_idx) {
                    // 跳过已使用的激光线
                    if (used_r_lines.find(r_laser_idx) != used_r_lines.end()) continue;
                    
                    const auto& r_line = laser_r[r_laser_idx].points;
                    if (r_line.empty()) continue;

                    // 在当前激光线上找最近点
                    float min_dist = FLT_MAX;
                    auto it = r_line.find(y_r);
                    if (it != r_line.end()) {
                        min_dist = std::hypot(it->second.x - x_r, it->second.y - y_r);
                    }
                    
                    // 找到阈值范围内的匹配点
                    if (min_dist != FLT_MAX) {
                        found_candidate = true;
                        info.r_scores[r_laser_idx] = min_dist;
                        
                        // 更新匹配映射
                        MatchKey key{l_laser_idx, plane_idx, r_laser_idx};
                        match_map[key].score_sum += min_dist; // 使用正分数方便计算
                        match_map[key].count += 1;
                        
                        // 更新该平面的匹配统计
                        result.r_line_scores[r_laser_idx].score_sum += min_dist;
                        result.r_line_scores[r_laser_idx].count += 1;
                    }
                }
                
                if (found_candidate) {
                    result.reprojected_points.push_back(info);
                    result.point_count++;
                }
            } // 结束点循环
            
            // 分析当前平面的匹配结果
            if (result.point_count > 0) {
                float best_score = FLT_MAX; // 注意：这里分数越小越好
                float second_best = FLT_MAX;
                int best_r_idx = -1;

                // 找出最佳和次佳匹配
                for (auto& [r_idx, score_acc] : result.r_line_scores) {
                    if (score_acc.count == 0) continue;
                    
                    float avg_score = score_acc.score_sum / score_acc.count;

                    if (avg_score < best_score) {
                        second_best = best_score;
                        best_score = avg_score;
                        best_r_idx = r_idx;
                    } 
                    else if (avg_score < second_best) {
                        second_best = avg_score;
                    }
                }
                
                // 记录结果
                result.best_r_idx = best_r_idx;
                result.avg_score = best_score;
                float score_gap = (second_best == FLT_MAX) ? 0 : (second_best - best_score);
#ifdef DEBUG_PLANE_MATCH                
                // 输出平面匹配信息
                data_ofs << plane_idx << " | " << result.best_r_idx << " | " 
                         << result.avg_score << " | " << result.point_count << " | " 
                         << l_points.size() << "\n";
#endif
                
                // 详细点信息
                for (const auto& info : result.reprojected_points) {
#ifdef DEBUG_PLANE_MATCH
                    data_ofs << "    P: (" << info.x_left << ", " << info.y_left << ") -> (" 
                             << info.x_right << ", " << info.y_right << ") | R Scores: [";
                    for (const auto& [r_idx, score] : info.r_scores) {
                        data_ofs << r_idx << ":" << score << " ";
                    }
                    data_ofs << "]\n";

                    // 输出当前最佳分数
                    if (result.best_r_idx >= 0 && info.r_scores.count(result.best_r_idx)) {
                        score_ofs << info.r_scores.at(result.best_r_idx) << "\n";
                    }
#endif

                }
                
#ifdef DEBUG_PLANE_MATCH
                score_ofs.close();                
                printf("左激光线 %d | 平面 %d | 最佳右线: %d | 平均距离: %.3f | 点覆盖率: %.1f%%\n",
                       l_laser_idx, plane_idx, best_r_idx, best_score, 
                       100.0f * result.point_count / l_points.size());
#endif
                
                // 如果找到有效匹配候选
                if (best_r_idx >= 0) {
                    CandidateMatch candidate{
                        plane_idx, best_r_idx, best_score, score_gap
                    };
                    current_unmatched.candidates.push_back(candidate);
                    
                }
            }
        } // 结束平面循环
        
        // ======================= 开始：平面循环后的决策 =======================
        if (!current_unmatched.candidates.empty()) {
            // 1. 找出全局最佳候选（所有平面中得分最低的）
            auto best_candidate = current_unmatched.candidates.end();
            float min_score = FLT_MAX;
            float second_min_score = FLT_MAX;
            
            for (auto it = current_unmatched.candidates.begin(); it != current_unmatched.candidates.end(); ++it) {
                if (it->avg_score < min_score) {
                    second_min_score = min_score;
                    min_score = it->avg_score;
                    best_candidate = it;
                } else if (it->avg_score < second_min_score) {
                    second_min_score = it->avg_score;
                }
            }
            
            if (best_candidate != current_unmatched.candidates.end()) {
                const auto& cand = *best_candidate;
                
                // 2. 计算与次优候选的差距
                float gap = (second_min_score != FLT_MAX) ? (second_min_score - cand.avg_score) : 0;
                
                // 3. 检查锁定条件
                bool should_lock = false;
                std::string lock_reason;
                
                // 条件1：距离<5且差距>10
                if (cand.avg_score < 7.65f && gap > 10.0f) {
                    should_lock = true;
                    lock_reason = "距离<5且差距>10";
                }
                // 条件2：唯一候选且距离<5
                else if (current_unmatched.candidates.size() == 1 && cand.avg_score < 7.65f) {
                    should_lock = true;
                    lock_reason = "唯一候选且距离<5";
                }
                
                // 4. 应用锁定决策
                if (should_lock && used_r_lines.find(cand.r_laser_idx) == used_r_lines.end()) {
                    // 记录匹配结果
                    match_results.emplace_back(l_laser_idx, cand.plane_idx, cand.r_laser_idx);

                    used_r_lines.insert(cand.r_laser_idx);
                    final_matches[l_laser_idx] = std::make_tuple(cand.plane_idx, cand.r_laser_idx, cand.avg_score);
                    has_match = true;

#ifdef DEBUG_PLANE_MATCH
                    printf("   [锁定匹配] 平面%d→R%d 距离:%.2f 差距:%.2f %s\n", 
                        cand.plane_idx, cand.r_laser_idx, cand.avg_score, gap, lock_reason.c_str());
#endif
                }
            }
        }
        // ======================= 结束：平面循环后的决策 =======================

        // 如果未锁定，加入未匹配列表
        if (!has_match) {
            current_unmatched.l_laser_idx = l_laser_idx;
            unmatched_info.push_back(current_unmatched);

#ifdef DEBUG_PLANE_MATCH
            printf("   [待定] 添加到未匹配列表 (候选数: %d)\n", 
                static_cast<int>(current_unmatched.candidates.size()));
#endif
        }

#ifdef DEBUG_PLANE_MATCH
        data_ofs.close();
#endif
        
#ifdef DEBUG_PLANE_MATCH
        // 可视化候选激光线
        for (const auto& [key, acc] : match_map) {
            if (key.l_laser_idx != l_laser_idx) continue;
            if (acc.count < 5) continue;  // 最小点数阈值

            const auto& r_points = laser_r[key.r_laser_idx].points;
            cv::Scalar color;
            std::string status;
            
            if (used_r_lines.find(key.r_laser_idx) != used_r_lines.end()) {
                color = cv::Scalar(0, 255, 255); // 已占用-黄色
                status = "be occupied";
            } 
            else if (has_match && key.r_laser_idx == std::get<1>(final_matches[l_laser_idx])) {
                color = cv::Scalar(128, 0, 128); // 当前匹配-紫色
                status = "select";
            } 
            else {
                color = cv::Scalar(0, 180, 0); // 候选-绿色
                status = "wait";
            }

            // 绘制激光线点
            for (const auto& [y, pt] : r_points) {
                cv::circle(vis_img, cv::Point(cvRound(pt.x) + cols, y), 2, color, -1);
            }
            
            // 标注激光线ID
            if (!r_points.empty()) {
                auto mid_it = r_points.lower_bound(rectify_r.rows / 2);
                if (mid_it == r_points.end()) mid_it = std::prev(r_points.end());
                
                cv::putText(vis_img, "R" + std::to_string(key.r_laser_idx),
                            cv::Point(mid_it->second.x + cols + 10, mid_it->first),
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, color, 1);
            }
        }
    

        // 显示匹配状态
        std::string status = has_match ? 
            "Best Match: L" + std::to_string(l_laser_idx) +
            " - R" + std::to_string(std::get<1>(final_matches[l_laser_idx])) +
            " (P " + std::to_string(std::get<0>(final_matches[l_laser_idx])) + 
            " / S " + std::to_string(std::get<2>(final_matches[l_laser_idx])) + ")" :
            "L" +  std::to_string(l_laser_idx) + " Best Match: Wait";


        
        cv::putText(vis_img, status,
            cv::Point((vis_img.cols - 500) / 2, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        cv::namedWindow("Sample Points Projection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Sample Points Projection", vis_img.cols, vis_img.rows);
        cv::imshow("Sample Points Projection", vis_img);
        if (cv::waitKey(0) == 27) break;  // ESC 跳出所有可视化
#endif
    } // 结束左激光线循环

    // 第二轮匹配：处理未匹配项
#ifdef DEBUG_PLANE_MATCH
    printf("\n===== 第二轮匹配开始 (待处理: %d) =====\n", static_cast<int>(unmatched_info.size()));
#endif
    
    for (const auto& info : unmatched_info) {
        int best_plane = -1;
        int best_r = -1;
        float best_score = FLT_MAX;
        bool found = false;
        
        for (const auto& cand : info.candidates) {
            // 跳过已被占用的右激光线
            if (used_r_lines.find(cand.r_laser_idx) != used_r_lines.end()) continue;
            
            // 寻找最佳可用匹配
            if (cand.avg_score < best_score) {
                best_score = cand.avg_score;
                best_r = cand.r_laser_idx;
                best_plane = cand.plane_idx;
                found = true;
            }
        }
        
        // 应用阈值
        if (found) {
            if (best_score < 7.65f) { // 小于5才匹配
                // 记录匹配结果
                match_results.emplace_back(info.l_laser_idx, best_plane, best_r);

                used_r_lines.insert(best_r);
                final_matches[info.l_laser_idx] = std::make_tuple(best_plane, best_r, best_score);
                printf("左激光线 %d: 匹配到 平面%d -> R%d (距离: %.3f)\n", 
                      info.l_laser_idx, best_plane, best_r, best_score);
            } else {
                printf("左激光线 %d: 最佳候选距离(%.3f)>=5，放弃匹配\n", info.l_laser_idx, best_score);
            }
        } else {
            printf("左激光线 %d: 无有效候选\n", info.l_laser_idx);
        }
    }
    
#ifdef DEBUG_PLANE_MATCH
    // 生成最终匹配报告
    printf("\n===== 最终匹配结果 =====\n");
    for (int i = 0; i < final_matches.size(); ++i) {
        const auto& match = final_matches[i];
        if (std::get<1>(match) >= 0) {
            printf("左线 %2d -> 右线 %2d (平面%d, 平均距离: %.2f像素)\n", 
                  i, std::get<1>(match), std::get<0>(match), std::get<2>(match));
        } else {
            printf("左线 %2d -> 未匹配\n", i);
        }
    }
    printf("\n======================\n");
#endif

    return match_results;
}


/*************************************************************************************** */



/************************************** Match Four ***************************************/
struct MatchCandidate {
    int l_idx;
    int plane_idx;
    int r_idx;
    float score;
    float avg_dist;
    float coverage;
};

float LaserProcessor::computeCompScore4(float avgDist, float coverage, float wD, float wC) {
    return wD * avgDist + wC * (1.0f - coverage);
}


std::vector<std::tuple<int,int,int>> LaserProcessor::match4(
    const std::vector<std::map<float,float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r)
{
    const auto calib = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();
    const double fx_l=calib.P[0].at<double>(0,0), fy_l=calib.P[0].at<double>(1,1);
    const double cx_l=calib.P[0].at<double>(0,2), cy_l=calib.P[0].at<double>(1,2);
    const double fx_r=calib.P[1].at<double>(0,0), fy_r=calib.P[1].at<double>(1,1);
    const double cx_r=calib.P[1].at<double>(0,2), cy_r=calib.P[1].at<double>(1,2);
    const double baseline=-calib.P[1].at<double>(0,3)/fx_r;


    int L = sample_points.size();
    std::vector<std::vector<MatchCandidate>> allCands(L);
    std::set<int> used_r;
    std::vector<bool> locked(L, false);
    std::vector<std::tuple<int,int,int>> match_results;

    // 收集候选
    for(int l=0; l<L; ++l) {
        const auto& pts = sample_points[l];
        for(int p=0; p<(int)planes.size(); ++p) {
            int inCnt=0;
            std::map<int,int> hits;
            std::map<int,std::pair<float,int>> acc;
            auto& coef=planes[p].coefficients;
            for(auto [y,x]:pts) {
                cv::Point3f ray((x-cx_l)/fx_l, (y-cy_l)/fy_l,1.0f);
                ray *= 1.0f/cv::norm(ray);
                auto ips = findIntersection({0,0,0}, ray, coef);
                if(ips.empty()) continue;
                cv::Point3f pt3; bool ok=false;
                for(auto& q:ips) if(q.z>100&&q.z<1200){ pt3=q; ok=true; break; }
                if(!ok) continue;
                cv::Point3f pr(pt3.x-baseline, pt3.y, pt3.z);
                float xr = fx_r*pr.x/pr.z + cx_r;
                int yr = cvRound(fy_r*pr.y/pr.z + cy_r);
                if(xr<0||xr>=rectify_r.cols||yr<0||yr>=rectify_r.rows) continue;
                ++inCnt;
                for(int r=0; r<(int)laser_r.size(); ++r) {
                    if(used_r.count(r)) continue;
                    auto it=laser_r[r].points.find(yr);
                    if(it==laser_r[r].points.end()) continue;
                    float d = hypot(it->second.x - xr, it->second.y - yr);
                    hits[r]++;
                    acc[r].first += d;
                    acc[r].second++;
                }
            }
            if(!inCnt) continue;
            for(auto& [r,c]:hits) {
                auto pr = acc[r];
                float avg = pr.first/pr.second;
                float cov = float(c)/inCnt;
                float comp = computeCompScore4(avg, cov);
                allCands[l].push_back({l,p,r,comp,avg,cov});
            }
        }
    }

    // 第一次统一可视化
#ifdef DEBUG_PLANE_MATCH
    cv::Mat vis_global;
    cv::hconcat(rectify_l, rectify_r, vis_global);
    if(vis_global.channels()==1) cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
    int off = rectify_l.cols;
    cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
    cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif
    
    for(int l=0; l<L; ++l) {
        if(allCands[l].empty()) continue;

#ifdef DEBUG_PLANE_MATCH
        cv::Mat vis = vis_global.clone();
        // 左激光线点
        for(auto [y,x]:sample_points[l])
            cv::circle(vis, cv::Point2f(x,y), 1.5, cv::Scalar(0,255,0), -1);
        // 左激光线ID
        int l_middle = sample_points[l].size() / 2;
        auto l_mid_it = sample_points[l].begin();
        std::advance(l_mid_it, l_middle);
        cv::putText(vis, "L"+std::to_string(l), cv::Point2f(l_mid_it->second,l_mid_it->first), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,255,0), 2);
#endif
        // 按 plane_idx 提取最佳候选
        std::map<int, MatchCandidate> bestOfPlane;
        for(auto& m: allCands[l]) {
            auto& b = bestOfPlane[m.plane_idx];
            if(b.score==0 || m.score < b.score) b = m;
        }
        // 全局最佳
        MatchCandidate globalBest = bestOfPlane.begin()->second;
        for(auto& kv: bestOfPlane)
            if(kv.second.score < globalBest.score) globalBest = kv.second;

#ifdef DEBUG_PLANE_MATCH
        // 重投影点（红）
        for(auto& kv: bestOfPlane) {
            auto m = kv.second;
            auto& coef = planes[m.plane_idx].coefficients;
            for(auto [y,x]: sample_points[l]) {
                cv::Point3f ray((x-cx_l)/fx_l, (y-cy_l)/fy_l,1.0f);
                ray*=1.0f/cv::norm(ray);
                auto ips = findIntersection({0,0,0}, ray, coef);
                if(ips.empty()) continue;
                cv::Point3f pt3; bool ok=false;
                for(auto& q:ips) if(q.z>100&&q.z<1200){pt3=q; ok=true; break;} if(!ok) continue;
                cv::Point3f pr(pt3.x-baseline, pt3.y, pt3.z);
                float xr = fx_r*pr.x/pr.z+cx_r; int yr=cvRound(fy_r*pr.y/pr.z+cy_r);
                if(xr<0||xr>=rectify_r.cols||yr<0||yr>=rectify_r.rows) continue;
                cv::circle(vis, cv::Point2f(xr+off, yr), 1.5, cv::Scalar(0,0,255), -1);
            }
        }

        // 绘制右激光线点和ID
        for(auto& kv: bestOfPlane) {
            auto m = kv.second;
            bool isGlobal = (m.r_idx==globalBest.r_idx);
            cv::Scalar col = used_r.count(m.r_idx)? cv::Scalar(0,255,255)
                            : isGlobal? cv::Scalar(128,0,128)
                            : cv::Scalar(0,180,0);
            for(auto& [y,pt]: laser_r[m.r_idx].points)
                cv::circle(vis, cv::Point2f(pt.x+off, y), 1.5, col, -1);
            auto it = laser_r[m.r_idx].points.begin();
            int r_middle = laser_r[m.r_idx].points.size() / 2;
            std::advance(it, r_middle);
            if(it==laser_r[m.r_idx].points.end()) it=std::prev(laser_r[m.r_idx].points.end());
            cv::putText(vis, "R"+std::to_string(m.r_idx), cv::Point2f(it->second.x+off, it->first),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, col,2);
        }

        // 状态文本
        char buf[128];
        snprintf(buf, sizeof(buf), "Round1: L%d->R%d (P %d / S %.2f)",
                 globalBest.l_idx, globalBest.r_idx,
                 globalBest.plane_idx, globalBest.score);
        cv::putText(vis, buf, {10,60}, cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0,255,255),2);
        cv::imshow("vis_img", vis);
        cv::waitKey(0);
#endif
    }

    // 多轮严格锁定
    bool prog=true;
    while(prog) {
        prog=false;
        for(int l=0;l<L;++l) if(!locked[l]){
            std::vector<MatchCandidate> av;
            for(auto& m: allCands[l]) if(!used_r.count(m.r_idx)) av.push_back(m);
            if(av.empty()) continue;
            std::sort(av.begin(), av.end(), [](auto&a,auto&b){return a.score<b.score;});
            float b0=av[0].score, b1=av.size()>1?av[1].score:FLT_MAX;
            if(b0<=10.0f && (av.size()==1 || b1-b0>=10.0f)) {
                locked[l]=true;
                used_r.insert(av[0].r_idx);
                match_results.emplace_back(l,av[0].plane_idx,av[0].r_idx);
                prog=true;
                printf("   [锁定匹配] 左%d 平面%d→右%d 平均距离:%.2f 与次优差距:%.2f 综合得分:%.2f 覆盖率:%.2f%%\n",
                       l,av[0].plane_idx,av[0].r_idx,av[0].avg_dist,b1-b0,av[0].score,av[0].coverage*100.0f);
            }
        }
    }


#ifdef DEBUG_PLANE_MATCH    
    // 全局可视化锁定结果
    static const std::vector<cv::Scalar> palette30 = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
        {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
        {64, 0, 128}, {128, 64, 0}, {0, 128, 64}, {64, 128, 0}, {0, 64, 128}, {128, 0, 64},
        {192, 192, 0}, {192, 0, 192}, {64, 255, 0}, {255, 64, 0}, {0, 64, 255}, {0, 255, 64},
        {255, 0, 64}, {64, 0, 255}, {192, 0, 64}, {64, 192, 0}, {0, 192, 64}, {64, 0, 192}
    };
    // 标注左线ID
    for (int l = 0; l < L; ++l) {
        int mid = sample_points[l].size() / 2;
        auto it = sample_points[l].begin();
        std::advance(it, mid);
        cv::putText(vis_global, "L" + std::to_string(l),
                    cv::Point2f(it->second, it->first),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,255,0), 2);
    }
    // 标注右线ID
    for (int r = 0; r < (int)laser_r.size(); ++r) {
        int mid = laser_r[r].points.size() / 2;
        auto it = laser_r[r].points.begin();
        std::advance(it, mid);
        cv::putText(vis_global, "R" + std::to_string(r),
                    cv::Point2f(it->second.x + off, it->first),
                    cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0,255,255), 2);
    }

    // 仅为配对线条着色（不绘制连线）
    for (int idx = 0; idx < match_results.size(); ++idx) {
        auto [l, p, r] = match_results[idx];
        cv::Scalar clr = palette30[idx % palette30.size()];
        // 左线点着色
        for (auto& [y, x] : sample_points[l]) {
            cv::circle(vis_global, cv::Point(cvRound(x), y), 3, clr, -1);
        }
        // 右线点着色
        for (auto& [y, pt] : laser_r[r].points) {
            cv::circle(vis_global, cv::Point(cvRound(pt.x) + off, y), 3, clr, -1);
        }
    }
    cv::putText(vis_global, "vis_img", {10,30}, cv::FONT_HERSHEY_SIMPLEX,1.3, cv::Scalar(255,0,255),2);
    cv::imshow("vis_img", vis_global);
    cv::waitKey(0);
    cv::destroyWindow("vis_img");
#endif

    return match_results;
}
/*************************************************************************************** */


/************************************** Match Five **************************************/
// 改进的得分计算函数，考虑距离方差和长度归一化
float LaserProcessor::computeEnhancedScore(
    const std::vector<std::pair<float, float>>& distance_pairs,
    int pts_repro_cnt) {
    
    if (distance_pairs.empty()) return FLT_MAX;
    
    // 1. 计算平均距离
    float sum_dist = 0.0f;
    for (const auto& [y, d] : distance_pairs) {
        sum_dist += d;
    }
    float avg_dist = sum_dist / distance_pairs.size();
    
    // 2. 计算距离方差
    float variance = 0.0f;
    for (const auto& [y, d] : distance_pairs) {
        float diff = d - avg_dist;
        variance += diff * diff;
    }
    variance /= distance_pairs.size();
    float std_dev = std::sqrt(variance);
    
    // 3. 计算覆盖率
    float coverage = float(distance_pairs.size()) / pts_repro_cnt;
    
    // 4. 长度归一化惩罚
    // 对短线施加更严格的要求
    float length_penalty = 1.0f;
    if (pts_repro_cnt < 60) {
        // 短线惩罚：线越短，惩罚越重
        length_penalty = 1.0f + (60.0f - pts_repro_cnt) * 0.05f;
    }
    
    // 5. 距离一致性惩罚（进一步增大惩罚力度）
    // 标准差大说明距离分布不均匀，应该被重点惩罚
    float consistency_penalty = 1.0f + std_dev * 1.5f;
    
    // 对于标准差过大的情况，施加更严厉的惩罚
    if (std_dev > 2.5f) {
        // 指数增长的惩罚，严重不一致的匹配会被大幅惩罚
        consistency_penalty *= (1.0f + std::pow(std_dev - 2.5f, 1.5f) * 0.3f);
    }
    
    // 6. 综合得分计算（越低越好）
    // 基础得分：平均距离 + 覆盖率惩罚（进一步减少覆盖率惩罚影响）
    // float base_score = avg_dist * 0.5f + (1.0f - coverage) * 1.2f;  // 覆盖率权重进一步降到1.2f
    float base_score = avg_dist * 0.1f + (1.0f - coverage) * 3.5f; 
    
    // printf("avg_dist: %.2f / std_dev: %.2f / cov: %.2f / base_score: %.2f\n", avg_dist, std_dev, coverage, base_score);

    // 应用惩罚因子
    float final_score = base_score * consistency_penalty * length_penalty;
    
    return final_score;
}



std::vector<IntervalMatch> LaserProcessor::match5(
    const std::vector<std::map<float,float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r)
{
    // 1. 相机与阈值初始化
    const auto calib = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();
    double fx_l = calib.P[0].at<double>(0,0), fy_l = calib.P[0].at<double>(1,1);
    double cx_l = calib.P[0].at<double>(0,2), cy_l = calib.P[0].at<double>(1,2);
    double fx_r = calib.P[1].at<double>(0,0), fy_r = calib.P[1].at<double>(1,1);
    double cx_r = calib.P[1].at<double>(0,2), cy_r = calib.P[1].at<double>(1,2);
    double baseline = -calib.P[1].at<double>(0,3) / fx_r;

    constexpr float D_thresh = 20.0f;
    constexpr float S_thresh = 20.0f;
    constexpr float Delta_thresh = 3.0f;
    constexpr int MIN_LEN = 45; // 存在问题：大一些会过滤掉正确导致匹配错误，小一些容易配错

    int L = (int)sample_points.size();
    int R = (int)laser_r.size();

    // 结果与锁定区间
    std::vector<IntervalMatch> final_matches;
    std::vector<std::vector<Interval>> locked_l(L), locked_r(R);
    
    // lambda: 合并已锁定区间
    auto merge_intervals = [&](std::vector<Interval>& v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end(), [](auto &a, auto &b){ return a.y_start < b.y_start; });
        std::vector<Interval> merged;
        Interval cur = v[0];
        for (int i = 1; i < (int)v.size(); ++i) {
            auto &n = v[i];
            if (n.y_start <= cur.y_end + EPS) {
                cur.y_end = std::max(cur.y_end, n.y_end);
                cur.count += n.count;
            } else {
                merged.push_back(cur);
                cur = n;
            }
        }
        merged.push_back(cur);
        v.swap(merged);
    };

#ifdef DEBUG_PLANE_MATCH_FINAL
    // 全局可视化底图
    cv::Mat vis_global;
    cv::hconcat(rectify_l, rectify_r, vis_global);
    if (vis_global.channels() == 1)
        cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
    int off = rectify_l.cols;
    static const cv::Scalar proc_interval_col(0,255,255); // 候选区间颜色
    static const cv::Scalar r_laser_col(255,0,0); // 整条右线颜色
    cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
    cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

    // --- 阶段：唯一候选先轮完
    bool progress = true;
    while (progress) {
        progress = false;
        for(int l=0; l<L; ++l) {
            const auto& pts = sample_points[l];
            int repro_cnt_total = 0;
            std::vector<IntervalMatch> cands;

            // 针对每个平面+右线构建支持集合
            for(int p=0; p<(int)planes.size(); ++p) {
                int repro_cnt = 0;
                const auto& coef = planes[p].coefficients;
                std::map<int,std::vector<std::pair<float,float>>> support; // r_idx->(y,distance)
                // if (l == 0 && p == 5)
                //     puts("");
                for(auto [y_f, x_f]: pts) {
                    // 左侧已锁区跳过
                    bool skipL = false;
                    for (auto &iv : locked_l[l]) {
                        if (y_f >= iv.y_start - EPS && y_f <= iv.y_end + EPS) { skipL = true; break; }
                    }
                    if (skipL) continue;
                    // 重投影
                    cv::Point3f ray((x_f-cx_l)/fx_l, (y_f-cy_l)/fy_l,1.0f);
                    ray *= 1.0f/cv::norm(ray);
                    auto ips = findIntersection({0,0,0}, ray, coef);
                    if (ips.empty()) continue;
                    cv::Point3f pt3; bool ok=false;
                    for (auto &q: ips) if (q.z>100 && q.z<1500) { pt3=q; ok=true; break; }
                    if (!ok) continue;
                    cv::Point3f pr(pt3.x-baseline, pt3.y, pt3.z);
                    float xr = fx_r*pr.x/pr.z + cx_r;
                    float yr = alignToPrecision(fy_r*pr.y/pr.z + cy_r);
                    if (xr<0 || xr>=rectify_r.cols || yr<0 || yr>=rectify_r.rows) continue;
                    repro_cnt++; repro_cnt_total++;
                    // 遍历右线
                    for (int r = 0; r < R; ++r) {
                        auto it = laser_r[r].points.lower_bound(yr);
                        if (it != laser_r[r].points.begin()) {
                            auto prev = std::prev(it);
                            if (it == laser_r[r].points.end() || fabs(prev->first-yr) < fabs(it->first-yr))
                                it = prev;
                        }
                        if (fabs(it->first - yr) > EPS) continue;
                        float d = hypot(it->second.x - xr, it->second.y - yr);
                        if (d > D_thresh) continue;
                        bool skipR = false;
                        for (auto &iv : locked_r[r]) {
                            if (yr >= iv.y_start - EPS && yr <= iv.y_end + EPS) { skipR = true; break; }
                        }
                        if (!skipR) support[r].emplace_back(yr, d);
                    }
                }
                if (support.empty()) continue;
                // 对每个右线构建子段集
                for (auto &ent : support) {
                    auto &vec = ent.second;
                    if ((int)vec.size() < MIN_LEN) continue;
                    std::sort(vec.begin(), vec.end(), [](auto &a, auto &b){ return a.first < b.first; });
                    // 拆分子段
                    std::vector<Interval> segs;
                    int start=0;
                    for (int i=1; i<(int)vec.size(); ++i) {
                        float gap = vec[i].first - vec[i-1].first;
                        if (gap > 2*precision + EPS) {
                            Interval iv{ alignToPrecision(vec[start].first),
                                         alignToPrecision(vec[i-1].first),
                                         i-start };
                            segs.push_back(iv);
                            start = i;
                        }
                    }
                    Interval ivlast{ alignToPrecision(vec[start].first),
                                     alignToPrecision(vec.back().first),
                                     (int)vec.size()-start };
                    segs.push_back(ivlast);
                    if (segs.empty()) continue;
                    // 汇总此 (l,p,r)所有子段
                    int total_count=0;
                    for (auto &iv: segs) total_count += iv.count;
                    if (total_count < MIN_LEN) continue;
                    float coverage = total_count / float(repro_cnt);
                    // 统一打分
                    std::vector<std::pair<float,float>> allpd;
                    for (auto &pd : vec) {
                        for (auto &iv : segs) 
                        {
                            if (pd.first>=iv.y_start-EPS && pd.first<=iv.y_end+EPS) { allpd.push_back(pd); break; }
                        }
                    }
                    float score = computeEnhancedScore(allpd, repro_cnt);
                    if (score <= S_thresh) {
                        cands.push_back({l, p, ent.first, segs, score, coverage});
                    }
                }
            }
            if (cands.empty() || cands[0].score < 0) continue;
            
            // 如果唯一候选
            bool lock=false;
            const auto& m = cands[0];
            if(cands.size() == 1 && m.score <= 5.0f) lock = true;
            else if (cands.size() > 1) {
                std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.score<b.score;});
                const auto& mm = cands[1];
                if (m.p_idx == mm.p_idx && mm.score - m.score <= 1.5) lock = true;
                else if (mm.score - m.score >= 5.0f && m.score <= 12.0f && m.coverage >= 0.65) lock = true;
            }
            if(lock) {
                final_matches.push_back(m);
                locked_l[m.l_idx].insert(locked_l[m.l_idx].end(),m.intervals.begin(),m.intervals.end());
                merge_intervals(locked_l[m.l_idx]);
                locked_r[m.r_idx].insert(locked_r[m.r_idx].end(),m.intervals.begin(),m.intervals.end());
                merge_intervals(locked_r[m.r_idx]);
                progress=true;
            }
        
            // 过程可视化
#ifdef DEBUG_PLANE_MATCH
            cv::Mat vis = vis_global.clone();
            // 左图激光点
            for (auto [y, x] : pts) cv::circle(vis, cv::Point2f(x, y), 1.5, {0,255,0}, -1);
            // 左线ID
            auto l_mid = pts.begin(); std::advance(l_mid, pts.size()/2);
            cv::putText(vis, "L" + std::to_string(l), cv::Point2f(l_mid->second, l_mid->first),
                        cv::FONT_HERSHEY_SIMPLEX, 1.3, {0,255,0}, 2);
            // 重投影点（红）及候选区间（黄）
            cv::Point2i test_point(10, 40);
            for (auto& c : cands) {
                // 蓝色整条右线
                for (auto& p_r : laser_r[c.r_idx].points)
                    cv::circle(vis,cv::Point2f(p_r.second.x+off,p_r.second.y),1.5,r_laser_col,-1);
                // 红色投影
                const auto& coef_c = planes[c.p_idx].coefficients;
                for (auto [y_f, x_f] : pts) {
                    cv::Point3f ray((x_f-cx_l)/fx_l, (y_f-cy_l)/fy_l,1.0f);
                    ray *= 1.0f/cv::norm(ray);
                    auto ips = findIntersection({0,0,0}, ray, coef_c);
                    if (ips.empty()) continue;
                    cv::Point3f pt3; bool ok=false;
                    for (auto& q:ips) if(q.z>100&&q.z<1200){pt3=q;ok=true;break;}
                    if(!ok) continue;
                    cv::Point3f pr(pt3.x-baseline, pt3.y, pt3.z);
                    float xr = fx_r*pr.x/pr.z + cx_r;
                    float yr = alignToPrecision(fy_r*pr.y/pr.z + cy_r);
                    if (xr>=0&&xr<rectify_r.cols&&yr>=0&&yr<rectify_r.rows)
                        cv::circle(vis,cv::Point2f(xr+off,yr),1.5,{0,0,255},-1);
                }
                // 黄色区间
                for (auto& iv : c.intervals)
                    for (float y0 = iv.y_start; y0<=iv.y_end; y0+=precision) {
                        auto it_r = laser_r[c.r_idx].points.lower_bound(y0);
                        if (it_r!=laser_r[c.r_idx].points.begin()) {
                            auto prev = std::prev(it_r);
                                if (it_r == laser_r[c.r_idx].points.end() ||
                                    fabs(prev->first - y0) < fabs(it_r->first - y0)) {
                                    it_r = prev;
                                }
                        }
                        if (fabs(it_r->first - y0) > EPS) continue;
                        cv::circle(vis,cv::Point2f(it_r->second.x+off,y0),1.5,proc_interval_col,-1);
                    }
                // 右线ID
                auto r_mid = laser_r[c.r_idx].points.begin(); std::advance(r_mid, laser_r[c.r_idx].points.size()/2);
                cv::putText(vis, "R" + std::to_string(c.r_idx), cv::Point2f(r_mid->second.x+off, r_mid->second.y),
                            cv::FONT_HERSHEY_SIMPLEX, 1.3, r_laser_col, 2);
                // 文本：候选数
                char text_buf[128];
                snprintf(text_buf, sizeof(text_buf), "L%d->P%d->R%d (S %.2f / C %.2f%%)",
                        c.l_idx, c.p_idx, c.r_idx, c.score, c.coverage*100);
                cv::putText(vis, text_buf, test_point,cv::FONT_HERSHEY_SIMPLEX,1.2,{0,255,255},2);
                test_point.y += 45;
            }
            cv::imshow("vis_img", vis);
            cv::waitKey(0);
#endif

        }
    }


#ifdef DEBUG_PLANE_MATCH_FINAL
    cv::Mat vis_all;
    cv::hconcat(rectify_l, rectify_r, vis_all);
    if (vis_all.channels()==1) cv::cvtColor(vis_all, vis_all, cv::COLOR_GRAY2BGR);
    int off_all = rectify_l.cols;
    static const std::vector<cv::Scalar> palette30 = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
        {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
        {64, 0, 128}, {128, 64, 0}, {0, 128, 64}, {64, 128, 0}, {0, 64, 128}, {128, 0, 64},
        {192, 192, 0}, {192, 0, 192}, {64, 255, 0}, {255, 64, 0}, {0, 64, 255}, {0, 255, 64},
        {255, 0, 64}, {64, 0, 255}, {192, 0, 64}, {64, 192, 0}, {0, 192, 64}, {64, 0, 192}
    };
    for (int idx=0; idx<(int)final_matches.size(); ++idx) {
        auto& m = final_matches[idx];
        cv::Scalar col = palette30[idx % palette30.size()];
        // 左图区间点
        for (auto& iv : m.intervals)
            for (float y0=iv.y_start; y0<=iv.y_end; y0+=precision) {
                auto itL = sample_points[m.l_idx].lower_bound(y0);
                if (itL != sample_points[m.l_idx].begin()) {
                    // 比较 it 和前一个元素哪个更接近 y0
                    auto prev = std::prev(itL);
                    if (itL == sample_points[m.l_idx].end() || fabs(prev->first - y0) < fabs(itL->first - y0)) {
                        itL = prev;
                    }
                }
                if(fabs(itL->first - y0) > EPS) continue;
                cv::circle(vis_all,cv::Point2f(itL->second,y0),2,col,-1);
            }
        // 右图区间点
        for (auto& iv : m.intervals)
            for (float y0=iv.y_start; y0<=iv.y_end; y0+=precision) {
                auto itR = laser_r[m.r_idx].points.lower_bound(y0);
                if (itR != laser_r[m.r_idx].points.begin()) {
                    // 比较 it 和前一个元素哪个更接近 y0
                    auto prev = std::prev(itR);
                    if (itR == laser_r[m.r_idx].points.end() || fabs(prev->first - y0) < fabs(itR->first - y0)) {
                        itR = prev;
                    }
                }
                if(fabs(itR->first - y0) > EPS) continue;
                cv::circle(vis_all,cv::Point2f(itR->second.x+off_all,y0),2,col,-1);
            }
    }
    cv::imshow("vis_img", vis_all);
    cv::waitKey(0);
    cv::destroyWindow("vis_img");
#endif

    return final_matches;
}
/*************************************************************************************** */


/************************************** Match Six **************************************/
// 浮点安全区间合并
void LaserProcessor::mergeIntervals(std::vector<Interval>& intervals, float prec) const {
    if (intervals.empty()) return;
    
    // 按起点排序
    std::sort(intervals.begin(), intervals.end(), 
        [this](const Interval& a, const Interval& b) {
            return a.y_start < b.y_start - EPS;
        });
    
    std::vector<Interval> merged;
    Interval current = intervals[0];
    
    for (size_t i = 1; i < intervals.size(); ++i) {
        const Interval& next = intervals[i];
        
        // 检查区间重叠或连接（考虑精度间隙）
        if (next.y_start <= current.y_end + prec + EPS) {
            // 合并区间
            current.y_end = std::max(current.y_end, next.y_end);
            current.count += next.count;
        } else {
            // 保存当前区间，开始新区间
            merged.push_back(current);
            current = next;
        }
    }
    merged.push_back(current);
    
    intervals = std::move(merged);
}

// 检查点是否在任意区间内
bool LaserProcessor::isPointLocked(float y, const std::vector<Interval>& intervals) const {
    for (const Interval& iv : intervals) {
        if (y >= iv.y_start - EPS && y <= iv.y_end + EPS) {
            return true;
        }
    }
    return false;
}

// 主匹配函数
std::vector<IntervalMatch> LaserProcessor::match6(
    const std::vector<std::map<float, float>>& sample_points,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r)
{
    // 1. 相机参数初始化
    const auto calib = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();
    
    double fx_l = calib.P[0].at<double>(0, 0);
    double fy_l = calib.P[0].at<double>(1, 1);
    double cx_l = calib.P[0].at<double>(0, 2);
    double cy_l = calib.P[0].at<double>(1, 2);
    
    double fx_r = calib.P[1].at<double>(0, 0);
    double fy_r = calib.P[1].at<double>(1, 1);
    double cx_r = calib.P[1].at<double>(0, 2);
    double cy_r = calib.P[1].at<double>(1, 2);
    
    double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

    // 2. 阈值参数
    constexpr float D_thresh = 12.5f;    // 距离阈值
    constexpr float S_thresh = 20.0f;    // 分数阈值
    constexpr float Delta_thresh = 3.0f; // 分数差阈值
    constexpr int MIN_LEN = 45;          // 最小匹配点数

    // 3. 初始化数据结构
    int L = static_cast<int>(sample_points.size());
    int R = static_cast<int>(laser_r.size());
    
    std::vector<IntervalMatch> final_matches;
    std::vector<std::vector<Interval>> locked_l(L);  // 左线锁定区间
    std::vector<std::vector<Interval>> locked_r(R);  // 右线锁定区间

    // 4. 迭代匹配
    bool progress = true;
    while (progress) {
        progress = false;
        
        // 遍历所有左线
        for (int l = 0; l < L; ++l) {
            const auto& pts = sample_points[l];
            if (pts.empty()) continue;
            
            int repro_cnt_total = 0;
            std::vector<IntervalMatch> cands;

            // 遍历所有平面
            for (int p = 0; p < static_cast<int>(planes.size()); ++p) {
                const auto& coef = planes[p].coefficients;
                int repro_cnt = 0;
                
                // 支持集合: 右线索引 -> (y坐标, 重投影距离)
                std::map<int, std::vector<std::pair<float, float>>> support;
                
                // 遍历左线所有点
                for (const auto& [y_f, x_f] : pts) {
                    // 检查左线点是否已锁定
                    if (isPointLocked(y_f, locked_l[l])) continue;
                    
                    // 光线方程
                    cv::Point3f ray(
                        (x_f - cx_l) / fx_l,
                        (y_f - cy_l) / fy_l,
                        1.0f
                    );
                    ray /= cv::norm(ray);
                    
                    // 求与平面交点
                    auto ips = findIntersection({0, 0, 0}, ray, coef);
                    if (ips.empty()) continue;
                    
                    // 选择有效交点
                    cv::Point3f pt3;
                    bool valid_pt = false;
                    for (const auto& q : ips) {
                        if (q.z > 100.0f && q.z < 1200.0f) {
                            pt3 = q;
                            valid_pt = true;
                            break;
                        }
                    }
                    if (!valid_pt) continue;
                    
                    // 计算右图像坐标
                    cv::Point3f pr(pt3.x - baseline, pt3.y, pt3.z);
                    float xr = fx_r * pr.x / pr.z + cx_r;
                    float yr = alignToPrecision(fy_r * pr.y / pr.z + cy_r);
                    
                    // 检查边界
                    if (xr < 0 || xr >= rectify_r.cols || yr < 0 || yr >= rectify_r.rows) {
                        continue;
                    }
                    
                    repro_cnt++;
                    repro_cnt_total++;
                    
                    // 在右线中查找匹配点
                    for (int r = 0; r < R; ++r) {
                        // 检查右线点是否已锁定
                        if (isPointLocked(yr, locked_r[r])) continue;
                        
                        // 精确查找（已对齐点）
                        auto it = laser_r[r].points.find(yr);
                        if (it == laser_r[r].points.end()) {
                            continue;
                        }
                        
                        // 计算重投影距离
                        float dx = it->second.x - xr;
                        float dy = it->second.y - yr;
                        float distance = std::hypot(dx, dy);
                        
                        if (distance <= D_thresh + EPS) {
                            support[r].emplace_back(yr, distance);
                        }
                    }
                }
                
                if (support.empty()) continue;
                
                // 处理每个右线的支持集合
                for (auto& [r_idx, points] : support) {
                    // 跳过点数不足的集合
                    if (static_cast<int>(points.size()) < MIN_LEN) {
                        continue;
                    }
                    
                    // 按y坐标排序
                    std::sort(points.begin(), points.end(),
                        [this](const auto& a, const auto& b) {
                            return a.first < b.first - EPS;
                        });
                    
                    // 分割连续区间
                    std::vector<Interval> segs;
                    int start_idx = 0;
                    
                    for (size_t i = 1; i < points.size(); ++i) {
                        float gap = points[i].first - points[i-1].first;
                        
                        // 考虑浮点精度和精度间隔
                        if (gap > 2.0f * precision + EPS) {
                            segs.push_back({
                                points[start_idx].first,
                                points[i-1].first,
                                static_cast<int>(i - start_idx)
                            });
                            start_idx = i;
                        }
                    }
                    
                    // 添加最后一个区间
                    segs.push_back({
                        points[start_idx].first,
                        points.back().first,
                        static_cast<int>(points.size() - start_idx)
                    });
                    
                    // 计算总匹配点数
                    int total_count = 0;
                    for (const auto& seg : segs) {
                        total_count += seg.count;
                    }
                    
                    // 跳过总点数不足的候选
                    if (total_count < MIN_LEN) {
                        continue;
                    }
                    
                    // 计算覆盖率
                    float coverage = static_cast<float>(total_count) / repro_cnt;
                    
                    // 计算增强分数（此处简化，实际需实现）
                    float score = computeEnhancedScore(points, repro_cnt);
                    
                    if (score <= S_thresh + EPS) {
                        cands.push_back({
                            l, p, r_idx, segs, score, coverage
                        });
                    }
                }
            }
            
            // 跳过无候选的情况
            if (cands.empty()) {
                continue;
            }
            
            // 候选排序：分数从低到高（分数越低越好）
            std::sort(cands.begin(), cands.end(),
                [this](const IntervalMatch& a, const IntervalMatch& b) {
                    return a.score < b.score - EPS;
                });
            
            // 选择最佳候选
            bool should_lock = false;
            const IntervalMatch* best_match = nullptr;
            
            if (cands.size() == 1) {
                // 唯一候选：分数足够低时锁定
                if (cands[0].score <= 5.0f + EPS) {
                    should_lock = true;
                    best_match = &cands[0];
                }
            } else {
                const IntervalMatch& best = cands[0];
                const IntervalMatch& second = cands[1];
                
                // 情况1：相同平面且分数接近
                if (best.p_idx == second.p_idx && 
                    (second.score - best.score) <= Delta_thresh + EPS) {
                    should_lock = true;
                    best_match = &best;
                }
                // 情况2：分数差距大且覆盖率接近
                else if ((second.score - best.score) >= 5.0f - EPS &&
                            best.score <= 12.0f + EPS &&
                            std::abs(second.coverage - best.coverage) <= 0.4f + EPS) {
                    should_lock = true;
                    best_match = &best;
                }
            }
            
            // 锁定匹配区间
            if (should_lock && best_match != nullptr) {
                final_matches.push_back(*best_match);
                
                // 为左线构建锁定区间
                std::vector<Interval> left_intervals;
                for (const Interval& seg : best_match->intervals) {
                    // 查找左线实际y范围
                    float min_y = FLT_MAX, max_y = -FLT_MAX;
                    for (const auto& [y_f, x_f] : sample_points[l]) {
                        if (y_f >= seg.y_start - EPS && y_f <= seg.y_end + EPS) {
                            min_y = std::min(min_y, y_f);
                            max_y = std::max(max_y, y_f);
                        }
                    }
                    
                    if (min_y < max_y) { // 有效区间
                        left_intervals.push_back({min_y, max_y, seg.count});
                    }
                }
                
                // 添加锁定区间
                locked_l[l].insert(locked_l[l].end(), 
                                    left_intervals.begin(), left_intervals.end());
                locked_r[best_match->r_idx].insert(locked_r[best_match->r_idx].end(),
                                    best_match->intervals.begin(), best_match->intervals.end());
                
                // 合并区间（考虑精度）
                mergeIntervals(locked_l[l], precision);
                mergeIntervals(locked_r[best_match->r_idx], precision);
                
                progress = true;
            }
        }
    }
    
    return final_matches;
}

/*************************************************************************************** */






/************************************** 同名点匹配 **************************************/
std::vector<cv::Point3f> LaserProcessor::generateCloudPoints(
    const std::vector<std::tuple<int, int, int>>& laser_match,
    const std::vector<LaserLine> laser_l,
    const std::vector<LaserLine> laser_r) {

    // 记录点云生成结果
    std::vector<cv::Point3f> cloud_points;

    // 标定参数
    const auto calib_info = ConfigManager::getInstance().getCalibInfo();
    const float cx1 = calib_info.P[0].at<double>(0, 2);
    const float cy1 = calib_info.P[0].at<double>(1, 2);
    const float cx2 = calib_info.P[1].at<double>(0, 2);
    
    for (const auto& [l_idx, p_idx, r_idx] : laser_match) {
        // 划分左右图像的点
        std::vector<cv::Point2f> left_points, right_points;
        for (const auto& p : laser_l[l_idx].points) {
            const auto it = laser_r[r_idx].points.find(p.second.y);
            if (it == laser_r[r_idx].points.end()) continue;

            // if (p.second.x >= 995 && p.second.x <= 1034 && p.second.y >= 560 && p.second.y <= 591)
            //     puts("");

            left_points.emplace_back(p.second.x, p.second.y);
            right_points.emplace_back(it->second.x, it->second.y);
        }

        // 三角测量
        cv::Mat points4D;
        cv::triangulatePoints(calib_info.P[0], calib_info.P[1],
                            left_points, right_points, points4D);
        
        // 转换为3D点
        for (int i = 0; i < points4D.cols; i++) {
            cv::Point3f point;
            float w = points4D.at<float>(3, i);
            if (std::abs(w) < 1e-6) {
                std::cerr << "警告: 点 " << i << " 重建可能不准确 (w≈0)" << std::endl;
                continue;
            }
            point.x = points4D.at<float>(0, i) / w;
            point.y = points4D.at<float>(1, i) / w;
            point.z = points4D.at<float>(2, i) / w;
            cloud_points.emplace_back(point);
        }
    }

    // std::ofstream ofs("cloudpoints.txt");
    // for (const auto& pt : cloud_points) {
    //     ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
    // }
    // std::cout << "点云已保存到 " << "cloudpoints.txt" << std::endl;

    return cloud_points;
}

// 基于 IntervalMatch 进行三维重建
std::vector<cv::Point3f> LaserProcessor::generateCloudPoints2(
    const std::vector<IntervalMatch>& matches,
    const std::vector<LaserLine>& laser_l,
    const std::vector<LaserLine>& laser_r)
{
    std::vector<cv::Point3f> cloud;
    const auto calib_info = ConfigManager::getInstance().getCalibInfo();

    // 遍历每条匹配区间
    for(const auto& m : matches) {
        std::vector<cv::Point2f> lp, rp;
        // 左右同一区间内点对
    for(const auto& iv : m.intervals) {
        for(float y=iv.y_start; y<=iv.y_end; y+=precision) {
            // 左侧
            auto itL = laser_l[m.l_idx].points.lower_bound(y);
            if (itL != laser_l[m.l_idx].points.begin()) {
                auto prevL = std::prev(itL);
                if (itL == laser_l[m.l_idx].points.end() || fabs(prevL->first - y) < fabs(itL->first - y)) {
                    itL = prevL;
                }
            }
            // 右侧
            auto itR = laser_r[m.r_idx].points.lower_bound(y);
            if (itR != laser_r[m.r_idx].points.begin()) {
                auto prevR = std::prev(itR);
                if (itR == laser_r[m.r_idx].points.end() || fabs(prevR->first - y) < fabs(itR->first - y)) {
                    itR = prevR;
                }
            }
            // 判断是否在精度范围内
            if(itL != laser_l[m.l_idx].points.end() && fabs(itL->first - y) < EPS &&
            itR != laser_r[m.r_idx].points.end() && fabs(itR->first - y) < EPS) {
                lp.emplace_back(itL->second.x, y);
                rp.emplace_back(itR->second.x, y);
            }
        }
    }
        if(lp.empty()) continue;
        cv::Mat pts4;
        cv::triangulatePoints(calib_info.P[0], calib_info.P[1], lp, rp, pts4);
        for(int i=0;i<pts4.cols;++i){
            float w = pts4.at<float>(3,i);
            if(fabs(w)<1e-6) continue;
            cloud.emplace_back(
                pts4.at<float>(0,i)/w,
                pts4.at<float>(1,i)/w,
                pts4.at<float>(2,i)/w
            );
        }
    }
    return cloud;
}

/************************************************************************************* */
