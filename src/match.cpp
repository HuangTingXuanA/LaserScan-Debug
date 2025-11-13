#include "match.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

#define DEBUG_REGION

std::vector<RegionMatch>
MatchProcessor::match(const std::vector<LaserLine> &laser_l,
                      const std::vector<LaserLine> &laser_r,
                      const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
    // 算法流程：
    // 1. 并行生成匹配候选点
    // 2. 按(左线,右线,光平面)分组
    // 3. 贪心算法分割连续区间
    // 4. 按x坐标排序并分配区域ID
    
    // =============== 阶段0：获取基本参数 ===============
    const int img_rows = rectify_l.rows;
    const int img_cols = rectify_l.cols;
    const int L = static_cast<int>(laser_l.size());
    const int R = static_cast<int>(laser_r.size());

    const auto calib = ConfigManager::getInstance().getCalibInfo();
    const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
    const int P = static_cast<int>(surfaces.size());

    // 相机内参
    double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
    double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
    double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
    double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
    double baseline = 1 / calib.Q.at<double>(3, 2);

    // =============== 阶段1：预处理数据结构 ===============
    
    // 预计算左线采样数据和归一化射线
    std::vector<std::vector<LeftPoint>> left_samples(L);
    for (int l = 0; l < L; ++l) {
        const auto &mp = laser_l[l].points;
        left_samples[l].reserve(mp.size());

        for (const auto &kv : mp) {
            float y_f = kv.first;
            float x_f = kv.second.x;

            // 计算归一化射线方向
            cv::Point3f ray(
                (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
                (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
            float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
            if (rn > 0.0f)
                ray *= (1.0f / rn);
            
            left_samples[l].emplace_back(y_f, x_f, ray);
        }
    }

    // 预处理右线数据为向量形式，便于二分查找
    std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
    for (int r = 0; r < R; ++r) {
        right_vec[r].reserve(laser_r[r].points.size());
        for (const auto &kv : laser_r[r].points) {
            right_vec[r].emplace_back(kv.first, kv.second);
        }
    }


    // =============== 阶段2：并行生成匹配候选 ===============

    const int total_tasks = L * P;
    tbb::concurrent_vector<MatchCandidate> candidates;

    tbb::parallel_for(
        tbb::blocked_range<int>(0, total_tasks),
        [&](const tbb::blocked_range<int> &range) {
            for (int task_idx = range.begin(); task_idx != range.end(); ++task_idx) {
                int l = task_idx / P;
                int p = task_idx % P;
                
                const auto &ls = left_samples[l];
                const cv::Mat &coef = surfaces[p].coefficients;
                if (ls.empty()) continue;

                // 遍历左线的每个点
                for (const auto &lp : ls) {
                    auto ips = findIntersection(cv::Point3f(0, 0, 0), lp.ray, coef);
                    if (ips.empty()) continue;

                    // 选择合理深度的交点
                    cv::Point3f pt3;
                    bool ok = false;
                    for (auto &q : ips) {
                        if (q.z > 100 && q.z < 1500) {
                            pt3 = q;
                            ok = true;
                            break;
                        }
                    }
                    if (!ok) continue;

                    // 投影到右图像
                    cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
                    float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
                    float yr = alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
                    if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows) continue;
                
                    // 在所有右线中查找匹配
                    for (int r = 0; r < R; ++r) {
                        const auto &right_points = right_vec[r];

                        auto it_r = std::lower_bound(
                            right_points.begin(), right_points.end(), yr,
                            [](const std::pair<float, LaserPoint> &a, float val) {
                                return a.first < val;
                            });
                        
                        if (it_r != right_points.end() && std::fabs(it_r->first - yr) < EPS_) {
                            float d = std::hypot(it_r->second.x - xr, it_r->first - yr);
                            if (d <= D_thresh_) {
                                candidates.emplace_back(l, r, p, yr, d);
                            }
                        }
                    }
                }
            }
        });

    // =============== 阶段3：按(左线,右线,光平面)分组 ===============
    
    std::map<MatchKey, std::vector<std::pair<float, float>>> grouped_matches;
    for (const auto &cand : candidates) {
        MatchKey key{cand.l_idx, cand.r_idx, cand.p_idx};
        grouped_matches[key].emplace_back(cand.y, cand.distance);
    }
    printf("(l,r,p)组合数 %zu\n", grouped_matches.size());

#ifdef DEBUG_REGION
    // =============== 可视化候选区间（覆盖前） ===============
    
    cv::Mat vis_candidates;
    cv::hconcat(rectify_l, rectify_r, vis_candidates);
    if (vis_candidates.channels() == 1)
        cv::cvtColor(vis_candidates, vis_candidates, cv::COLOR_GRAY2BGR);
    
    int offset_x = rectify_l.cols;
    
    // 定义30种颜色
    static const std::vector<cv::Scalar> colors = {
        {255, 0, 0},   {0, 255, 0},   {0, 0, 255},   {255, 255, 0},  {255, 0, 255},
        {0, 255, 255}, {128, 0, 0},   {0, 128, 0},   {0, 0, 128},    {128, 128, 0},
        {128, 0, 128}, {0, 128, 128}, {64, 0, 128},  {128, 64, 0},   {0, 128, 64},
        {64, 128, 0},  {0, 64, 128},  {128, 0, 64},  {192, 192, 0},  {192, 0, 192},
        {64, 255, 0},  {255, 64, 0},  {0, 64, 255},  {0, 255, 64},   {255, 0, 64},
        {64, 0, 255},  {192, 0, 64},  {64, 192, 0},  {0, 192, 64},   {64, 0, 192}
    };
    
    // 为每个(l,r,p)组合分配颜色并可视化
    int color_idx = 0;
    for (const auto &[key, match_points] : grouped_matches) {
        cv::Scalar color = colors[color_idx % colors.size()];
        color_idx++;
        
        // 提取y坐标集合
        std::set<float> y_coords;
        for (const auto &[y, dist] : match_points) {
            y_coords.insert(y);
        }
        
        if (y_coords.empty()) continue;
        
        // 绘制左线的候选区间点
        for (float y : y_coords) {
            if (laser_l[key.l_idx].points.find(y) != laser_l[key.l_idx].points.end()) {
                float x = laser_l[key.l_idx].points.at(y).x;
                cv::circle(vis_candidates, cv::Point(cvRound(x), cvRound(y)), 2, color, -1);
            }
        }
        
        // 绘制右线的候选区间点
        for (float y : y_coords) {
            if (laser_r[key.r_idx].points.find(y) != laser_r[key.r_idx].points.end()) {
                float x = laser_r[key.r_idx].points.at(y).x;
                cv::circle(vis_candidates, cv::Point(cvRound(x) + offset_x, cvRound(y)), 2, color, -1);
            }
        }
        
        // 在区间中心标注信息
        float y_mid = (*y_coords.begin() + *y_coords.rbegin()) / 2.0f;
        if (laser_l[key.l_idx].points.find(y_mid) != laser_l[key.l_idx].points.end() ||
            !y_coords.empty()) {
            // 找最接近中点的y坐标
            float closest_y = *y_coords.begin();
            float min_diff = std::abs(closest_y - y_mid);
            for (float y : y_coords) {
                float diff = std::abs(y - y_mid);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_y = y;
                }
            }
            
            if (laser_l[key.l_idx].points.find(closest_y) != laser_l[key.l_idx].points.end()) {
                float x = laser_l[key.l_idx].points.at(closest_y).x;
                char text[64];
                snprintf(text, sizeof(text), "L%d-R%d-P%d", key.l_idx, key.r_idx, key.p_idx);
                cv::putText(vis_candidates, text, 
                           cv::Point(cvRound(x) - 60, cvRound(closest_y) - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }
    }
    
    cv::namedWindow("Candidate Intervals (Before Coverage)", cv::WINDOW_NORMAL);
    cv::resizeWindow("Candidate Intervals (Before Coverage)", vis_candidates.cols, vis_candidates.rows);
    cv::imshow("Candidate Intervals (Before Coverage)", vis_candidates);
    while(true) { if (cv::waitKey(0) == ' ') break; }
    cv::destroyWindow("Candidate Intervals (Before Coverage)");
#endif

    // =============== 阶段4：连续性检测和贪心区间覆盖 ===============
    
    // 将不连续的坐标集分割成多个连续段
    auto splitConnectedSegments = [](const std::vector<float> &coords, float gap = 1.5f) 
        -> std::vector<std::vector<float>> {
        if (coords.empty()) return {};
        
        std::vector<std::vector<float>> segments;
        std::vector<float> current;
        current.push_back(coords[0]);
        
        for (size_t i = 1; i < coords.size(); ++i) {
            if (coords[i] - coords[i-1] <= gap) {
                current.push_back(coords[i]);
            } else {
                segments.push_back(current);
                current.clear();
                current.push_back(coords[i]);
            }
        }
        if (!current.empty()) segments.push_back(current);
        
        return segments;
    };
    
    // 贪心算法：不相交区间覆盖
    auto greedyCover = [&splitConnectedSegments](
        std::vector<IntervalSegment> intervals, 
        const std::vector<float> &target,
        int min_len) -> std::vector<IntervalSegment> {
        
        if (intervals.empty() || target.empty()) return {};
        
        // 按覆盖点数降序排序
        std::sort(intervals.begin(), intervals.end(),
            [&target](const IntervalSegment &a, const IntervalSegment &b) {
                int count_a = 0, count_b = 0;
                for (float y : a.y_coords) {
                    if (std::binary_search(target.begin(), target.end(), y)) count_a++;
                }
                for (float y : b.y_coords) {
                    if (std::binary_search(target.begin(), target.end(), y)) count_b++;
                }
                return count_a > count_b;
            });
        
        std::vector<IntervalSegment> result;
        std::vector<float> covered;
        std::vector<bool> used(intervals.size(), false);
        
        // 贪心选择
        for (size_t i = 0; i < intervals.size(); ++i) {
            if (used[i]) continue;
            
            // 提取未覆盖的点
            std::vector<float> new_coords;
            for (float y : intervals[i].y_coords) {
                if (!std::binary_search(covered.begin(), covered.end(), y) &&
                    std::binary_search(target.begin(), target.end(), y)) {
                    new_coords.push_back(y);
                }
            }
            
            if (new_coords.empty()) {
                used[i] = true;
                continue;
            }
            
            // 分割连续段
            auto segments = splitConnectedSegments(new_coords);
            
            for (const auto &seg : segments) {
                if (static_cast<int>(seg.size()) < min_len) continue;
                
                IntervalSegment new_seg;
                new_seg.line_idx = intervals[i].line_idx;
                new_seg.pair_idx = intervals[i].pair_idx;
                new_seg.p_idx = intervals[i].p_idx;
                new_seg.y_coords = seg;
                
                // 提取对应的匹配点
                for (const auto &[y, dist] : intervals[i].match_points) {
                    if (std::binary_search(seg.begin(), seg.end(), y)) {
                        new_seg.match_points.emplace_back(y, dist);
                    }
                }
                
                // 计算平均距离
                if (!new_seg.match_points.empty()) {
                    float sum = 0.0f;
                    for (const auto &[y, dist] : new_seg.match_points) sum += dist;
                    new_seg.avg_distance = sum / new_seg.match_points.size();
                }
                
                result.push_back(new_seg);
            }
            
            // 更新已覆盖集合
            covered.insert(covered.end(), new_coords.begin(), new_coords.end());
            std::sort(covered.begin(), covered.end());
            
            used[i] = true;
            if (covered.size() >= target.size()) break;
        }
        
        // 按y坐标排序
        std::sort(result.begin(), result.end(),
            [](const IntervalSegment &a, const IntervalSegment &b) {
                return a.y_start() < b.y_start();
            });
        
        return result;
    };

    // =============== 阶段5：为每条激光线应用贪心算法分割区间 ===============
    
    // 处理左线区间
    std::vector<LaserRegion> left_regions;
    for (int l = 0; l < L; ++l) {
        // 获取目标覆盖集合
        std::vector<float> target;
        for (const auto &[y, point] : laser_l[l].points) {
            target.push_back(alignToPrecision(y));
        }
        if (target.empty()) continue;
        std::sort(target.begin(), target.end());
        
        // 收集候选区间
        std::vector<IntervalSegment> intervals;
        for (const auto &[key, match_points] : grouped_matches) {
            if (key.l_idx != l) continue;
            
            IntervalSegment seg;
            seg.line_idx = l;
            seg.pair_idx = key.r_idx;
            seg.p_idx = key.p_idx;
            seg.match_points = match_points;
            
            for (const auto &[y, dist] : match_points) {
                seg.y_coords.push_back(y);
            }
            std::sort(seg.y_coords.begin(), seg.y_coords.end());
            
            float sum = 0.0f;
            for (const auto &[y, dist] : match_points) sum += dist;
            seg.avg_distance = sum / match_points.size();
            
            intervals.push_back(seg);
        }
        
        // 应用贪心算法
        auto selected = greedyCover(intervals, target, MIN_LEN_);
        
        // 转换为 LaserRegion
        for (const auto &seg : selected) {
            LaserRegion region;
            region.line_idx = seg.line_idx;
            region.pair_idx = seg.pair_idx;
            region.p_idx = seg.p_idx;
            region.y_coords = seg.y_coords;
            region.avg_distance = seg.avg_distance;
            
            // 构建实际像素坐标
            for (float y : seg.y_coords) {
                if (laser_l[l].points.find(y) != laser_l[l].points.end()) {
                    region.points.emplace_back(y, laser_l[l].points.at(y).x);
                }
            }
            
            // 计算中心x坐标
            if (!region.points.empty()) {
                size_t mid = region.points.size() / 2;
                region.center_x = region.points[mid].second;
            }
            
            left_regions.push_back(region);
        }
    }
    
    // 处理右线区间
    std::vector<LaserRegion> right_regions;
    for (int r = 0; r < R; ++r) {
        std::vector<float> target;
        for (const auto &[y, point] : laser_r[r].points) {
            target.push_back(alignToPrecision(y));
        }
        if (target.empty()) continue;
        std::sort(target.begin(), target.end());
        
        std::vector<IntervalSegment> intervals;
        for (const auto &[key, match_points] : grouped_matches) {
            if (key.r_idx != r) continue;
            
            IntervalSegment seg;
            seg.line_idx = r;
            seg.pair_idx = key.l_idx;  // 右线视角存左线索引
            seg.p_idx = key.p_idx;
            seg.match_points = match_points;
            
            for (const auto &[y, dist] : match_points) {
                seg.y_coords.push_back(y);
            }
            std::sort(seg.y_coords.begin(), seg.y_coords.end());
            
            float sum = 0.0f;
            for (const auto &[y, dist] : match_points) sum += dist;
            seg.avg_distance = sum / match_points.size();
            
            intervals.push_back(seg);
        }
        
        auto selected = greedyCover(intervals, target, MIN_LEN_);
        
        for (const auto &seg : selected) {
            LaserRegion region;
            region.line_idx = seg.line_idx;
            region.pair_idx = seg.pair_idx;
            region.p_idx = seg.p_idx;
            region.y_coords = seg.y_coords;
            region.avg_distance = seg.avg_distance;
            
            for (float y : seg.y_coords) {
                if (laser_r[r].points.find(y) != laser_r[r].points.end()) {
                    region.points.emplace_back(y, laser_r[r].points.at(y).x);
                }
            }
            
            if (!region.points.empty()) {
                size_t mid = region.points.size() / 2;
                region.center_x = region.points[mid].second;
            }
            
            right_regions.push_back(region);
        }
    }
    
    // =============== 阶段6：按x坐标排序并分配区域ID ===============
    
    std::sort(left_regions.begin(), left_regions.end(),
              [](const LaserRegion &a, const LaserRegion &b) {
                  return a.center_x < b.center_x;
              });
    
    for (size_t i = 0; i < left_regions.size(); ++i) {
        left_regions[i].region_id = static_cast<int>(i);
    }
    
    std::sort(right_regions.begin(), right_regions.end(),
              [](const LaserRegion &a, const LaserRegion &b) {
                  return a.center_x < b.center_x;
              });
    
    for (size_t i = 0; i < right_regions.size(); ++i) {
        right_regions[i].region_id = static_cast<int>(i);
    }

    printf("区间分割完成：左图区间数 %zu，右图区间数 %zu\n", 
           left_regions.size(), right_regions.size());

#ifdef DEBUG_REGION
    // =============== 可视化分割后的区间 ===============
    
    cv::Mat vis_merged;
    cv::hconcat(rectify_l, rectify_r, vis_merged);
    if (vis_merged.channels() == 1)
        cv::cvtColor(vis_merged, vis_merged, cv::COLOR_GRAY2BGR);
    
    // 可视化左图区间
    for (size_t i = 0; i < left_regions.size(); ++i) {
        const auto &region = left_regions[i];
        cv::Scalar color = colors[i % colors.size()];
        
        for (const auto &[y, x] : region.points) {
            cv::circle(vis_merged, cv::Point(cvRound(x), cvRound(y)), 2, color, -1);
        }
        
        if (!region.points.empty()) {
            auto mid_point = region.points[region.points.size() / 2];
            char text[64];
            snprintf(text, sizeof(text), "L%d-Reg%d", region.line_idx, region.region_id);
            cv::putText(vis_merged, text, 
                       cv::Point(cvRound(mid_point.second) - 50, cvRound(mid_point.first) - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }
    }
    
    // 可视化右图区间
    for (size_t i = 0; i < right_regions.size(); ++i) {
        const auto &region = right_regions[i];
        cv::Scalar color = colors[i % colors.size()];
        
        for (const auto &[y, x] : region.points) {
            cv::circle(vis_merged, cv::Point(cvRound(x) + offset_x, cvRound(y)), 2, color, -1);
        }
        
        if (!region.points.empty()) {
            auto mid_point = region.points[region.points.size() / 2];
            char text[64];
            snprintf(text, sizeof(text), "R%d-Reg%d", region.line_idx, region.region_id);
            cv::putText(vis_merged, text, 
                       cv::Point(cvRound(mid_point.second) + offset_x + 10, cvRound(mid_point.first) - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }
    }
    
    // 显示可视化结果
    cv::namedWindow("Merged Regions Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("Merged Regions Visualization", vis_merged.cols, vis_merged.rows);
    cv::imshow("Merged Regions Visualization", vis_merged);
    while(true) { if (cv::waitKey(0) == ' ') break; }
    cv::destroyWindow("Merged Regions Visualization");
#endif

    // =============== 阶段4：基于分割后的区间进行匈牙利算法匹配 ===============
    
    std::vector<RegionMatch> final_match;

#ifdef DEBUG_REGION
    // =============== 可视化最佳区间候选 ===============
    
    // 创建左右图像拼接的可视化底图
    cv::Mat vis_global;
    cv::hconcat(rectify_l, rectify_r, vis_global);
    if (vis_global.channels() == 1)
        cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
    
    int off = rectify_l.cols;  // 右图在拼接图中的x偏移量
    
    // 定义30种不同的颜色用于区分不同的匹配对
    static const std::vector<cv::Scalar> palette30 = {
        {255, 0, 0},   {0, 255, 0},   {0, 0, 255},   {255, 255, 0},  {255, 0, 255},
        {0, 255, 255}, {128, 0, 0},   {0, 128, 0},   {0, 0, 128},    {128, 128, 0},
        {128, 0, 128}, {0, 128, 128}, {64, 0, 128},  {128, 64, 0},   {0, 128, 64},
        {64, 128, 0},  {0, 64, 128},  {128, 0, 64},  {192, 192, 0},  {192, 0, 192},
        {64, 255, 0},  {255, 64, 0},  {0, 64, 255},  {0, 255, 64},   {255, 0, 64},
        {64, 0, 255},  {192, 0, 64},  {64, 192, 0},  {0, 192, 64},   {64, 0, 192}
    };
    
    // 用彩色显示最佳匹配的区间点
    for (size_t idx = 0; idx < final_match.size(); ++idx) {
        const auto &candidate = final_match[idx];
        cv::Scalar color = palette30[idx % palette30.size()];
        
        // 找到对应的左右区间
        const LaserRegion *left_region_ptr = nullptr;
        const LaserRegion *right_region_ptr = nullptr;
        
        for (const auto &region : left_regions) {
            if (region.region_id == candidate.l_region_id) {
                left_region_ptr = &region;
                break;
            }
        }
        
        for (const auto &region : right_regions) {
            if (region.region_id == candidate.r_region_id) {
                right_region_ptr = &region;
                break;
            }
        }
        
        if (!left_region_ptr || !right_region_ptr) continue;
        
        // 重新计算实际匹配的点
        const cv::Mat &coef = surfaces[candidate.p_idx].coefficients;
        const auto &ls = left_samples[candidate.l_idx];
        
        std::set<float> matched_y_coords;
        
        // 对左区间的每个点进行重投影
        for (const auto &[y_f, x_f] : left_region_ptr->points) {
            // 查找对应的射线
            auto it = std::find_if(ls.begin(), ls.end(),
                [y_f, this](const LeftPoint &lp) {
                    return std::fabs(lp.y - y_f) < EPS_;
                });
            if (it == ls.end()) continue;
            
            const cv::Point3f &ray = it->ray;
            
            // 与光平面求交
            auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
            if (ips.empty()) continue;
            
            cv::Point3f pt3;
            bool ok = false;
            for (auto &q : ips) {
                if (q.z > 100 && q.z < 1500) {
                    pt3 = q;
                    ok = true;
                    break;
                }
            }
            if (!ok) continue;
            
            // 投影到右图像
            cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
            float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
            float yr = alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
            if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows) continue;
            
            // 在右区间中查找匹配点
            auto it_r = std::lower_bound(
                right_region_ptr->points.begin(), right_region_ptr->points.end(), yr,
                [](const std::pair<float, float> &a, float val) {
                    return a.first < val;
                });
            
            if (it_r != right_region_ptr->points.end() && std::fabs(it_r->first - yr) < EPS_) {
                float d = std::hypot(it_r->second - xr, it_r->first - yr);
                if (d <= D_thresh_) {
                    matched_y_coords.insert(yr);
                }
            }
        }
        
        // 从原始激光线数据中获取实际匹配的点
        std::vector<std::pair<float, float>> matched_left_points;
        std::vector<std::pair<float, float>> matched_right_points;
        
        // 绘制左线实际匹配的点（彩色）
        const auto &left_line = laser_l[candidate.l_idx];
        for (const auto &[y, point] : left_line.points) {
            // 检查该 y 坐标是否在实际匹配的坐标集合中
            if (matched_y_coords.find(alignToPrecision(y)) != matched_y_coords.end()) {
                cv::circle(vis_global, cv::Point(cvRound(point.x), cvRound(y)), 2, color, -1);
                matched_left_points.emplace_back(y, point.x);
            }
        }
        
        // 绘制右线实际匹配的点（彩色）
        const auto &right_line = laser_r[candidate.r_idx];
        for (const auto &[y, point] : right_line.points) {
            // 检查该 y 坐标是否在实际匹配的坐标集合中
            if (matched_y_coords.find(alignToPrecision(y)) != matched_y_coords.end()) {
                cv::circle(vis_global, cv::Point(cvRound(point.x) + off, cvRound(y)), 2, color, -1);
                matched_right_points.emplace_back(y, point.x);
            }
        }
        
        // 在左区间中心标注线ID和得分
        if (!matched_left_points.empty()) {
            auto left_mid = matched_left_points[matched_left_points.size() / 2];
            char info_text[128];
            snprintf(info_text, sizeof(info_text), "L%d P%d S:%.1f", 
                    candidate.l_idx, candidate.p_idx, candidate.score_info.score);
            cv::putText(vis_global, info_text, 
                       cv::Point(cvRound(left_mid.second) - 40, cvRound(left_mid.first) - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2);
        }
        
        // 在右区间中心标注线ID
        if (!matched_right_points.empty()) {
            auto right_mid = matched_right_points[matched_right_points.size() / 2];
            char info_text[128];
            snprintf(info_text, sizeof(info_text), "R%d", candidate.r_idx);
            cv::putText(vis_global, info_text, 
                       cv::Point(cvRound(right_mid.second) + off + 10, cvRound(right_mid.first)),
                       cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 2);
        }
    }
    
    // 显示可视化结果
    cv::namedWindow("Region Match Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("Region Match Visualization", vis_global.cols, vis_global.rows);
    cv::imshow("Region Match Visualization", vis_global);
    while(true) { if (cv::waitKey(0) == ' ') break; }
    cv::destroyWindow("Region Match Visualization");
    
    // 可选：保存可视化结果到文件
    // cv::imwrite("debug_region_matches.jpg", vis_global);
    
#endif

    return final_match;
}

std::vector<cv::Point3f>
MatchProcessor::findIntersection(const cv::Point3f &point,
                                 const cv::Point3f &normal,
                                 const cv::Mat &Coeff6x1) {
  float a = Coeff6x1.at<float>(0);
  float b = Coeff6x1.at<float>(1);
  float c = Coeff6x1.at<float>(2);
  float d = Coeff6x1.at<float>(3);
  float e = Coeff6x1.at<float>(4);
  float f = Coeff6x1.at<float>(5);

  std::vector<cv::Point3f> intersections;

  // 计算二次方程的系数
  float A = a * normal.y * normal.y + b * normal.y * normal.z +
            c * normal.z * normal.z;

  float B = 2 * a * point.y * normal.y +
            b * (point.y * normal.z + point.z * normal.y) +
            2 * c * point.z * normal.z + d * normal.y + e * normal.z - normal.x;

  float C = a * point.y * point.y + b * point.y * point.z +
            c * point.z * point.z + d * point.y + e * point.z + f - point.x;

  // 解二次方程 A*t² + B*t + C = 0
  float discriminant = B * B - 4 * A * C;

  if (std::abs(A) < 1e-6) { // 线性情况
    if (std::abs(B) > 1e-6) {
      float t = -C / B;
      intersections.push_back(point + t * normal);
    }
  } else if (discriminant > 0) { // 两个实数解
    float sqrt_discriminant = std::sqrt(discriminant);
    float t1 = (-B + sqrt_discriminant) / (2 * A);
    float t2 = (-B - sqrt_discriminant) / (2 * A);
    intersections.push_back(point + t1 * normal);
    intersections.push_back(point + t2 * normal);
  } else if (std::abs(discriminant) < 1e-6) { // 一个实数解
    float t = -B / (2 * A);
    intersections.push_back(point + t * normal);
  }

  return intersections;
}

ScoreInfo MatchProcessor::computePointScore(
    const std::vector<std::pair<float, float>> &distance_pairs,
    int left_point_count, int right_point_count) {

  ScoreInfo score_res;

  if (distance_pairs.empty())
    return score_res;

  // 1. 确定较短线段作为基准（现在传入的参数已经是有效点数）
  int shorter_length = std::min(left_point_count, right_point_count);
  int longer_length = std::max(left_point_count, right_point_count);

  // 边界保护：避免极短线段
  const int MIN_SEGMENT_LENGTH = 10;
  if (shorter_length < MIN_SEGMENT_LENGTH) {
    return score_res;
  }

  int matched_count = static_cast<int>(distance_pairs.size());

  // 2. 核心指标：区间重叠比（基于较短线段）
  score_res.coverage =
      static_cast<float>(matched_count) / static_cast<float>(shorter_length);
  score_res.coverage = std::min(score_res.coverage, 1.0f); // 防止超过1.0

  // 3. 几何精度项：使用众数距离（反映主要匹配质量）
  std::vector<float> distances;
  distances.reserve(distance_pairs.size());
  float sum_distance = 0.0f;
  for (const auto &[y, d] : distance_pairs) {
    distances.push_back(d);
    sum_distance += d;
  }

  // 计算众数距离（几何精度）- 反映最常见的匹配质量
  if (!distances.empty()) {
    // 使用分箱方法计算众数
    const float bin_width = 0.15f; // 0.15像素的分箱宽度
    std::map<int, int> histogram;

    // 构建直方图
    for (float d : distances) {
      int bin = static_cast<int>(std::round(d / bin_width));
      histogram[bin]++;
    }

    // 找到频次最高的分箱
    int max_count = 0;
    int mode_bin = 0;
    for (const auto &[bin, count] : histogram) {
      if (count > max_count) {
        max_count = count;
        mode_bin = bin;
      }
    }

    // 计算众数值（分箱中心）
    score_res.dis_mode = mode_bin * bin_width;

    // 如果众数频次太低（<20%），回退到中位数
    if (max_count < static_cast<int>(distances.size() * 0.2)) {
      std::nth_element(distances.begin(),
                       distances.begin() + distances.size() / 2,
                       distances.end());
      score_res.dis_mode = distances[distances.size() / 2];
    }
  }

  // 同时计算平均距离用于标准差计算
  score_res.dis_mean = sum_distance / distances.size();

  // 4. 几何一致性项：基于平均距离计算标准差
  float variance = 0.0f;
  for (float d : distances) {
    float diff = d - score_res.dis_mean;
    variance += diff * diff;
  }
  variance /= distances.size();
  score_res.dis_stddev = std::sqrt(variance);

  // 5. 尾段惩罚项：长度差异的轻微惩罚
  float length_diff_ratio = static_cast<float>(longer_length - shorter_length) /
                            static_cast<float>(longer_length);
  score_res.remain_penalty = length_diff_ratio;

  // 6. 权重设计：重叠质量为主要项，几何指标为辅助项
  const float alpha = 0.3f; // 几何精度权重
  const float beta = 0.4f;  // 几何一致性权重
  const float gamma = 0.2f; // 重叠质量权重
  const float delta = 0.1f; // 尾段惩罚权重

  // 重叠质量项：overlap_ratio越高越好，转换为惩罚项
  float overlap_penalty = std::max(0.0f, 1.0f - score_res.coverage);

  // 重叠质量阈值惩罚：overlap_ratio < 0.3时给予额外惩罚
  float low_overlap_penalty =
      (score_res.coverage < 0.3f) ? (0.3f - score_res.coverage) * 10.0f : 0.0f;

  score_res.score = alpha * score_res.dis_mode +  // 几何精度项（使用众数距离）
                    beta * score_res.dis_stddev + // 几何一致性项
                    gamma * overlap_penalty +     // 重叠质量项（主要）
                    gamma * low_overlap_penalty + // 低重叠额外惩罚
                    delta * length_diff_ratio;         // 尾段惩罚项（辅助）

  return score_res;
}


