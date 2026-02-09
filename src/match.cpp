#include "match.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>

#define DEBUG_REGION

std::vector<MatchResult>
MatchProcessor::match(const std::vector<LaserLine> &laser_l,
                      const std::vector<LaserLine> &laser_r,
                      const cv::Mat &rectify_l, const cv::Mat &rectify_r) {

    // 0. 参数准备
    static const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
    const int P = static_cast<int>(surfaces.size());
    const auto& calib_info = ConfigManager::getInstance().getCalibInfo();
    double fx_r = calib_info.P[1].at<double>(0, 0);
    double baseline = std::abs(calib_info.P[1].at<double>(0, 3) / fx_r); 

    // 1. 双向生成条带 (保留原有逻辑)
    std::vector<Band> bands_l = generateBands(laser_l, calib_info.P[0], rectify_l.rows);
    std::vector<Band> bands_r = generateBands(laser_r, calib_info.P[1], rectify_r.rows);
    int num_bands = std::min(bands_l.size(), bands_r.size());

#ifdef DEBUG_REGION
    // =============== 可视化：条带分割结果 ===============
    cv::Mat vis_bands = rectify_r.clone();
    if (vis_bands.channels() == 1) cv::cvtColor(vis_bands, vis_bands, cv::COLOR_GRAY2BGR);
    
    for (const auto& band : bands_r) {
        cv::Scalar color = getBandColor(band.idx);
        
        // 绘制条带分割线
        int y_line = (band.idx + 1) * SLICE_HEIGHT_;
        if (y_line < vis_bands.rows) {
            cv::line(vis_bands, cv::Point(0, y_line), cv::Point(vis_bands.cols, y_line), cv::Scalar(100,100,100), 1);
        }

        // 绘制该条带内的切片
        for (const auto* slice : band.slices) {
            for (const auto& pt : slice->pts) {
                // 实心点更清晰
                vis_bands.at<cv::Vec3b>(cvRound(pt.y), cvRound(pt.x)) = cv::Vec3b(color[0], color[1], color[2]);
            }
            // 可选：绘制切片ID
            cv::putText(vis_bands, std::to_string(slice->id), slice->center_pt, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1);
        }
    }
    
    cv::namedWindow("Bands Segmentation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Bands Segmentation", vis_bands.cols, vis_bands.rows);
    cv::imshow("Bands Segmentation", vis_bands);
    cv::waitKey(1);
#endif

    std::vector<MatchResult> all_results;
    std::mutex res_mutex;

    // 2. 并行处理每一条带
    tbb::parallel_for(0, num_bands, [&](int b_idx) {
        const auto& band_l = bands_l[b_idx];
        const auto& band_r = bands_r[b_idx];
        int NL = band_l.slices.size();
        int NR = band_r.slices.size();
        
        // 如果任意一方没有切片，无法进行匹配
        if (NL == 0 || NR == 0) return;

        const float INF = 1e9f;
        const float COST_THRESHOLD = 8.5f;  // 单切片有效匹配的阈值
        const float COST_SKIP = 10.0f;       // 跳过一个左切片(视为鬼影)的代价

        // DP 状态定义: dp[i][j][k]
        // i: 当前左切片索引 (0..NL-1)
        // j: 当前右切片索引 (0..NR-1)
        // k: 当前匹配光平面 k (0..P-1)
        std::vector<std::vector<std::vector<DpNode>>> dp(
            NL, std::vector<std::vector<DpNode>>(NR, std::vector<DpNode>(P, {INF, -1, -1, -1}))
        );

        // === 初始化 (处理第0个左切片) ===
        // 允许 L[0] 匹配 R[0...NR-1] 中的任意一个
        // 隐含条件：R[0] 前面的右切片被跳过(不计代价)，L[0] 前面的左切片不存在
        for (int j = 0; j < NR; ++j) {
            for (int k = 0; k < P; ++k) {
                auto info = computeCost(band_l.slices[0], band_r.slices[j], surfaces[k].coefficients, rectify_l, rectify_r);
                if (info.score < COST_THRESHOLD) {
                    dp[0][j][k] = {info.score, -1, -1, -1};
                }
            }
        }

        // === 状态转移 ===
        for (int i = 1; i < NL; ++i) { // 遍历左切片
            for (int j = 0; j < NR; ++j) { // 遍历右切片
                for (int k = 0; k < P; ++k) { // 遍历光平面
                    
                    // 1. 计算当前匹配代价
                    auto info = computeCost(band_l.slices[i], band_r.slices[j], surfaces[k].coefficients, rectify_l, rectify_r);
                    if (info.score >= COST_THRESHOLD) continue; // 剪枝

                    float min_prev_cost = INF;
                    int best_pi = -1, best_pj = -1, best_pk = -1;

                    // -------------------------------------------------
                    // 策略 A: 连接到前驱 (Link to Previous)
                    // -------------------------------------------------
                    
                    // Lookback L: 允许回溯左切片 (跳过中间的噪声)
                    int lookback_l = std::max(0, i - 5); 
                    
                    for (int pi = i - 1; pi >= lookback_l; --pi) {
                        float l_skip_penalty = (i - 1 - pi) * COST_SKIP;
                        
                        // Lookback R: 严格单调递增 (pj < j)
                        // 优化：右切片一般不会跳跃太远，取前10个
                        int lookback_r = std::max(0, j - 10);

                        for (int pj = j - 1; pj >= lookback_r; --pj) {
                            
                            // Check 4 (Geometry): 几何比例约束 (带内弹性约束)
                            float dist_L = band_l.slices[i]->center_pt.x - band_l.slices[pi]->center_pt.x;
                            float dist_R = band_r.slices[j]->center_pt.x - band_r.slices[pj]->center_pt.x;
                            
                            // 物理合理性：X必须递增
                            if (dist_L < 0.1f || dist_R < 0.1f) continue; 

                            // 比例约束：防止将远距离的左切片匹配到近距离的右切片
                            // float ratio = dist_R / dist_L;
                            // if (ratio < 0.2f || ratio > 3.5f) continue; 

                            // Lookback Plane: pk < k (严格单调，因为同一条带内不允许重复匹配平面)
                            for (int pk = 0; pk < k; ++pk) {
                                if (dp[pi][pj][pk].cost >= INF) continue;

                                int p_diff = k - pk; // >= 1
                                int l_idx_diff = i - pi;
                                int r_idx_diff = j - pj;

                                // Check 1 (Plane Monotonicity): 严格单调
                                // 循环条件 pk < k 已经保证了 p_diff >= 1

                                // Check 3 (Left Index Gap): 左图索引跨度约束
                                // 允许跳过的平面数 <= 物理跳过的切片数 + 1
                                if (p_diff > l_idx_diff + 1) continue;

                                // Check New (Right Index Gap): 右图索引跨度约束
                                // 右图同理，物理分布也限制了平面的跳跃
                                if (p_diff > r_idx_diff + 1) continue;

                                float total = dp[pi][pj][pk].cost + l_skip_penalty + info.score;
                                
                                if (total < min_prev_cost) {
                                    min_prev_cost = total;
                                    best_pi = pi; best_pj = pj; best_pk = pk;
                                }
                            }
                        }
                    }

                    // -------------------------------------------------
                    // 策略 B: 作为新起点 (Start Fresh)
                    // -------------------------------------------------
                    // 假设 0...i-1 的左切片全是噪声，当前匹配为第一对有效匹配
                    // 限制 j 的范围，防止从右图很靠后的位置突然开始
                    if (j < 6) { 
                        float start_cost = i * COST_SKIP + info.score;
                        if (start_cost < min_prev_cost) {
                            min_prev_cost = start_cost;
                            best_pi = -1; best_pj = -1; best_pk = -1;
                        }
                    }

                    // 更新状态
                    if (min_prev_cost < INF) {
                        dp[i][j][k] = {min_prev_cost, best_pi, best_pj, best_pk};
                    }
                }
            }
        }

        // === 回溯 (Backtracking) ===
        float min_end_cost = INF;
        int ei = -1, ej = -1, ek = -1;

        // 寻找最优终点 (允许最后几个左切片被跳过)
        for (int i = 0; i < NL; ++i) {
            float tail_penalty = (NL - 1 - i) * COST_SKIP;
            for (int j = 0; j < NR; ++j) {
                for (int k = 0; k < P; ++k) {
                    if (dp[i][j][k].cost < INF) {
                        float final_c = dp[i][j][k].cost + tail_penalty;
                        if (final_c < min_end_cost) {
                            min_end_cost = final_c;
                            ei = i; ej = j; ek = k;
                        }
                    }
                }
            }
        }

        // 路径重建
        std::vector<MatchResult> local_res;
        while (ei != -1) {
            // 需要重新计算/获取 info (实际工程中可在DP Node里存info或另开Cost表)
            auto info = computeCost(band_l.slices[ei], band_r.slices[ej], surfaces[ek].coefficients, rectify_l, rectify_r);
            
            MatchResult res;
            res.band_id = band_l.idx;
            res.l_slice_id = band_l.slices[ei]->id;
            res.r_slice_id = band_r.slices[ej]->id;
            res.l_idx = band_l.slices[ei]->laser_idx;
            res.r_idx = band_r.slices[ej]->laser_idx;
            res.p_idx = ek;
            res.info = info;
            local_res.push_back(res);

            auto& node = dp[ei][ej][ek];
            ei = node.prev_l;
            ej = node.prev_r;
            ek = node.prev_p;
        }

        if (!local_res.empty()) {
            std::lock_guard<std::mutex> lock(res_mutex);
            all_results.insert(all_results.end(), local_res.begin(), local_res.end());
        }
    });

    // filterMatches(all_results, bands_l, bands_r);

#ifdef DEBUG_REGION
    int w = rectify_l.cols; 
    int h = rectify_l.rows;
    cv::Mat canvas(h, w * 2, CV_8UC3);
    
    cv::Mat color_l, color_r;
    cv::cvtColor(rectify_l, color_l, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectify_r, color_r, cv::COLOR_GRAY2BGR);

    color_l.copyTo(canvas(cv::Rect(0,0,w,h)));
    color_r.copyTo(canvas(cv::Rect(w,0,w,h)));

    // 索引映射
    std::map<int, const MatchResult*> l_map;
    std::map<int, const MatchResult*> r_map;
    for(const auto& res : all_results) {
        l_map[res.l_slice_id] = &res;
        r_map[res.r_slice_id] = &res;
        printf("slice L_ID=%d-R_ID=%d-P_ID=%d | Score=%.2f, dis_mean=%.2f, dis_mode=%.2f, dis_stddev=%.2f, norm_census=%.2f\n",
               res.l_slice_id, res.r_slice_id, res.p_idx, res.info.score, res.info.dis_mean, res.info.dis_mode, res.info.dis_stddev, res.info.norm_census);
    }

    // 绘制左图
    for(const auto& band : bands_l) {
        cv::Scalar color = getBandColor(band.idx);
        for(const auto* s : band.slices) {
            // 只绘制匹配成功的，或者全部绘制但未匹配的暗一点
            bool matched = l_map.count(s->id);
            if (!matched) continue; 
            
            for(const auto& p : s->pts) cv::circle(canvas, cv::Point(p.x, p.y), 1, color, -1);
            
            // 显示 L_ID
            cv::putText(canvas, std::to_string(s->laser_idx) + "-" + std::to_string(s->id),
                        cv::Point2f(s->center_pt.x+13, s->center_pt.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 1
                    );
        }
    }

    // 绘制右图
    for(const auto& band : bands_r) {
        cv::Scalar color = getBandColor(band.idx);
        for(const auto* s : band.slices) {
            if (r_map.count(s->id)) {
                const auto* res = r_map[s->id];
                // 绘制点 (偏移 w)
                for(const auto& p : s->pts) cv::circle(canvas, cv::Point(p.x + w, p.y), 1, color, -1);
                
                // 显示 R_ID (L_ID)
                std::string txt = std::to_string(res->r_idx) + "-" + std::to_string(res->p_idx) + "-" + std::to_string(res->l_slice_id);
                cv::putText(canvas, txt, cv::Point(s->center_pt.x + w + 13, s->center_pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 1);
            }
        }
    }
    
    // 画分割线
    for(int y=0; y<h; y+=SLICE_HEIGHT_) cv::line(canvas, cv::Point(0,y), cv::Point(w*2,y), cv::Scalar(50,50,50));
    cv::namedWindow("Dual Slice Matching", cv::WINDOW_NORMAL);
    cv::resizeWindow("Dual Slice Matching", rectify_l.cols, rectify_l.rows);
    cv::imshow("Dual Slice Matching", canvas);
    cv::waitKey(0);
#endif


    return all_results;
}

void MatchProcessor::filterMatches(
    std::vector<MatchResult>& matches,
    const std::vector<Band>& bands_l,
    const std::vector<Band>& bands_r) 
{
    if (matches.empty()) return;

    // =========================================================
    // Pass 1: 基于左连通域 (Left Laser) 修复光平面跳变
    // 覆盖 ABA (中间) 和 AAAB (尾部) 模式
    // =========================================================
    
    // 1. 分组：按左线 ID (l_idx)
    std::map<int, std::vector<MatchResult*>> l_groups;
    for (auto& m : matches) {
        if (m.l_idx != -1) l_groups[m.l_idx].push_back(&m);
    }

    for (auto& [l_id, group] : l_groups) {
        if (group.empty()) continue;

        // 2. 排序：确保按条带空间顺序 (Band ID)
        std::sort(group.begin(), group.end(), [](MatchResult* a, MatchResult* b) {
            return a->band_id < b->band_id;
        });

        // 3. RLE 分析光平面 ID (p_idx)
        std::vector<Run> runs;
        int curr_p = group[0]->p_idx;
        int curr_start = 0;
        int curr_len = 1;

        for (size_t i = 1; i < group.size(); ++i) {
            if (group[i]->p_idx == curr_p) {
                curr_len++;
            } else {
                runs.push_back({curr_p, curr_start, curr_len});
                curr_p = group[i]->p_idx;
                curr_start = static_cast<int>(i);
                curr_len = 1;
            }
        }
        runs.push_back({curr_p, curr_start, curr_len});

        // 4. 检测并修复异常模式
        for (size_t i = 0; i < runs.size(); ++i) {
            bool need_fix = false;
            int target_run_idx = -1; // 指向参考段 (A) 的 run 索引

            // 必须有前驱节点才能修复 (A 在前)
            if (i > 0) {
                Run& prev = runs[i-1];
                Run& curr = runs[i];

                // 情况 A: ABA 模式 (中间段)
                // 条件: 前后光平面一致 且 当前长度 < 2 (即只有1个点)
                if (i < runs.size() - 1) {
                    Run& next = runs[i+1];
                    if (prev.p_idx == next.p_idx && curr.len < 2) {
                        need_fix = true;
                        target_run_idx = static_cast<int>(i - 1); // 参考前一段 A
                    }
                }

                // 情况 B: AAAB 模式 (尾段)
                // 条件: 是最后一个段 且 当前长度 < 3 (长度1或2)
                if (!need_fix && i == runs.size() - 1) {
                    if (curr.len < 3) {
                        need_fix = true;
                        target_run_idx = static_cast<int>(i - 1); // 参考前一段 A
                    }
                }
            }

            // 执行修复逻辑
            if (need_fix && target_run_idx != -1) {
                Run& target_run = runs[target_run_idx]; // 这是参考段 A
                Run& curr_run = runs[i];                // 这是待修复段 B

                // === 核心修改：计算参考段 A 中 r_idx 的众数 ===
                std::unordered_map<int, int> r_idx_counts;
                for (int k = target_run.start_idx; k < target_run.start_idx + target_run.len; ++k) {
                    int rid = group[k]->r_idx;
                    if (rid != -1) {
                        r_idx_counts[rid]++;
                    }
                }

                int target_r_idx = -1;
                int max_freq = -1;
                for (const auto& [rid, count] : r_idx_counts) {
                    if (count > max_freq) {
                        max_freq = count;
                        target_r_idx = rid;
                    }
                }

                // 如果参考段有效 (找到了众数)，则进行修复
                if (target_r_idx != -1) {
                    int target_p_idx = target_run.p_idx; // 目标光平面即 A 的光平面

                    for (int k = curr_run.start_idx; k < curr_run.start_idx + curr_run.len; ++k) {
                        MatchResult* m = group[k];
                        
                        // 利用 条带+laser 确定唯一的切片
                        const Band& band = bands_r[m->band_id];
                        auto it = band.laser2slice.find(target_r_idx);

                        if (it != band.laser2slice.end()) {
                            // 找到目标切片，执行修正
                            m->r_idx = target_r_idx;
                            m->r_slice_id = it->second->id; // 获取 Slice指针 后取 id
                            m->p_idx = target_p_idx;        // 修正为 A 的光平面
                        } else {
                            // 目标右线在该条带无切片（物理断裂），标记无效
                            m->r_idx = -1; 
                            m->l_idx = -1; 
                        }
                    }
                }
            }
        }
    }

    // =========================================================
    // Pass 3 (Cleanup): 物理移除无效匹配
    // =========================================================
    matches.erase(
        std::remove_if(matches.begin(), matches.end(), 
            [](const MatchResult& m) { return m.r_idx == -1; }),
        matches.end());
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

int MatchProcessor::computeCensus(const cv::Mat& img1, const cv::Point2f& pt1,
                         const cv::Mat& img2, const cv::Point2f& pt2, int window_size) {
    int cx1 = cvRound(pt1.x), cy1 = cvRound(pt1.y);
    int cx2 = cvRound(pt2.x), cy2 = cvRound(pt2.y);
    
    // 计算半窗口大小
    int half_window = window_size / 2;
    
    // 边界检查
    if (cx1 < half_window || cx1 >= img1.cols - half_window || 
        cy1 < half_window || cy1 >= img1.rows - half_window ||
        cx2 < half_window || cx2 >= img2.cols - half_window || 
        cy2 < half_window || cy2 >= img2.rows - half_window)
        return window_size * window_size; // 边界惩罚，最大可能代价

    int cost = 0;
    const uchar* ptr1 = img1.ptr<uchar>(cy1);
    const uchar* ptr2 = img2.ptr<uchar>(cy2);
    
    // 中心像素值
    int center1 = ptr1[cx1];
    int center2 = ptr2[cx2];
    
    // 遍历窗口内的所有像素
    for(int dy = -half_window; dy <= half_window; ++dy) {
        for(int dx = -half_window; dx <= half_window; ++dx) {
            bool bit1 = img1.at<uchar>(cy1 + dy, cx1 + dx) > center1;
            bool bit2 = img2.at<uchar>(cy2 + dy, cx2 + dx) > center2;
            if (bit1 != bit2) cost += 1;
        }
    }
    return cost;
}

std::vector<Band> MatchProcessor::generateBands(const std::vector<LaserLine>& laser, const cv::Mat& P, int img_rows) {
    // 初始化条带容器
    int num_bands = std::ceil((float)img_rows / SLICE_HEIGHT_);
    std::vector<Band> bands(num_bands);
    for (int i = 0; i < num_bands; ++i) bands[i].idx = i;

    // 获取相机内参 
    float fx = static_cast<float>(P.at<double>(0, 0));
    float fy = static_cast<float>(P.at<double>(1, 1));
    float cx = static_cast<float>(P.at<double>(0, 2));
    float cy = static_cast<float>(P.at<double>(1, 2));

    // 切片点缓存
    std::vector<LeftPoint> current_pts;
    current_pts.reserve(SLICE_HEIGHT_ / precision_); 

    for (int l_idx = 0; l_idx < static_cast<int>(laser.size()); ++l_idx) {
        const auto& line = laser[l_idx];
        if (line.points.empty()) continue;

        int current_band_idx = -1;
        
        for (const auto& kv : line.points) {
            float y = kv.first;
            float x = kv.second.x;
            
            // 计算当前点属于哪个 Band
            int band_idx = static_cast<int>(y / SLICE_HEIGHT_);
            
            // 边界检查
            if (band_idx < 0 || band_idx >= num_bands) continue;

            // 状态切换检测
            if (band_idx != current_band_idx) {
                // 如果之前的 buffer 不为空，且点数达标，则生成一个 Slice
                if (current_band_idx != -1 && current_pts.size() >= 15) {
                    auto s = std::make_unique<Slice>();
                    s->laser_idx = l_idx;
                    s->band_idx = current_band_idx;
                    
                    // 计算重心
                    int mid = current_pts.size() / 2;
                    s->center_pt = cv::Point2f(current_pts[mid].x, current_pts[mid].y);
                    
                    s->pts = std::move(current_pts);
                    
                    // 存入 Band
                    bands[current_band_idx].addSlice(std::move(s));
                }
                
                // 重置状态
                current_band_idx = band_idx;
                current_pts.clear();
            }

            // 计算射线并加入 buffer
            // 归一化射线计算 (x-cx)/fx, (y-cy)/fy, 1.0
            float rx = (x - cx) / fx;
            float ry = (y - cy) / fy;
            float rz = 1.0f;
            float norm = std::sqrt(rx*rx + ry*ry + rz*rz);
            
            current_pts.emplace_back(x, y, cv::Point3f(rx/norm, ry/norm, rz/norm));
        }

        // 处理该连通域最后一个切片 (收尾)
        if (current_band_idx != -1 && current_pts.size() >= 15) {
            auto s = std::make_unique<Slice>();
            s->laser_idx = l_idx;
            s->band_idx = current_band_idx;
            
            float sum_x = 0, sum_y = 0;
            for (const auto& p : current_pts) { sum_x += p.x; sum_y += p.y; }
            s->center_pt = cv::Point2f(sum_x / current_pts.size(), sum_y / current_pts.size());
            
            s->pts = std::move(current_pts);
            bands[current_band_idx].addSlice(std::move(s));
        }
        
        current_pts.clear(); // 准备处理下一个连通域
    }

    // 带内排序 (Intra-Band Sorting)
    // 利用 TBB 并行排序，提升效率
    tbb::parallel_for(0, num_bands, [&](int i) {
        if (!bands[i].slices.empty()) {
            // 对裸指针进行排序，速度极快
            std::sort(bands[i].slices.begin(), bands[i].slices.end(), 
                [](const Slice* a, const Slice* b) {
                    return a->center_pt.x < b->center_pt.x;
                });
        }
    });

    // 全局 ID 分配
    // 此时顺序已定：Band 0 (x小->大) -> Band 1 (x小->大) ...
    int global_id = 0;
    for (int i = 0; i < num_bands; ++i) {
        for (auto* s : bands[i].slices) {
            s->id = global_id++;
        }
    }

    // 构建查找表
    for(auto& b : bands) b.buildLaserToSlice();

    return bands;
}

ScoreInfo MatchProcessor::computeCost(const Slice* slice_l, const Slice* slice_r, const cv::Mat& coef,
                                const cv::Mat& rectify_l, const cv::Mat& rectify_r) {
    // 初始化为无效代价
    ScoreInfo info = {COST_INVALID_, 0, 0, 0, 0};

    // 相机内参
    const auto calib = ConfigManager::getInstance().getCalibInfo();
    double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
    double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
    double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
    double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
    double baseline = 1 / calib.Q.at<double>(3, 2);

    // 快速剪枝
    auto ips_center = findIntersection(
        cv::Point3f(0, 0, 0), 
        cv::Point3f((slice_l->center_pt.x - cx_l) / fx_l, (slice_l->center_pt.y - cy_l) / fy_l, 1.0f), 
        coef);
    if (ips_center.empty()) return info;
    cv::Point3f pt3_center;
    bool valid_center = false;
    for (auto &q : ips_center) {
        if (q.z > 100 && q.z < 600) {
            pt3_center = q;
            valid_center = true;
            break;
        }
    }
    if (!valid_center) return info;

    // 中心点投影到右图
    cv::Point3f proj(pt3_center.x - static_cast<float>(baseline), pt3_center.y, pt3_center.z);
    float proj_x = static_cast<float>(fx_r * proj.x / proj.z + cx_r);
    if (std::abs(proj_x - slice_r->center_pt.x) > 8.0f) return info;

    // 逐点匹配
    struct MatchPair {
        float dist;          // 几何误差
        cv::Point2f pt_l;    // 左图原始坐标 (用于Census)
        cv::Point2f pt_r;    // 右图匹配到的坐标 (用于Census)
    };
    std::vector<MatchPair> valid_pts;
    valid_pts.reserve(slice_l->pts.size());
    for (const auto& lp : slice_l->pts) {
        // 射线求交
        auto ips = findIntersection(cv::Point3f(0, 0, 0), lp.ray, coef);
        if (ips.empty()) continue;

        // 筛选深度范围内的交点
        cv::Point3f pt3;
        bool ok = false;
        for (auto &q : ips) {
            if (q.z > 100 && q.z < 600) {
                pt3 = q;
                ok = true;
                break;
            }
        }
        if (!ok) continue;

        // 投影到右图
        cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
        float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
        float yr = lp.y;
        if (xr < 0 || xr >= rectify_l.cols) continue;

        // 在 r_slice 中寻找匹配点
        float min_dist = 1e9f;
        const LeftPoint* best_rp = nullptr;
        for (const auto& rp : slice_r->pts) {
            if (std::abs(rp.y - yr) < EPS_) {
                float dist = std::abs(rp.x - xr);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_rp = &rp;
                }
            }
        }

        if (min_dist <= D_thresh_ && best_rp != nullptr) {
            valid_pts.push_back({min_dist, cv::Point2f(lp.x, lp.y), cv::Point2f(best_rp->x, best_rp->y)});
        }
    }

    // 有效性检查
    if (valid_pts.size() < slice_l->pts.size() * 0.7) return info;
   
    // 计算统计量
    float sum = 0;
    for(auto& p : valid_pts) sum += p.dist;
    info.dis_mean = sum / valid_pts.size();

    // 计算标准差
    float sq_sum = 0;
    for(auto& p : valid_pts) sq_sum += (p.dist - info.dis_mean)*(p.dist - info.dis_mean);
    info.dis_stddev = std::sqrt(sq_sum / valid_pts.size());

    // 计算众数
    const float bin_width = 0.15f; // 0.3像素的分箱宽度
    std::map<int, int> hist;
    for (const auto& mp : valid_pts) hist[std::round(mp.dist / bin_width)]++;
    int max_bin_cnt = 0;
    int mode_bin = 0;
    for (auto& kv : hist) {
        if (kv.second > max_bin_cnt) {
            max_bin_cnt = kv.second;
            mode_bin = kv.first;
        }
    }
    info.dis_mode = mode_bin * bin_width;

    // 计算Census代价
    std::vector<size_t> indices = {valid_pts.size()/4, valid_pts.size()/2, (3*valid_pts.size())/4};
    int total_census = 0;
    for (size_t idx : indices) {
        const auto& pair = valid_pts[idx];
        total_census += computeCensus(rectify_l, pair.pt_l, rectify_r, pair.pt_r);
    }
    info.norm_census = (float)total_census / indices.size();

    // 最终代价聚合
    float w_mode = 0.75f;   // 几何精度（众数距离）
    float w_std = 1.2f;    // 几何一致性（标准差）
    float w_cen = 1.0f;    // 纹理代价

    info.score = w_mode * info.dis_mode + 
                 w_std * info.dis_stddev + 
                 w_cen * info.norm_census;
    
    return info;
}

cv::Scalar MatchProcessor::getBandColor(int band_idx) {
    // 使用黄金分割生成区别明显的颜色
    const float golden_ratio_conjugate = 0.618033988749895;
    float h = std::fmod((band_idx * golden_ratio_conjugate), 1.0f);
    
    // HSV to RGB conversion (simplified)
    // 这里简单用固定表，实际可用完整HSV转换
    static const std::vector<cv::Scalar> palette = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, 
        {255, 0, 255}, {0, 255, 255}, {128, 0, 0}, {0, 128, 0}, 
        {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128}
    };
    return palette[band_idx % palette.size()];
}
