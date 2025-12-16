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

    static const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
    const int P = static_cast<int>(surfaces.size());
    const auto& calib_info = ConfigManager::getInstance().getCalibInfo();
    double baseline = 1 / calib_info.Q.at<double>(3, 2);
    double fx_r = calib_info.P[1].at<double>(0, 0), fy_r = calib_info.P[1].at<double>(1, 1);
    double cx_r = calib_info.P[1].at<double>(0, 2), cy_r = calib_info.P[1].at<double>(1, 2);

    // 1. 生成条带
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

    // 2. 并行处理每一条带
    std::vector<MatchResult> all_results;
    std::mutex res_mutex;
    tbb::parallel_for(0, num_bands, [&](int b_idx) {
        const auto& band_l = bands_l[b_idx];
        const auto& band_r = bands_r[b_idx];
        int NL = band_l.slices.size();
        int NR = band_r.slices.size();
        if (NL == 0 || NR == 0) return;

        const float INF = 1e9f;
        const float COST_THRESHOLD = 12.0f;  // 单切片有效匹配的阈值 (ScoreInfo.score)
        const float COST_SKIP = 8.5f;      // 跳过一个切片(视为鬼影)的代价

        // dp[i][j][k]
        // i: 当前考虑到左切片 i (0..NL-1)
        // j: 当前匹配到右切片 j (0..NR-1)
        // k: 当前匹配光平面 k (0..P-1)
        // 含义：L[i] 匹配 R[j] via P[k] 的最小代价
        // 注意：这意味着 L[i] 必须匹配 R[j]。如果 L[i] 被跳过，则不在这个状态里，而是通过转移处理。
        std::vector<std::vector<std::vector<DpNode>>> dp(
            NL, std::vector<std::vector<DpNode>>(NR, std::vector<DpNode>(P, {INF, -1, -1, -1}))
        );
        for (int j = 0; j < NR; ++j) {
            for (int k = 0; k < P; ++k) {
                auto info = computeCost(band_l.slices[0], band_r.slices[j], surfaces[k].coefficients, rectify_l, rectify_r);
                if (info.score < COST_THRESHOLD) {
                    dp[0][j][k] = {info.score, -1, -1, -1};
                }
            }
        }

        // === 状态转移 ===
        for (int i = 1; i < NL; ++i) {
            for (int j = 0; j < NR; ++j) {
                for (int k = 0; k < P; ++k) {
                    
                    // 1. 计算当前匹配代价 (L[i] <-> R[j] via P[k])
                    // 这一步比较耗时，可以考虑加简单的几何预筛选
                    auto info = computeCost(band_l.slices[i], band_r.slices[j], surfaces[k].coefficients, rectify_l, rectify_r);
                    if (info.score >= COST_THRESHOLD) continue;

                    float min_prev = INF;
                    int best_l = -1, best_r = -1, best_p = -1;

                    // 2. 寻找前驱
                    // Lookback L: 允许跳过左切片 (视为噪声)
                    int lookback_l = std::max(0, i - 5);
                    
                    for (int pi = i - 1; pi >= lookback_l; --pi) {
                        float l_skip_penalty = (i - 1 - pi) * COST_SKIP;
                        
                        // Delta L: 左边跨越了几个切片
                        int delta_l = i - pi; // 最小为1 (相邻)
                        
                        // 【核心修改】应用右图切片跨度约束
                        // 规则：总会预留 1 个位置 -> R_diff <= L_diff + 1
                        // 比如 L 相邻 (diff=1)，R diff <= 2 (pj 可以是 j-1, j-2)
                        int max_delta_r = delta_l + 1 ;
                        
                        // 计算 pj 的起始位置 (不能小于 0)
                        int min_pj = std::max(0, j - max_delta_r);
                        
                        // Lookback R: 严格单调 pj < j
                        for (int pj = min_pj; pj < j; ++pj) { 
                            
                            // 检查光平面约束: pk <= k
                            // 优化：同样可以限制 pk 的范围，比如 pk >= k-1
                            for (int pk = 0; pk <= k; ++pk) {
                                if (dp[pi][pj][pk].cost >= INF) continue;

                                float total = dp[pi][pj][pk].cost + l_skip_penalty + info.score;
                                
                                // (可选) 加上右图跳过惩罚
                                // int delta_r = j - pj;
                                // total += (delta_r - 1) * 5.0f;

                                if (total < min_prev) {
                                    min_prev = total;
                                    best_l = pi; best_r = pj; best_p = pk;
                                }
                            }
                        }
                    }

                    // 3. 作为新起点 (Start Fresh)
                    // 只有当 j 比较小的时候才允许作为起点，防止在很后面突然断开重连
                    // 或者加上右图跳过的代价
                    if (j < 5) { 
                        float start_cost = i * COST_SKIP + info.score; // + j * R_SKIP
                        if (start_cost < min_prev) {
                            min_prev = start_cost;
                            best_l = -1; best_r = -1; best_p = -1;
                        }
                    }

                    if (min_prev < INF) {
                        dp[i][j][k] = {min_prev, best_l, best_r, best_p};
                    }
                }
            }
        }

        // === 回溯 ===
        float min_end = INF;
        int ei = -1, ej = -1, ek = -1;

        // 寻找最优终点 (考虑最后几个左切片被跳过)
        for (int i = 0; i < NL; ++i) {
            float tail_penalty = (NL - 1 - i) * COST_SKIP;
            for (int j = 0; j < NR; ++j) {
                for (int k = 0; k < P; ++k) {
                    if (dp[i][j][k].cost < INF) {
                        float final_c = dp[i][j][k].cost + tail_penalty;
                        if (final_c < min_end) {
                            min_end = final_c;
                            ei = i; ej = j; ek = k;
                        }
                    }
                }
            }
        }

        // 路径重建
        std::vector<MatchResult> local_res;
        while (ei != -1) {
            // 重新计算一下 info 用于保存
            auto info = computeCost(band_l.slices[ei], band_r.slices[ej], surfaces[ek].coefficients, rectify_l, rectify_r);
            
            MatchResult res;
            res.l_slice_id = band_l.slices[ei]->id;       // 左切片全局ID
            res.r_slice_id = band_r.slices[ej]->id;     // 右切片全局ID
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
        // printf("slice L_ID=%d-R_ID=%d-P_ID=%d | Score=%.2f, dis_mean=%.2f, dis_mode=%.2f, dis_stddev=%.2f, norm_census=%.2f\n",
        //        res.l_slice_id, res.r_slice_id, res.p_idx, res.info.score, res.info.dis_mean, res.info.dis_mode, res.info.dis_stddev, res.info.norm_census);
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
            cv::putText(canvas, std::to_string(s->id), cv::Point2f(s->center_pt.x+13, s->center_pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 1);
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
                std::string txt = std::to_string(res->p_idx) + "<-" + std::to_string(res->l_slice_id);
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
    float w_mode = 1.0f;   // 几何精度（众数距离）
    float w_std = 0.5f;    // 几何一致性（标准差）
    float w_cen = 0.2f;    // 纹理代价

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
