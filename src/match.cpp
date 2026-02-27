#include "match.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>
#include <mutex>

// ============================================================
// MatchProcessor::match
// 执行 DP 条带匹配，返回匹配结果并填充 MatchStats
// ============================================================
std::vector<MatchResult>
MatchProcessor::match(const std::vector<LaserLine>& laser_l,
                      const std::vector<LaserLine>& laser_r,
                      const cv::Mat& rectify_l, const cv::Mat& rectify_r,
                      MatchStats& stats) {

    // 0. 参数准备（不使用 static，避免 TBB 并行区数据竞争）
    const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
    const int P = static_cast<int>(surfaces.size());
    const auto& calib = ConfigManager::getInstance().getCalibInfo();
    double fx_r    = calib.P[1].at<double>(0, 0);
    double baseline = std::abs(calib.P[1].at<double>(0, 3) / fx_r);

    // 1. 生成双向条带
    std::vector<Band> bands_l = generateBands(laser_l, calib.P[0], rectify_l.rows);
    std::vector<Band> bands_r = generateBands(laser_r, calib.P[1], rectify_r.rows);
    int num_bands = static_cast<int>(std::min(bands_l.size(), bands_r.size()));

    std::vector<MatchResult> all_results;
    std::mutex res_mutex;

    // 2. 并行处理每一条带
    tbb::parallel_for(0, num_bands, [&](int b_idx) {
        const auto& band_l = bands_l[b_idx];
        const auto& band_r = bands_r[b_idx];
        int NL = static_cast<int>(band_l.slices.size());
        int NR = static_cast<int>(band_r.slices.size());
        if (NL == 0 || NR == 0) return;

        const float INF            = 1e9f;
        const float COST_THRESHOLD = 8.5f;
        const float COST_SKIP      = 10.0f;

        // DP 状态：dp[i][j][k]
        std::vector<std::vector<std::vector<DpNode>>> dp(
            NL, std::vector<std::vector<DpNode>>(
                NR, std::vector<DpNode>(P, {INF, -1, -1, -1})));

        // 初始化第 0 个左切片
        for (int j = 0; j < NR; ++j)
            for (int k = 0; k < P; ++k) {
                auto info = computeCost(band_l.slices[0], band_r.slices[j],
                                        surfaces[k].coefficients, rectify_l, rectify_r);
                if (info.score < COST_THRESHOLD)
                    dp[0][j][k] = {info.score, -1, -1, -1};
            }

        // 状态转移
        for (int i = 1; i < NL; ++i)
            for (int j = 0; j < NR; ++j)
                for (int k = 0; k < P; ++k) {
                    auto info = computeCost(band_l.slices[i], band_r.slices[j],
                                            surfaces[k].coefficients, rectify_l, rectify_r);
                    if (info.score >= COST_THRESHOLD) continue;

                    float min_prev = INF;
                    int best_pi = -1, best_pj = -1, best_pk = -1;

                    // 策略 A：连接到前驱（允许小范围回溯）
                    int lb_l = std::max(0, i - 5);
                    for (int pi = i-1; pi >= lb_l; --pi) {
                        float l_skip = (i - 1 - pi) * COST_SKIP;
                        int lb_r = std::max(0, j - 10);
                        for (int pj = j-1; pj >= lb_r; --pj) {
                            // X 方向必须单调递增
                            float dL = band_l.slices[i]->center_pt.x - band_l.slices[pi]->center_pt.x;
                            float dR = band_r.slices[j]->center_pt.x - band_r.slices[pj]->center_pt.x;
                            if (dL < 0.1f || dR < 0.1f) continue;
                            for (int pk = 0; pk < k; ++pk) {
                                if (dp[pi][pj][pk].cost >= INF) continue;
                                int p_diff = k - pk;
                                if (p_diff > i - pi + 1) continue;
                                if (p_diff > j - pj + 1) continue;
                                float total = dp[pi][pj][pk].cost + l_skip + info.score;
                                if (total < min_prev) {
                                    min_prev = total;
                                    best_pi = pi; best_pj = pj; best_pk = pk;
                                }
                            }
                        }
                    }

                    // 策略 B：作为新起点（右切片索引较小时才允许）
                    if (j < 6) {
                        float start_cost = i * COST_SKIP + info.score;
                        if (start_cost < min_prev) {
                            min_prev = start_cost;
                            best_pi = -1; best_pj = -1; best_pk = -1;
                        }
                    }

                    if (min_prev < INF)
                        dp[i][j][k] = {min_prev, best_pi, best_pj, best_pk};
                }

        // 回溯：找全局最优终点
        float min_end = INF;
        int ei = -1, ej = -1, ek = -1;
        for (int i = 0; i < NL; ++i) {
            float tail = (NL - 1 - i) * COST_SKIP;
            for (int j = 0; j < NR; ++j)
                for (int k = 0; k < P; ++k)
                    if (dp[i][j][k].cost < INF) {
                        float fc = dp[i][j][k].cost + tail;
                        if (fc < min_end) { min_end = fc; ei = i; ej = j; ek = k; }
                    }
        }

        // 路径重建
        std::vector<MatchResult> local_res;
        while (ei != -1) {
            auto info = computeCost(band_l.slices[ei], band_r.slices[ej],
                                    surfaces[ek].coefficients, rectify_l, rectify_r);
            MatchResult res;
            res.band_id    = band_l.idx;
            res.l_slice_id = band_l.slices[ei]->id;
            res.r_slice_id = band_r.slices[ej]->id;
            res.l_idx      = band_l.slices[ei]->laser_idx;
            res.r_idx      = band_r.slices[ej]->laser_idx;
            res.p_idx      = ek;
            res.info       = info;
            local_res.push_back(res);

            auto& node = dp[ei][ej][ek];
            ei = node.prev_l; ej = node.prev_r; ek = node.prev_p;
        }

        if (!local_res.empty()) {
            std::lock_guard<std::mutex> lock(res_mutex);
            all_results.insert(all_results.end(), local_res.begin(), local_res.end());
        }
    });

    // 3. 填充匹配统计（用于计算完成率）
    {
        // 统计参与匹配的左/右线
        std::unordered_map<int, int> l_total, r_total; // laser_idx -> 总切片数
        std::unordered_set<int>      matched_l_ids, matched_r_ids; // 已匹配切片 ID

        for (const auto& res : all_results) {
            matched_l_ids.insert(res.l_slice_id);
            matched_r_ids.insert(res.r_slice_id);
        }

        // 统计参与匹配的激光线所有切片总数
        std::unordered_set<int> active_l, active_r;
        for (const auto& res : all_results) { active_l.insert(res.l_idx); active_r.insert(res.r_idx); }

        for (const auto& band : bands_l)
            for (const auto* s : band.slices)
                if (active_l.count(s->laser_idx)) l_total[s->laser_idx]++;

        for (const auto& band : bands_r)
            for (const auto* s : band.slices)
                if (active_r.count(s->laser_idx)) r_total[s->laser_idx]++;

        stats.total_l_slices   = 0;
        stats.total_r_slices   = 0;
        for (auto& [k, v] : l_total) stats.total_l_slices += v;
        for (auto& [k, v] : r_total) stats.total_r_slices += v;
        stats.matched_l_slices = static_cast<int>(matched_l_ids.size());
        stats.matched_r_slices = static_cast<int>(matched_r_ids.size());
    }

    return all_results;
}

// ============================================================
// filterMatches：修复光平面跳变（ABA / AAAB 模式）
// ============================================================
void MatchProcessor::filterMatches(std::vector<MatchResult>& matches,
                                   const std::vector<Band>& bands_l,
                                   const std::vector<Band>& bands_r) {
    if (matches.empty()) return;

    // 按 l_idx 分组
    std::map<int, std::vector<MatchResult*>> l_groups;
    for (auto& m : matches)
        if (m.l_idx != -1) l_groups[m.l_idx].push_back(&m);

    for (auto& [l_id, group] : l_groups) {
        if (group.empty()) continue;
        std::sort(group.begin(), group.end(),
                  [](MatchResult* a, MatchResult* b){ return a->band_id < b->band_id; });

        // RLE 分析光平面 ID
        std::vector<Run> runs;
        int curr_p = group[0]->p_idx, curr_start = 0, curr_len = 1;
        for (size_t i = 1; i < group.size(); ++i) {
            if (group[i]->p_idx == curr_p) {
                curr_len++;
            } else {
                runs.push_back({curr_p, curr_start, curr_len});
                curr_p = group[i]->p_idx; curr_start = static_cast<int>(i); curr_len = 1;
            }
        }
        runs.push_back({curr_p, curr_start, curr_len});

        for (size_t i = 1; i < runs.size(); ++i) {
            bool need_fix = false;
            int  target_run_idx = -1;

            // ABA 模式：中间段
            if (i < runs.size() - 1) {
                if (runs[i-1].p_idx == runs[i+1].p_idx && runs[i].len < 2) {
                    need_fix = true; target_run_idx = static_cast<int>(i-1);
                }
            }
            // AAAB 模式：尾段
            if (!need_fix && i == runs.size() - 1 && runs[i].len < 3) {
                need_fix = true; target_run_idx = static_cast<int>(i-1);
            }

            if (!need_fix || target_run_idx < 0) continue;

            // 求参考段 r_idx 众数
            std::unordered_map<int,int> r_cnt;
            const Run& ref = runs[target_run_idx];
            for (int k = ref.start_idx; k < ref.start_idx + ref.len; ++k)
                if (group[k]->r_idx != -1) r_cnt[group[k]->r_idx]++;

            int target_r = -1, max_f = -1;
            for (auto& [rid, cnt] : r_cnt)
                if (cnt > max_f) { max_f = cnt; target_r = rid; }

            if (target_r < 0) continue;
            int target_p = ref.p_idx;

            const Run& curr = runs[i];
            for (int k = curr.start_idx; k < curr.start_idx + curr.len; ++k) {
                const Band& band = bands_r[group[k]->band_id];
                auto it = band.laser2slice.find(target_r);
                if (it != band.laser2slice.end()) {
                    group[k]->r_idx      = target_r;
                    group[k]->r_slice_id = it->second->id;
                    group[k]->p_idx      = target_p;
                } else {
                    group[k]->r_idx = -1;
                    group[k]->l_idx = -1;
                }
            }
        }
    }

    // 删除无效匹配
    matches.erase(
        std::remove_if(matches.begin(), matches.end(),
                       [](const MatchResult& m){ return m.r_idx == -1; }),
        matches.end());
}

// ============================================================
// findIntersection：射线与二次曲面求交
// ============================================================
std::vector<cv::Point3f>
MatchProcessor::findIntersection(const cv::Point3f& point,
                                 const cv::Point3f& normal,
                                 const cv::Mat& Coeff6x1) {
    float a = Coeff6x1.at<float>(0), b = Coeff6x1.at<float>(1);
    float c = Coeff6x1.at<float>(2), d = Coeff6x1.at<float>(3);
    float e = Coeff6x1.at<float>(4), f = Coeff6x1.at<float>(5);

    float A = a*normal.y*normal.y + b*normal.y*normal.z + c*normal.z*normal.z;
    float B = 2*a*point.y*normal.y
            + b*(point.y*normal.z + point.z*normal.y)
            + 2*c*point.z*normal.z
            + d*normal.y + e*normal.z - normal.x;
    float C = a*point.y*point.y + b*point.y*point.z + c*point.z*point.z
            + d*point.y + e*point.z + f - point.x;

    std::vector<cv::Point3f> intersections;
    float disc = B*B - 4*A*C;
    if (std::abs(A) < 1e-6f) {
        if (std::abs(B) > 1e-6f) intersections.push_back(point + (-C/B)*normal);
    } else if (disc > 0) {
        float sq = std::sqrt(disc);
        intersections.push_back(point + ((-B+sq)/(2*A))*normal);
        intersections.push_back(point + ((-B-sq)/(2*A))*normal);
    } else if (std::abs(disc) < 1e-6f) {
        intersections.push_back(point + (-B/(2*A))*normal);
    }
    return intersections;
}

// ============================================================
// computeCensus：Census 变换代价
// ============================================================
int MatchProcessor::computeCensus(const cv::Mat& img1, const cv::Point2f& pt1,
                                  const cv::Mat& img2, const cv::Point2f& pt2,
                                  int window_size) {
    int cx1 = cvRound(pt1.x), cy1 = cvRound(pt1.y);
    int cx2 = cvRound(pt2.x), cy2 = cvRound(pt2.y);
    int hw = window_size / 2;
    if (cx1 < hw || cx1 >= img1.cols-hw || cy1 < hw || cy1 >= img1.rows-hw ||
        cx2 < hw || cx2 >= img2.cols-hw || cy2 < hw || cy2 >= img2.rows-hw)
        return window_size * window_size;

    int cost = 0;
    int c1 = img1.at<uchar>(cy1, cx1);
    int c2 = img2.at<uchar>(cy2, cx2);
    for (int dy = -hw; dy <= hw; ++dy)
        for (int dx = -hw; dx <= hw; ++dx) {
            bool b1 = img1.at<uchar>(cy1+dy, cx1+dx) > c1;
            bool b2 = img2.at<uchar>(cy2+dy, cx2+dx) > c2;
            if (b1 != b2) ++cost;
        }
    return cost;
}

// ============================================================
// generateBands：将激光线分割为水平条带切片
// ============================================================
std::vector<Band> MatchProcessor::generateBands(const std::vector<LaserLine>& laser,
                                                 const cv::Mat& P,
                                                 int img_rows) {
    int num_bands = static_cast<int>(std::ceil((float)img_rows / SLICE_HEIGHT_));
    std::vector<Band> bands(num_bands);
    for (int i = 0; i < num_bands; ++i) bands[i].idx = i;

    float fx = static_cast<float>(P.at<double>(0,0));
    float fy = static_cast<float>(P.at<double>(1,1));
    float cx = static_cast<float>(P.at<double>(0,2));
    float cy = static_cast<float>(P.at<double>(1,2));

    std::vector<LeftPoint> current_pts;
    current_pts.reserve(static_cast<int>(SLICE_HEIGHT_ / precision_));  // precision_=0.5，展开后约 60 个点

    for (int l_idx = 0; l_idx < static_cast<int>(laser.size()); ++l_idx) {
        const auto& line = laser[l_idx];
        if (line.empty()) continue;

        int current_band_idx = -1;

        // 使用 y_coords / points 遍历（新 LaserLine 结构）
        for (size_t pt_i = 0; pt_i < line.size(); ++pt_i) {
            float y = line.y_coords[pt_i];
            float x = line.points[pt_i].x;

            int band_idx = static_cast<int>(y / SLICE_HEIGHT_);
            if (band_idx < 0 || band_idx >= num_bands) continue;

            if (band_idx != current_band_idx) {
                // 保存上一个切片
                if (current_band_idx != -1 && current_pts.size() >= 15) {
                    auto s = std::make_unique<Slice>();
                    s->laser_idx = l_idx;
                    s->band_idx  = current_band_idx;
                    int mid = static_cast<int>(current_pts.size()) / 2;
                    s->center_pt = cv::Point2f(current_pts[mid].x, current_pts[mid].y);
                    s->pts = std::move(current_pts);
                    bands[current_band_idx].addSlice(std::move(s));
                }
                current_band_idx = band_idx;
                current_pts.clear();
            }

            float rx = (x - cx) / fx, ry = (y - cy) / fy, rz = 1.0f;
            float rn = std::sqrt(rx*rx + ry*ry + rz*rz);
            current_pts.emplace_back(x, y, cv::Point3f(rx/rn, ry/rn, rz/rn));
        }

        // 收尾
        if (current_band_idx != -1 && current_pts.size() >= 15) {
            auto s = std::make_unique<Slice>();
            s->laser_idx = l_idx;
            s->band_idx  = current_band_idx;
            float sx = 0, sy = 0;
            for (const auto& p : current_pts) { sx += p.x; sy += p.y; }
            s->center_pt = cv::Point2f(sx / current_pts.size(), sy / current_pts.size());
            s->pts = std::move(current_pts);
            bands[current_band_idx].addSlice(std::move(s));
        }
        current_pts.clear();
    }

    // 条带内按 X 坐标排序
    tbb::parallel_for(0, num_bands, [&](int i) {
        std::sort(bands[i].slices.begin(), bands[i].slices.end(),
                  [](const Slice* a, const Slice* b){ return a->center_pt.x < b->center_pt.x; });
    });

    // 全局 ID 分配
    int global_id = 0;
    for (int i = 0; i < num_bands; ++i)
        for (auto* s : bands[i].slices) s->id = global_id++;

    for (auto& b : bands) b.buildLaserToSlice();
    return bands;
}

// ============================================================
// computeCost：计算一对切片的匹配代价
// ============================================================
ScoreInfo MatchProcessor::computeCost(const Slice* slice_l, const Slice* slice_r,
                                       const cv::Mat& coef,
                                       const cv::Mat& rectify_l, const cv::Mat& rectify_r) {
    ScoreInfo info = {COST_INVALID_, 0, 0, 0, 0};

    const auto calib = ConfigManager::getInstance().getCalibInfo();
    double fx_l = calib.P[0].at<double>(0,0), fy_l = calib.P[0].at<double>(1,1);
    double cx_l = calib.P[0].at<double>(0,2), cy_l = calib.P[0].at<double>(1,2);
    double fx_r = calib.P[1].at<double>(0,0), fy_r = calib.P[1].at<double>(1,1);
    double cx_r = calib.P[1].at<double>(0,2), cy_r = calib.P[1].at<double>(1,2);
    double baseline = 1.0 / calib.Q.at<double>(3, 2);

    // 快速剪枝：中心点预检
    auto ips_center = findIntersection(
        {0,0,0},
        {float((slice_l->center_pt.x-cx_l)/fx_l),
         float((slice_l->center_pt.y-cy_l)/fy_l), 1.0f},
        coef);
    if (ips_center.empty()) return info;
    cv::Point3f pt3c;
    bool valid_c = false;
    for (auto& q : ips_center)
        if (q.z > 100 && q.z < 600) { pt3c = q; valid_c = true; break; }
    if (!valid_c) return info;

    cv::Point3f proj(pt3c.x - float(baseline), pt3c.y, pt3c.z);
    float proj_x = float(fx_r * proj.x / proj.z + cx_r);
    if (std::abs(proj_x - slice_r->center_pt.x) > 8.0f) return info;

    // 逐点匹配
    struct MatchPair { float dist; cv::Point2f pt_l, pt_r; };
    std::vector<MatchPair> valid_pts;
    valid_pts.reserve(slice_l->pts.size());

    for (const auto& lp : slice_l->pts) {
        auto ips = findIntersection({0,0,0}, lp.ray, coef);
        if (ips.empty()) continue;
        cv::Point3f pt3;
        bool ok = false;
        for (auto& q : ips) if (q.z > 100 && q.z < 600) { pt3 = q; ok = true; break; }
        if (!ok) continue;

        cv::Point3f pr(pt3.x - float(baseline), pt3.y, pt3.z);
        float xr = float(fx_r * pr.x / pr.z + cx_r);
        float yr = lp.y;
        if (xr < 0 || xr >= rectify_l.cols) continue;

        float min_d = 1e9f;
        const LeftPoint* best_rp = nullptr;
        for (const auto& rp : slice_r->pts) {
            if (std::abs(rp.y - yr) < EPS_) {
                float d = std::abs(rp.x - xr);
                if (d < min_d) { min_d = d; best_rp = &rp; }
            }
        }
        if (min_d <= D_thresh_ && best_rp)
            valid_pts.push_back({min_d, {lp.x, lp.y}, {best_rp->x, best_rp->y}});
    }

    if (valid_pts.empty() || valid_pts.size() < slice_l->pts.size() * 0.7) return info;

    // 统计均值、标准差
    float sum = 0;
    for (auto& p : valid_pts) sum += p.dist;
    info.dis_mean = sum / valid_pts.size();
    float sq = 0;
    for (auto& p : valid_pts) sq += (p.dist - info.dis_mean) * (p.dist - info.dis_mean);
    info.dis_stddev = std::sqrt(sq / valid_pts.size());

    // 众数（直方图法）
    const float bin_w = 0.15f;
    std::map<int,int> hist;
    for (const auto& mp : valid_pts) hist[std::round(mp.dist/bin_w)]++;
    int max_cnt = 0, mode_bin = 0;
    for (auto& kv : hist) if (kv.second > max_cnt) { max_cnt = kv.second; mode_bin = kv.first; }
    info.dis_mode = mode_bin * bin_w;

    // Census 代价（取 1/4、1/2、3/4 处各一点）
    std::vector<size_t> idxs = {valid_pts.size()/4, valid_pts.size()/2, 3*valid_pts.size()/4};
    int census = 0;
    for (size_t idx : idxs)
        census += computeCensus(rectify_l, valid_pts[idx].pt_l,
                                rectify_r, valid_pts[idx].pt_r);
    info.norm_census = float(census) / idxs.size();

    info.score = 0.75f * info.dis_mode + 1.2f * info.dis_stddev + 1.0f * info.norm_census;
    return info;
}

// ============================================================
// getBandColor：条带可视化颜色（调试用，仍保留供外部调用）
// ============================================================
cv::Scalar MatchProcessor::getBandColor(int band_idx) {
    static const std::vector<cv::Scalar> palette = {
        {255,0,0},{0,255,0},{0,0,255},{255,255,0},
        {255,0,255},{0,255,255},{128,0,0},{0,128,0},
        {0,0,128},{128,128,0},{128,0,128},{0,128,128}
    };
    return palette[band_idx % palette.size()];
}


// ============================================================
// visualizeSlices
// 仅绘制匹配的切片（同一条带同色），左图标注 "S<id>"，右图标注对应左切片 ID
// ============================================================
cv::Mat MatchProcessor::visualizeSlices(
    const std::vector<LaserLine>& laser_l,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r,
    const std::vector<MatchResult>& results) {

    cv::Mat color_l, color_r, canvas;
    cv::cvtColor(rectify_l, color_l, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectify_r, color_r, cv::COLOR_GRAY2BGR);
    cv::hconcat(color_l, color_r, canvas);
    int off = rectify_l.cols;

    const auto& calib = ConfigManager::getInstance().getCalibInfo();
    auto bands_l = generateBands(laser_l, calib.P[0], rectify_l.rows);
    auto bands_r = generateBands(laser_r, calib.P[1], rectify_r.rows);

    // 匹配映射
    std::unordered_set<int> matched_l_ids;
    std::unordered_map<int, int> r_to_l_id;
    for (const auto& res : results) {
        matched_l_ids.insert(res.l_slice_id);
        r_to_l_id[res.r_slice_id] = res.l_slice_id;
    }

    // 左图：仅绘制匹配的切片
    for (const auto& band : bands_l) {
        cv::Scalar col = getBandColor(band.idx);
        for (const Slice* s : band.slices) {
            if (!matched_l_ids.count(s->id)) continue;
            for (const auto& lp : s->pts) {
                int px = cvRound(lp.x), py = cvRound(lp.y);
                if (px >= 0 && px < canvas.cols && py >= 0 && py < canvas.rows)
                    canvas.at<cv::Vec3b>(py, px) =
                        cv::Vec3b((uchar)col[0], (uchar)col[1], (uchar)col[2]);
            }
            int tx = cvRound(s->center_pt.x) + 5, ty = cvRound(s->center_pt.y);
            if (tx >= 3 && tx < canvas.cols - 40 && ty >= 5 && ty < canvas.rows - 5)
                cv::putText(canvas, "S" + std::to_string(s->id),
                            cv::Point(tx, ty),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv::LINE_AA);
        }
    }

    // 右图：仅绘制匹配的切片，标注对应左切片 ID
    for (const auto& band : bands_r) {
        cv::Scalar col = getBandColor(band.idx);
        for (const Slice* s : band.slices) {
            auto it = r_to_l_id.find(s->id);
            if (it == r_to_l_id.end()) continue;
            for (const auto& lp : s->pts) {
                int px = cvRound(lp.x) + off, py = cvRound(lp.y);
                if (px >= 0 && px < canvas.cols && py >= 0 && py < canvas.rows)
                    canvas.at<cv::Vec3b>(py, px) =
                        cv::Vec3b((uchar)col[0], (uchar)col[1], (uchar)col[2]);
            }
            int tx = cvRound(s->center_pt.x) + off + 5, ty = cvRound(s->center_pt.y);
            if (tx >= 3 && tx < canvas.cols - 40 && ty >= 5 && ty < canvas.rows - 5)
                cv::putText(canvas, "S" + std::to_string(it->second),
                            cv::Point(tx, ty),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv::LINE_AA);
        }
    }

    return canvas;
}
