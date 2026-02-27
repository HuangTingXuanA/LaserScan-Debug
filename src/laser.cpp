#include "laser.h"
#include <fstream>

// ============================================================
// 工具函数：取最近奇数
// ============================================================
int LaserProcessor::convert_to_odd_number(float num) {
    int r = static_cast<int>(std::round(num));
    if (r % 2 != 0) return r;
    float dp = std::abs(num - (r - 1));
    float dn = std::abs(num - (r + 1));
    return (dp < dn) ? r - 1 : r + 1;
}

// ============================================================
// 一维高斯导数核（用于 Steger 中心提取，目前仅保留供参考）
// ============================================================
cv::Mat LaserProcessor::computeGaussianDerivatives(float sigma, float epsilon) {
    float x0 = sigma * std::sqrt(-2.f * std::log(epsilon / 2.f));
    int radius = static_cast<int>(std::ceil(x0));
    int ksize = 2 * radius + 1;
    cv::Mat kernel(ksize, 1, CV_32F);
    auto phi = [](float x, float s) {
        return 0.5f * (1.f + std::erf(x / (s * std::sqrt(2.f))));
    };
    float sum = 0;
    for (int i = 0; i < ksize; ++i) {
        float xv = i - radius;
        float v = phi(xv + 0.5f, sigma) - phi(xv - 0.5f, sigma);
        kernel.at<float>(i) = v;
        sum += v;
    }
    kernel /= sum;
    return kernel;
}

// ============================================================
// 双线性插值（float / double 两版）
// ============================================================
float LaserProcessor::interpolateChannel(const cv::Mat& img, float x, float y) {
    int xi = std::clamp(static_cast<int>(x), 0, img.cols - 2);
    int yi = std::clamp(static_cast<int>(y), 0, img.rows - 2);
    float dx = x - xi, dy = y - yi;
    const float* r0 = img.ptr<float>(yi);
    const float* r1 = img.ptr<float>(yi + 1);
    return r0[xi]*(1-dx)*(1-dy) + r0[xi+1]*dx*(1-dy)
         + r1[xi]*(1-dx)*dy    + r1[xi+1]*dx*dy;
}
double LaserProcessor::interpolateChannel(const cv::Mat& img, double x, double y) {
    int xi = std::clamp(static_cast<int>(x), 0, img.cols - 2);
    int yi = std::clamp(static_cast<int>(y), 0, img.rows - 2);
    double dx = x-xi, dy = y-yi;
    const float* r0 = img.ptr<float>(yi);
    const float* r1 = img.ptr<float>(yi+1);
    return r0[xi]*(1-dx)*(1-dy) + r0[xi+1]*dx*(1-dy)
         + r1[xi]*(1-dx)*dy     + r1[xi+1]*dx*dy;
}

// ============================================================
// 质心法中心提取（findSymmetricCenter4）
// 在给定方向 dir 的 [-2, R] 范围内采样，以灰度加权质心作为中心
// ============================================================
float LaserProcessor::findSymmetricCenter4(const cv::Mat& img, float x, float y,
                                            cv::Vec2f dir, float R) {
    const float step = 0.1f;
    std::vector<float> Ts, Vs;
    Ts.reserve(static_cast<size_t>((R + 2.f) / step) + 1);
    Vs.reserve(Ts.capacity());

    for (float t = -2.f; t <= R; t += step) {
        float xt = x + t * dir[0], yt = y + t * dir[1];
        int xi = std::round(xt), yi = std::round(yt);
        if (xi < 0 || xi >= img.cols || yi < 0 || yi >= img.rows) continue;
        Ts.push_back(t);
        Vs.push_back(img.at<float>(yt, xt));
    }
    int N = static_cast<int>(Ts.size());
    if (N < 3) return FLT_MAX;

    // 找峰值
    float maxV = -FLT_MAX; int peakIdx = -1;
    for (int i = 0; i < N; ++i)
        if (Vs[i] > maxV && Ts[i] <= R) { maxV = Vs[i]; peakIdx = i; }
    if (Vs[peakIdx] < 125.f) return FLT_MAX;

    // 灰度加权质心
    double num = 0, den = 0;
    for (int i = 0; i < N; ++i)
        if (Vs[i] > 0) { num += Ts[i]*Vs[i]; den += Vs[i]; }
    return static_cast<float>(num / den) + 0.1f;
}

// ============================================================
// 逐行灰度加权质心（用于水平方向扫描线）
// ============================================================
float LaserProcessor::computeRowCentroid(const cv::Mat& img, int y, float xL, float xR) {
    if (y < 0 || y >= img.rows) return 0.5f*(xL+xR);
    if (xL > xR) std::swap(xL, xR);
    xL = std::clamp(xL, 0.f, (float)img.cols-1.f);
    xR = std::clamp(xR, 0.f, (float)img.cols-1.f);
    if (xR - xL < 1e-3f) return 0.5f*(xL+xR);

    const float* row = img.ptr<float>(y);
    double A = 0, M = 0;
    int xiL = (int)std::floor(xL), xiR = (int)std::floor(xR);

    if (xiL < img.cols-1) {
        float IL = row[xiL] + (row[xiL+1]-row[xiL])*(xL-xiL);
        double dx = xiL+1-xL, avg = 0.5*(IL+row[xiL+1]);
        A += avg*dx; M += avg*((xL+xiL+1)*0.5)*dx;
    }
    for (int i = xiL+1; i < xiR; ++i) {
        double avg = 0.5*(row[i]+row[i+1]);
        A += avg; M += avg*(i+0.5);
    }
    if (xiR < img.cols-1) {
        float IR = row[xiR] + (row[xiR+1]-row[xiR])*(xR-xiR);
        double dx = xR-xiR, avg = 0.5*(row[xiR]+IR);
        A += avg*dx; M += avg*((xiR+xR)*0.5)*dx;
    }
    return (A <= 1e-9) ? 0.5f*(xL+xR) : float(M/A);
}

// ============================================================
// processCenters：相邻浮点中心点插值到精度网格
// ============================================================
std::vector<cv::Point2f>
LaserProcessor::processCenters(const std::map<float, float>& orig) {
    std::vector<cv::Point2f> out;
    if (orig.empty() || precision_ <= 0) return out;

    // 1. 对齐原始点
    std::map<int, float> k2x;
    for (auto& [y, x] : orig) {
        int k = int(std::lround(y / precision_));
        if (std::abs(y - k*precision_) < EPS_)
            k2x.emplace(k, x);
    }

    // 2. 相邻对插值
    auto prev = orig.begin();
    for (auto it = std::next(orig.begin()); it != orig.end(); ++it, ++prev) {
        float yA = prev->first, xA = prev->second;
        float yB = it->first,   xB = it->second;
        if (yB <= yA + EPS_ || (yB-yA) > 1.5f + EPS_) continue;
        int k0 = int(std::ceil ((yA+EPS_)/precision_));
        int k1 = int(std::floor((yB+EPS_)/precision_));
        for (int k = k0; k <= k1; ++k) {
            if (k2x.count(k)) continue;
            float yi = k * precision_;
            float t  = (yi - yA) / (yB - yA);
            k2x.emplace(k, xA + t*(xB-xA));
        }
    }

    // 3. 有序输出（Point2f(x, y)）
    out.reserve(k2x.size());
    for (auto& [k, x] : k2x)
        out.emplace_back(x, k * precision_);
    return out;
}

// ============================================================
// getAxisEndpoints：取旋转矩形两条短边的中点（主轴两端点）
// ============================================================
std::pair<cv::Point2f, cv::Point2f>
LaserProcessor::getAxisEndpoints(const cv::RotatedRect& rect) {
    cv::Point2f v[4]; rect.points(v);
    float min_len = FLT_MAX; int si = 0;
    for (int i = 0; i < 4; ++i) {
        float l = cv::norm(v[i] - v[(i+1)%4]);
        if (l < min_len) { min_len = l; si = i; }
    }
    int oi = (si+2)%4;
    return {0.5f*(v[si]+v[(si+1)%4]), 0.5f*(v[oi]+v[(oi+1)%4])};
}

// ============================================================
// findIntersection：射线与二次曲面求交 (x = ay²+byz+cz²+dy+ez+f)
// ============================================================
std::vector<cv::Point3f>
LaserProcessor::findIntersection(const cv::Point3f& point,
                                  const cv::Point3f& normal,
                                  const cv::Mat& Coeff6x1) {
    float a=Coeff6x1.at<float>(0), b=Coeff6x1.at<float>(1);
    float c=Coeff6x1.at<float>(2), d=Coeff6x1.at<float>(3);
    float e=Coeff6x1.at<float>(4), f=Coeff6x1.at<float>(5);

    float A = a*normal.y*normal.y + b*normal.y*normal.z + c*normal.z*normal.z;
    float B = 2*a*point.y*normal.y + b*(point.y*normal.z+point.z*normal.y)
            + 2*c*point.z*normal.z + d*normal.y + e*normal.z - normal.x;
    float C = a*point.y*point.y + b*point.y*point.z + c*point.z*point.z
            + d*point.y + e*point.z + f - point.x;

    std::vector<cv::Point3f> ips;
    float disc = B*B - 4*A*C;
    if (std::abs(A) < 1e-6f) {
        if (std::abs(B) > 1e-6f) ips.push_back(point + (-C/B)*normal);
    } else if (disc > 0) {
        float sq = std::sqrt(disc);
        ips.push_back(point + ((-B+sq)/(2*A))*normal);
        ips.push_back(point + ((-B-sq)/(2*A))*normal);
    } else if (std::abs(disc) < 1e-6f) {
        ips.push_back(point + (-B/(2*A))*normal);
    }
    return ips;
}

// ============================================================
// evaluateQuadSurf：计算点到曲面的距离（用于标定评估）
// ============================================================
double LaserProcessor::evaluateQuadSurf(const cv::Mat& C, const cv::Point3f& p) {
    float a=C.at<float>(0),b=C.at<float>(1),c_=C.at<float>(2);
    float d=C.at<float>(3),e=C.at<float>(4),f=C.at<float>(5);
    float xe = a*p.y*p.y + b*p.y*p.z + c_*p.z*p.z + d*p.y + e*p.z + f;
    float res = std::abs(xe - p.x);
    cv::Point3f norm(1, -2*a*p.y-b*p.z-d, -b*p.y-2*c_*p.z-e);
    for (auto& ip : findIntersection(p, norm, C))
        res = std::min(res, (float)cv::norm(ip-p));
    return res;
}
double LaserProcessor::evaluateQuadSurf(const cv::Mat& C,
                                         const std::vector<cv::Point3f>& pts) {
    float ss = 0;
    for (auto& p : pts) ss += std::pow(float(evaluateQuadSurf(C,p)), 2.f);
    return std::sqrt(ss / pts.size());
}

// ============================================================
// computeEnhancedScore：综合评分（median距离 + 标准差 + 覆盖率）
// ============================================================
float LaserProcessor::computeEnhancedScore(
    const std::vector<std::pair<float,float>>& distance_pairs,
    int pts_repro_cnt, float& coverage, float& std_dev) {

    if (distance_pairs.empty() || pts_repro_cnt <= 0) return FLT_MAX;

    std::vector<float> ds;
    ds.reserve(distance_pairs.size());
    for (auto& [y,d] : distance_pairs) ds.push_back(d);

    // 中位数
    std::nth_element(ds.begin(), ds.begin()+ds.size()/2, ds.end());
    float median = ds[ds.size()/2];

    // 标准差
    float mean = std::accumulate(ds.begin(),ds.end(),0.f)/ds.size();
    float var = 0;
    for (float v : ds) var += (v-mean)*(v-mean);
    std_dev = std::sqrt(var/ds.size());

    // 覆盖率
    coverage = float(distance_pairs.size()) / pts_repro_cnt;

    // 线长惩罚
    const int thr = 250;
    float short_pen = (pts_repro_cnt < thr) ? (thr-pts_repro_cnt)*0.15f : 0.f;

    float w_std = (int)distance_pairs.size() > thr ? 1.5f : 2.5f;
    return 0.25f*median + w_std*std_dev + 23.f*(0.9f-coverage) + short_pen;
}

// ============================================================
// extractLine2：从连通区域提取激光线中心点（质心法）
// 输入：极线校正图、连通区域对（左右边沿点对）
// 输出：有序 LaserLine 列表
// ============================================================
std::vector<LaserLine> LaserProcessor::extractLine2(
    const cv::Mat& rectify_img,
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
    int /*img_idx*/) {

    std::vector<LaserLine> laser_lines;
    cv::Mat img_float;
    rectify_img.convertTo(img_float, CV_32F);

    for (const auto& edge_pair : contours) {
        // 1. 激光线中心点提取（质心法，水平方向）
        std::map<float, float> orign_centers; // y -> x
        for (const auto& p : edge_pair) {
            float search_range = float(p.second.x - p.first.x + 1);
            cv::Vec2f dir(1, 0);
            float t_peak = findSymmetricCenter4(img_float, float(p.first.x),
                                                float(p.first.y), dir, search_range);
            if (t_peak == FLT_MAX) continue;
            orign_centers[float(p.first.y)] = p.first.x + t_peak;
        }
        if (orign_centers.empty()) continue;

        // 2. 插值到精度网格
        std::vector<cv::Point2f> centers = processCenters(orign_centers);
        if (centers.empty()) continue;

        // 3. 计算激光线角度（用旋转矩形的最小外接矩形近似）
        std::vector<cv::Point> ctpts;
        for (auto& ep : edge_pair) {
            ctpts.push_back(ep.first); ctpts.push_back(ep.second);
        }
        cv::RotatedRect rr = cv::minAreaRect(ctpts);
        float angle_deg = rr.angle; // OpenCV 旋转矩形角度

        // 4. 构建 LaserLine（vector + y_coords，有序输入）
        std::vector<LaserPoint> pts;
        std::vector<float>      ys;
        pts.reserve(centers.size()); ys.reserve(centers.size());
        for (auto& c : centers) {
            pts.emplace_back(c.x, c.y);
            ys.push_back(c.y);
        }
        LaserLine line;
        line.angle_deg = angle_deg;
        line.addPoints(std::move(pts), std::move(ys));
        laser_lines.emplace_back(std::move(line));
    }
    return laser_lines;
}

// ============================================================
// match5：基于光曲面重投影的贪心区间匹配
//
// 思路：
//   对每条左激光线 l，遍历每个光曲面 p，把 l 的点重投影到右图，
//   计算与各右激光线 r 的距离；若支持点足够则构建 IntervalMatch 候选。
//   采用多轮"唯一候选先锁定"策略，逐步扩大匹配集合。
//
// 新接口：直接接收 vector<LaserLine>，内部按 y_coords 遍历
// ============================================================
std::vector<IntervalMatch> LaserProcessor::match5(
    const std::vector<LaserLine>& laser_l,
    const std::vector<LaserLine>& laser_r,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r) {

    // --- 初始化相机参数 ---
    const auto calib  = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();

    double fx_l = calib.P[0].at<double>(0,0), fy_l = calib.P[0].at<double>(1,1);
    double cx_l = calib.P[0].at<double>(0,2), cy_l = calib.P[0].at<double>(1,2);
    double fx_r = calib.P[1].at<double>(0,0), fy_r = calib.P[1].at<double>(1,1);
    double cx_r = calib.P[1].at<double>(0,2), cy_r = calib.P[1].at<double>(1,2);
    double baseline = -calib.P[1].at<double>(0,3) / fx_r;

    int L = static_cast<int>(laser_l.size());
    int R = static_cast<int>(laser_r.size());
    const float S_thresh = 30.f;

    std::vector<IntervalMatch> final_matches;
    // 已锁定的 y 区间：locked_l[l] / locked_r[r]
    std::vector<std::vector<Interval>> locked_l(L), locked_r(R);

    // 合并重叠区间
    auto merge_intervals = [&](std::vector<Interval>& v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.y_start<b.y_start; });
        std::vector<Interval> m; Interval cur = v[0];
        for (int i = 1; i < (int)v.size(); ++i) {
            auto& n = v[i];
            if (n.y_start <= cur.y_end + EPS_) {
                cur.y_end = std::max(cur.y_end, n.y_end); cur.count += n.count;
            } else { m.push_back(cur); cur = n; }
        }
        m.push_back(cur); v.swap(m);
    };

    // 检查 y 是否在已锁定区间内
    auto isLocked = [&](const std::vector<Interval>& ivs, float y) {
        for (auto& iv : ivs)
            if (y >= iv.y_start - EPS_ && y <= iv.y_end + EPS_) return true;
        return false;
    };

    // === 多轮"唯一候选先锁定"循环 ===
    bool progress = true;
    while (progress) {
        progress = false;
        for (int l = 0; l < L; ++l) {
            const auto& line_l = laser_l[l];
            int repro_cnt_total = 0;
            std::vector<IntervalMatch> cands;

            for (int p_idx = 0; p_idx < (int)planes.size(); ++p_idx) {
                int repro_cnt = 0;
                const auto& coef = planes[p_idx].coefficients;
                // r_idx -> [(yr, dist)]
                std::map<int, std::vector<std::pair<float,float>>> support;

                // 遍历左线的每个点（用 y_coords + points vector）
                for (size_t pt_i = 0; pt_i < line_l.size(); ++pt_i) {
                    float y_f = line_l.y_coords[pt_i];
                    float x_f = line_l.points[pt_i].x;

                    if (isLocked(locked_l[l], y_f)) continue;

                    // 重投影：左像素 -> 3D -> 右像素
                    cv::Point3f ray(float((x_f-cx_l)/fx_l), float((y_f-cy_l)/fy_l), 1.f);
                    ray *= 1.f / cv::norm(ray);
                    auto ips = findIntersection({0,0,0}, ray, coef);
                    if (ips.empty()) continue;

                    cv::Point3f pt3;
                    bool ok = false;
                    for (auto& q : ips) if (q.z > 100 && q.z < 1500) { pt3=q; ok=true; break; }
                    if (!ok) continue;

                    cv::Point3f pr(pt3.x-float(baseline), pt3.y, pt3.z);
                    float xr = float(fx_r*pr.x/pr.z+cx_r);
                    float yr = alignToPrecision(float(fy_r*pr.y/pr.z+cy_r));
                    if (xr<0 || xr>=rectify_r.cols || yr<0 || yr>=rectify_r.rows) continue;
                    ++repro_cnt; ++repro_cnt_total;

                    // 与各右线匹配
                    for (int r = 0; r < R; ++r) {
                        const auto* rp = laser_r[r].findPoint(yr);
                        if (!rp) continue;
                        float d = std::hypot(rp->x - xr, rp->y - yr);
                        if (d > D_thresh_) continue;
                        if (isLocked(locked_r[r], yr)) continue;
                        support[r].emplace_back(yr, d);
                    }
                }
                if (support.empty()) continue;

                // 对每条右线构建支持区间
                for (auto& [r_idx, vec] : support) {
                    std::sort(vec.begin(), vec.end(),
                              [](auto& a, auto& b){ return a.first < b.first; });

                    // 拆分连续子段
                    std::vector<Interval> segs;
                    int s = 0;
                    for (int i = 1; i < (int)vec.size(); ++i) {
                        if (vec[i].first - vec[i-1].first > 2*precision_+EPS_) {
                            segs.push_back({alignToPrecision(vec[s].first),
                                            alignToPrecision(vec[i-1].first), i-s});
                            s = i;
                        }
                    }
                    segs.push_back({alignToPrecision(vec[s].first),
                                    alignToPrecision(vec.back().first),
                                    (int)vec.size()-s});

                    int total_cnt = 0;
                    for (auto& iv : segs) total_cnt += iv.count;
                    if (total_cnt < MIN_LEN_) continue;

                    // 计算评分
                    std::vector<std::pair<float,float>> allpd;
                    for (auto& pd : vec)
                        for (auto& iv : segs)
                            if (pd.first >= iv.y_start-EPS_ && pd.first <= iv.y_end+EPS_) {
                                allpd.push_back(pd); break;
                            }

                    float coverage = 0, std_dev = 0;
                    float score = computeEnhancedScore(allpd, repro_cnt, coverage, std_dev);
                    if (score <= S_thresh)
                        cands.push_back({l, p_idx, r_idx, segs, score, coverage, std_dev});
                }
            }
            if (cands.empty()) continue;

            // 判断是否可锁定
            bool lock = false;
            auto& m = cands[0];
            if (cands.size() == 1) {
                if (m.score <= 7.f  && m.coverage >= 0.65f) lock = true;
                else if (m.score <= 11.f && m.coverage >= 0.70f) lock = true;
                else if (m.score <= 14.5f&& m.coverage >= 0.85f) lock = true;
                else if (m.score <= 17.f && m.coverage >= 0.91f && m.std_dev < 5.f) lock = true;
            } else {
                std::sort(cands.begin(), cands.end(),
                          [](auto& a, auto& b){ return a.score < b.score; });
                auto& mm = cands[1];
                if (m.p_idx == mm.p_idx && std::fabs(mm.score-m.score) <= 1.5f
                    && std::fabs(m.std_dev-mm.std_dev) < 1.f) lock = true;
                else if (mm.score-m.score >= 10.f && m.score <= 13.f
                         && m.coverage >= 0.65f) lock = true;
                else if (std::fabs(mm.score-m.score) <= 3.f
                         && std::fabs(m.std_dev-mm.std_dev) < 1.f) {
                    if (mm.coverage > m.coverage) std::swap(m, mm);
                    if (m.coverage - mm.coverage > 0.6f) lock = true;
                }
            }
            if (lock) {
                final_matches.push_back(m);
                locked_l[m.l_idx].insert(locked_l[m.l_idx].end(),
                                          m.intervals.begin(), m.intervals.end());
                merge_intervals(locked_l[m.l_idx]);
                locked_r[m.r_idx].insert(locked_r[m.r_idx].end(),
                                          m.intervals.begin(), m.intervals.end());
                merge_intervals(locked_r[m.r_idx]);
                progress = true;
            }
        }
    }
    return final_matches;
}
