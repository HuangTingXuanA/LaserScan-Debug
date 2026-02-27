#pragma once
#include "type.h"
#include "configer.h"
#include <unordered_map>
#include <map>
#include <numeric>
#include <opencv2/freetype.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>

// ============================================================
// 重投影区间（match5 内部结构）
// ============================================================
struct Interval { float y_start, y_end; int count; };

struct IntervalMatch {
    int   l_idx, p_idx, r_idx;
    std::vector<Interval> intervals;
    float score;
    float coverage;
    float std_dev;
};

// ============================================================
// 激光线处理器
// ============================================================
class LaserProcessor {
public:
    LaserProcessor() = default;

    // ---------- 图像工具 ----------
    float  interpolateChannel(const cv::Mat& img, float x, float y);
    double interpolateChannel(const cv::Mat& img, double x, double y);

    cv::Mat computeGaussianDerivatives(float sigma, float epsilon = 1e-4);

    float findSymmetricCenter4(const cv::Mat& img, float x, float y,
                               cv::Vec2f dir, float search_range); // 质心法

    float computeRowCentroid(const cv::Mat& img, int y, float xL, float xR);
    int   convert_to_odd_number(float num);

    // ---------- 激光线提取（返回有序激光线） ----------
    std::vector<cv::Point2f> processCenters(const std::map<float, float>& orign_centers);
    std::pair<cv::Point2f, cv::Point2f> getAxisEndpoints(const cv::RotatedRect& rect);

    std::vector<LaserLine> extractLine2(
        const cv::Mat& rectify_img,
        std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
        int img_idx);

    // ---------- 曲面求交 ----------
    double evaluateQuadSurf(const cv::Mat& Coeff6x1, const cv::Point3f& p);
    double evaluateQuadSurf(const cv::Mat& Coeff6x1, const std::vector<cv::Point3f>& points);
    std::vector<cv::Point3f> findIntersection(const cv::Point3f& point,
                                              const cv::Point3f& normal,
                                              const cv::Mat&     Coeff6x1);

    // ---------- 评分函数 ----------
    float computeEnhancedScore(const std::vector<std::pair<float, float>>& distance_pairs,
                               int left_line_total_points,
                               float& coverage, float& std_dev);

    // ---------- match5：局部贪心重投影匹配 ----------
    // 参数：laser_l 左激光线，laser_r 右激光线，rectify_l/r 极线校正图
    // 返回：每条左线 -> 最佳右线 的区间匹配结果
    std::vector<IntervalMatch> match5(
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);

private:
    const float D_thresh_ = 30.0f;
    const float S_thresh_ = 3.5f;
    const int   MIN_LEN_  = 160;
    const float EPS_      = 1e-4f;
    const float precision_ = 0.5f;

    static bool isInteger(float v) {
        return std::fabs(v - std::round(v)) < 1e-6f;
    }

    inline float alignToPrecision(float y) {
        return std::round(y / precision_) * precision_;
    }

    static inline float clampFloat(float v, float a, float b) {
        return std::max(a, std::min(b, v));
    }

    inline float gradientAlongNormal(const cv::Mat& img,
                                     float px, float py,
                                     cv::Point2f normal, float eps) {
        float nx = normal.x, ny = normal.y;
        float nlen = std::sqrt(nx*nx + ny*ny);
        if (nlen == 0.0f) return 0.0f;
        nx /= nlen; ny /= nlen;
        float v1 = interpolateChannel(img, px + nx*eps, py + ny*eps);
        float v0 = interpolateChannel(img, px - nx*eps, py - ny*eps);
        return (v1 - v0) * 0.5f / eps;
    }
};
