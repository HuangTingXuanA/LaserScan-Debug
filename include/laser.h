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

struct Center {
    float x, y;
    cv::Vec2f dir;  // 局部方向，必须保证 dir[1] != 0
};

struct Interval { float y_start, y_end; int count;};
struct IntervalMatch {
    int l_idx, p_idx, r_idx;
    std::vector<Interval> intervals;
    float score;
    float coverage;
    float std_dev;
};

class LaserProcessor {
public:
    LaserProcessor() = default;

    float interpolateChannel(const cv::Mat& img, float x, float y);
    double interpolateChannel(const cv::Mat& img, double x, double y);
    std::tuple<cv::Mat, cv::Mat, int> computeGaussianDerivatives(float sigma, float angle_rad, bool h_is_long_edge);
    cv::Mat computeGaussianDerivatives(float sigma, float epsilon = 1e-4);
    cv::Mat getSafeROI(const cv::Mat& img, int x, int y, int size);
    cv::Point2f computeStegerCenter(
        const cv::Mat& img, 
        int x, int y, 
        float sigma
    );
    float findSymmetricCenter(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range);
    float findSymmetricCenter2(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range, size_t is_right);
    float findSymmetricCenter3(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range);    // 抛物线拟合
    float findSymmetricCenter4(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range);    // 质心法
    float computeRowCentroid(const cv::Mat& img, int y, float xL, float xR);

    int convert_to_odd_number(float num);

    std::vector<cv::Point2f> processCenters(const std::map<float, float>& orign_centers);
    std::pair<cv::Point2f, cv::Point2f> getAxisEndpoints(const cv::RotatedRect& rect);;
    std::vector<LaserLine> extractLine(     // 含ROI区域的抛物线拟合
        const std::vector<cv::RotatedRect>& rois,
        const cv::Mat& rectify_img, const cv::Mat& label_img,
        int img_idx);
    std::vector<LaserLine> extractLine2(    // 抛物线拟合
        const cv::Mat& rectify_img,
        std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
        int img_idx);
    std::vector<LaserLine2> extractLine3(    // steger算法抛物线拟合
        const cv::Mat& rectify_img,
        std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
        int img_idx);

    double evaluateQuadSurf(const cv::Mat &Coeff6x1, const cv::Point3f &p);
    double evaluateQuadSurf(const cv::Mat& Coeff6x1, const std::vector<cv::Point3f>& points);
    std::vector<cv::Point3f> findIntersection(const cv::Point3f &point, const cv::Point3f &normal,
                                          const cv::Mat &Coeff6x1);


    float computeCompScore(float avgDist, float coverage, float wD = 0.6f, float wC = 0.4f);
    float computeEnhancedScore(const std::vector<std::pair<float, float>>& distance_pairs,
        int left_line_total_points, float& coverage, float& std_dev);
    float computeEnhancedScoreV2(
        const std::vector<std::pair<float, float>>& distance_pairs,
        int left_point_count, int right_point_count,
        float& coverage, float& std_dev);

    std::vector<std::tuple<int,int,int>> match4( // 重投影 Debug 查看
        const std::vector<std::map<float,float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);
    
    std::vector<IntervalMatch> match5(  // 局部贪心 - 局部最优
        const std::vector<std::map<float,float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);

    std::vector<IntervalMatch> match6(  // 全局贪心 - 全局最优
        const std::vector<std::map<float,float>>& sample_points, // 左激光线: y -> x
        const std::vector<LaserLine>& laser_r                    // 右激光线
    );
    
    std::vector<IntervalMatch> match7(  // 匈牙利算法 - 全局最优
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r
    );

    std::vector<IntervalMatch> match8(
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r
    );

    void removeMatchedIntervals(LaserLine& line, const std::vector<Interval>& matched_intervals);
    std::vector<IntervalMatch> generateCandidatesForRemaining(
        const std::vector<LaserLine>& left_lines,
        const std::vector<LaserLine>& right_lines,
        const std::vector<QuadSurface>& surfaces,
        const CalibrationResult& calib);
    
    // 区间冲突检测：检查两个区间集合是否有重叠
    bool hasIntervalConflict(const std::vector<Interval>& intervals1, const std::vector<Interval>& intervals2);
    
    // 全局贪心匹配：支持一对多匹配（区间不重叠）
    std::vector<IntervalMatch> globalGreedyMatching(const std::vector<IntervalMatch>& candidates);
    std::vector<IntervalMatch> match9(
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r
    );

    std::vector<cv::Point3f> generateCloudPoints(
        const std::vector<std::tuple<int, int, int>>& laser_match,
        const std::vector<LaserLine> laser_l,
        const std::vector<LaserLine> laser_r);
    
    std::vector<cv::Point3f> generateCloudPoints2(
        const std::vector<IntervalMatch>& matches,
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r);

    std::vector<cv::Point3f> generateCloudPoints3(
        const std::vector<IntervalMatch>& matches,
        const std::vector<LaserLine2>& laser_l,
        const std::vector<LaserLine2>& laser_r);

    void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg);
    void Two_PassNew(const cv::Mat &img, cv::Mat &labImg);
    cv::Scalar GetRandomColor();

private:
    const float roi_scale_ = 1.05f;
    const float D_thresh_ = 10.0f;
    const float S_thresh_ = 3.3f;
    const int MIN_LEN_ = 80;
    const float EPS_ = 1e-4f;
    const float precision_ = 0.5f;
    const float N_HIGH = 1100.0f;
    const float C_MIN = 0.8f;
    const float MARGIN_RATIO = 0.85f;

    static bool isInteger(float v) {
        return std::fabs(v - std::round(v)) < 1e-6f;
    }

    static void compute_angles(const std::vector<cv::Point3f> &P3, std::vector<double> &angles){
        int n = (int)P3.size(); angles.assign(n, 0.0);
        if (n<3) return;
        for (int i=1;i+1<n;++i){
            cv::Point3f v1 = P3[i] - P3[i-1];
            cv::Point3f v2 = P3[i+1] - P3[i];
            double n1 = cv::norm(v1), n2 = cv::norm(v2);
            if (n1<1e-9 || n2<1e-9) { angles[i]=0; continue; }
            double dot = v1.dot(v2) / (n1*n2);
            dot = std::max(-1.0, std::min(1.0, dot));
            angles[i] = std::acos(dot);
        }
    }

    inline float alignToPrecision(float y) {
        return std::round(y / precision_) * precision_;
    }

    static inline float clampFloat(float v, float a, float b) {
        return std::max(a, std::min(b, v));
    }

    inline float gradientAlongNormal(const cv::Mat& img, float px, float py, cv::Point2f normal, float eps) {
        float nx = normal.x, ny = normal.y;
        float nlen = std::sqrt(nx*nx + ny*ny);
        if (nlen == 0.0f) return 0.0f;
        nx /= nlen; ny /= nlen;
        float v1 = interpolateChannel(img, px + nx * eps, py + ny * eps);
        float v0 = interpolateChannel(img, px - nx * eps, py - ny * eps);
        return (v1 - v0) * 0.5f / eps; // per-pixel derivative
    }
};
