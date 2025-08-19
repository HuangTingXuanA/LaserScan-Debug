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

struct MatchKey {
    int l_laser_idx;
    int plane_idx;
    int r_laser_idx;

    bool operator<(const MatchKey& other) const {
        if (l_laser_idx != other.l_laser_idx) return l_laser_idx < other.l_laser_idx;
        if (plane_idx != other.plane_idx) return plane_idx < other.plane_idx;
        return r_laser_idx < other.r_laser_idx;
    }
};
struct MatchResult {
    float score_sum = 0;
    int count = 0;
};
struct MatchKeyHash {
    size_t operator()(const MatchKey& k) const {
        return (size_t(k.l_laser_idx) << 20) ^ 
               (size_t(k.plane_idx) << 10) ^
               size_t(k.r_laser_idx);
    }
};
struct MatchKeyEqual {
    bool operator()(const MatchKey& a, const MatchKey& b) const {
        return a.l_laser_idx == b.l_laser_idx &&
               a.plane_idx == b.plane_idx &&
               a.r_laser_idx == b.r_laser_idx;
    }
};
typedef std::unordered_map<MatchKey, MatchResult, MatchKeyHash, MatchKeyEqual> MatchMap;

struct Center {
    float x, y;
    cv::Vec2f dir;  // 局部方向，必须保证 dir[1] != 0
};

// 平面匹配结果结构
struct ReprojectionInfo {
    float x_left;   // 左图像点x坐标
    float y_left;   // 左图像点y坐标
    float x_right;  // 右图像重投影x坐标
    float y_right;  // 右图像重投影y坐标
    std::map<int, float> r_scores; // 右激光线索引 -> 匹配得分
    
    ReprojectionInfo(float x, float y, float rx, float ry) 
        : x_left(x), y_left(y), x_right(rx), y_right(ry) {}
};

struct ScoreAccumulator {
    float score_sum = 0;
    int count = 0;
};

struct PlaneMatchResult {
    int plane_idx = -1; // 平面索引
    int point_count = 0; // 有效点数量（重投影在图像内的点数）
    int best_r_idx = -1; // 最佳匹配的右激光线索引
    float avg_score = -FLT_MAX; // 平均距离得分
    float composite_score = -FLT_MAX;  // 平均得分 与 覆盖率惩罚得分 组合
    std::vector<ReprojectionInfo> reprojected_points; // 所有重投影点信息
    std::map<int, ScoreAccumulator> r_line_scores; // 右激光线索引 -> 分数累计
};

// 新增匹配策略相关数据结构
struct CandidateMatch {
    int plane_idx;
    int r_laser_idx;
    float avg_score;
    float score_gap; // 与次优候选的得分差距
};

struct UnmatchedInfo {
    int l_laser_idx = -1;
    std::vector<CandidateMatch> candidates;
};

struct Interval { float y_start, y_end; int count;};
struct IntervalMatch {
    int l_idx, p_idx, r_idx;
    std::vector<Interval> intervals;
    float score;
    float coverage;
    float std_dev;
};
// 浮点安全区间比较
struct IntervalCompare {
    bool operator()(const Interval& a, const Interval& b) const {
        if (a.y_start < b.y_start - 1e-4) return true;
        if (a.y_start > b.y_start + 1e-4) return false;
        return a.y_end < b.y_end - 1e-4;
    }
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
    std::vector<float> sampleProfileAlongNormal(
        const cv::Mat& img, 
        float x, float y, 
        const cv::Vec2f& n, 
        float sample_step = 0.1f,
        float search_range = 2.5f
    );
    float solveSubpixelCenter(
        const std::vector<float>& profile,
        const cv::Vec2f& n,
        const cv::Matx22f& H,
        float sample_step
    );

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
    std::vector<LaserLine> extractLine3(    // steger算法抛物线拟合
        const cv::Mat& rectify_img,
        std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
        int img_idx);

    double evaluateQuadSurf(const cv::Mat &Coeff6x1, const cv::Point3f &p);
    double evaluateQuadSurf(const cv::Mat& Coeff6x1, const std::vector<cv::Point3f>& points);
    std::vector<cv::Point3f> findIntersection(const cv::Point3f &point, const cv::Point3f &normal,
                                          const cv::Mat &Coeff6x1);
    
    float computeCompScore3(float avg_dist, float coverage);
    std::vector<std::tuple<int, int, int>> match3(
        const std::vector<std::map<float, float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l, const cv::Mat& rectify_r);


    float computeCompScore4(float avgDist, float coverage, float wD = 0.6f, float wC = 0.4f);
    float computeEnhancedScore(const std::vector<std::pair<float, float>>& distance_pairs,
        int left_line_total_points, float& coverage, float& std_dev);

    std::vector<std::tuple<int,int,int>> match4(
        const std::vector<std::map<float,float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);
    
    std::vector<IntervalMatch> match5(
        const std::vector<std::map<float,float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);

    std::vector<IntervalMatch> match6(
        const std::vector<std::map<float,float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r);
    

    void mergeIntervals(std::vector<Interval>& intervals, float prec) const;
    bool isPointLocked(float y, const std::vector<Interval>& intervals) const;

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
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r);

    void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg);
    void Two_PassNew(const cv::Mat &img, cv::Mat &labImg);
    cv::Scalar GetRandomColor();

private:
    const float roi_scale_ = 1.05f;
    const float precision = 0.5f;
    const float EPS = 0.01;

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
        return std::round(y / precision) * precision;
    }

    static inline float clampFloat(float v, float a, float b) {
        return std::max(a, std::min(b, v));
    }

    // 在给定 y 行上估计左右边缘处的梯度（沿 x 方向的简单差分）
    inline float estimateGradientAtX(const cv::Mat& img, float x, float y, int halfWindow = 1) {
        // 使用中心差分： (I(x+eps)-I(x-eps)) / (2*eps) 这里 eps=1
        float left = interpolateChannel(img, x - 1.0f, y);
        float right = interpolateChannel(img, x + 1.0f, y);
        return (right - left) * 0.5f; // 近似导数
    }
};
