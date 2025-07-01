#pragma once
#include "type.h"
#include "configer.h"
#include <unordered_map>
#include <map>
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
    int plane_idx; // 平面索引
    int point_count = 0; // 有效点数量（重投影在图像内的点数）
    int best_r_idx = -1; // 最佳匹配的右激光线索引
    float avg_score = -FLT_MAX; // 最佳匹配的平均得分
    std::vector<ReprojectionInfo> reprojected_points; // 所有重投影点信息
    std::map<int, ScoreAccumulator> r_line_scores; // 右激光线索引 -> 分数累计
};


class LaserProcessor {
public:
    LaserProcessor() = default;

    float interpolateChannel(const cv::Mat& img, float x, float y);
    std::tuple<cv::Mat, cv::Mat, int> computeGaussianDerivatives(float sigma, float angle_rad, bool h_is_long_edge);
    float findSymmetricCenter(const cv::Mat& img, float x, float y, cv::Vec2f dir, float search_range);
    int convert_to_odd_number(float num);

    std::vector<cv::Point2f> processCenters(const std::map<float, float>& orign_centers);
    std::pair<cv::Point2f, cv::Point2f> getAxisEndpoints(const cv::RotatedRect& rect);
    std::vector<LaserLine> extractLine(const std::vector<cv::RotatedRect>& rois, const cv::Mat& rectify_img);

    void match(
        const std::vector<std::map<float, float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l, const cv::Mat& rectify_r);
    

    double evaluateQuadSurf(const cv::Mat &Coeff6x1, const cv::Point3f &p);
    double evaluateQuadSurf(const cv::Mat& Coeff6x1, const std::vector<cv::Point3f>& points);
    std::vector<cv::Point3f> findIntersection(const cv::Point3f &point, const cv::Point3f &normal,
                                          const cv::Mat &Coeff6x1);
    void match2(
        const std::vector<std::map<float, float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l, const cv::Mat& rectify_r);
    
    void match3(
        const std::vector<std::map<float, float>>& sample_points,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l, const cv::Mat& rectify_r);
    cv::Vec2f computePrincipalDirection(const std::vector<cv::Point2f>& pts);
    float orientationScore(const cv::Vec2f& v1, const cv::Vec2f& v2);
    float dtwScore(const std::vector<float>& seq1, const std::vector<float>& seq2);

    void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg);
    void Two_PassNew(const cv::Mat &img, cv::Mat &labImg);
    cv::Scalar GetRandomColor();

private:
    float roi_scale_ = 1.05f;

    static bool isInteger(float v) {
        return std::fabs(v - std::round(v)) < 1e-6f;
    }
};
