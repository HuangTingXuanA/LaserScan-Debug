#pragma once
#include "type.h"
#include "configer.h"
#include <numeric>
#include <fmt/core.h>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>

// ============ 核心数据结构 ============

// 左图采样点（预计算射线方向）
struct LeftPoint {
    float y, x;
    cv::Point3f ray;  // 归一化射线方向
    
    LeftPoint(float y_, float x_, cv::Point3f ray_) : y(y_), x(x_), ray(ray_) {}
};

// 匹配候选（并行阶段产生）
struct MatchCandidate {
    int l_idx, r_idx, p_idx;  // 左线、右线、光平面索引
    float y;                   // 匹配点的y坐标
    float distance;            // 匹配距离
    
    MatchCandidate(int l, int r, int p, float y_, float d) 
        : l_idx(l), r_idx(r), p_idx(p), y(y_), distance(d) {}
};

// 匹配三元组的键（用于分组）
struct MatchKey {
    int l_idx, r_idx, p_idx;
    
    bool operator<(const MatchKey &other) const {
        if (l_idx != other.l_idx) return l_idx < other.l_idx;
        if (r_idx != other.r_idx) return r_idx < other.r_idx;
        return p_idx < other.p_idx;
    }
    
    std::string toString() const {
        return "L" + std::to_string(l_idx) + "-R" + std::to_string(r_idx) + "-P" + std::to_string(p_idx);
    }
};

// 区间段（贪心算法的基本单元）
struct IntervalSegment {
    int line_idx;              // 所属激光线索引
    int pair_idx;              // 配对线索引（左线视角存右线索引，右线视角存左线索引）
    int p_idx;                 // 光平面索引
    std::vector<float> y_coords;  // 区间包含的y坐标（有序）
    std::vector<std::pair<float, float>> match_points;  // (y, distance)
    float avg_distance;        // 平均匹配距离
    
    // 辅助方法
    float y_start() const { return y_coords.empty() ? 0.0f : y_coords.front(); }
    float y_end() const { return y_coords.empty() ? 0.0f : y_coords.back(); }
    int size() const { return static_cast<int>(y_coords.size()); }
};

// 激光线区域（分割后的连续段）
struct LaserRegion {
    int line_idx;              // 所属激光线索引
    int region_id;             // 全局区域ID（按x坐标排序后分配）
    int pair_idx;              // 配对线索引
    int p_idx;                 // 光平面索引
    std::vector<float> y_coords;  // 区间y坐标（有序）
    std::vector<std::pair<float, float>> points;  // (y, x) 实际像素坐标
    float center_x;            // 区间中心点x坐标（用于排序）
    float avg_distance;        // 平均匹配距离
    
    // 辅助方法
    float y_start() const { return y_coords.empty() ? 0.0f : y_coords.front(); }
    float y_end() const { return y_coords.empty() ? 0.0f : y_coords.back(); }
    int size() const { return static_cast<int>(y_coords.size()); }
};

// 评分信息
struct ScoreInfo {
    float coverage;
    float dis_mode;
    float dis_mean;
    float dis_stddev;
    float remain_penalty;
    float score;
};

// 最终匹配结果
struct RegionMatch {
    int l_idx, r_idx, p_idx;        // 左右线索引、光平面索引
    int l_region_id, r_region_id;   // 左右区间ID
    float y_start, y_end;           // 匹配区间范围
    int match_count;                // 匹配点数
    ScoreInfo score_info;           // 评分信息
    
    std::string key() const {
        return std::string("L") + std::to_string(l_idx) + "(Reg" + std::to_string(l_region_id) + 
               ")-R" + std::to_string(r_idx) + "(Reg" + std::to_string(r_region_id) + 
               ")-P" + std::to_string(p_idx);
    }
    
    // 兼容性访问器
    float score() const { return score_info.score; }
    float std_dev() const { return score_info.dis_stddev; }
};

class MatchProcessor {
public:
    std::vector<RegionMatch> match(
        const std::vector<LaserLine> &laser_l, const std::vector<LaserLine> &laser_r,
        const cv::Mat &rectify_l, const cv::Mat &rectify_r
    );
    ScoreInfo computePointScore(
        const std::vector<std::pair<float, float>> &distance_pairs,
        int pt_cnt_l, int pt_cnt_r
    );
    std::vector<cv::Point3f> findIntersection(const cv::Point3f &point, const cv::Point3f &normal, const cv::Mat &Coeff6x1);
    
private:
    const float precision_ = 0.5f;
    const float D_thresh_ = 10.0f;
    const float S_thresh_ = 3.5f;
    const float EPS_ = 1e-4f;
    const int MIN_LEN_ = 160;
    
    // 辅助函数：将y坐标对齐到精度网格
    inline float alignToPrecision(float y) {
        return std::round(y / precision_) * precision_;
    }
};