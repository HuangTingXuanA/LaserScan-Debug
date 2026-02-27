#pragma once
#include "type.h"
#include "configer.h"
#include <numeric>
#include <fmt/core.h>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>

// ============================================================
// 左图采样点（预计算射线方向）
// ============================================================
struct LeftPoint {
    float x, y;
    cv::Point3f ray; // 归一化射线方向

    LeftPoint(float x_, float y_, cv::Point3f ray_) : x(x_), y(y_), ray(ray_) {}
};

// ============================================================
// 切片（LaserLine 在水平扫描条带内的片段）
// ============================================================
struct Slice {
    int id;             // 全局唯一 ID（= all_slices 数组下标）
    int laser_idx;      // 原始激光线连通域索引（laser_l 或 laser_r 的下标）
    int band_idx;       // 属于第几扫描条带

    cv::Point2f          center_pt; // 切片重心
    std::vector<LeftPoint> pts;     // 切片内所有像素点（含预计算射线）
};

// ============================================================
// 扫描条带
// ============================================================
struct Band {
    int idx;

    // 切片所有权（unique_ptr 管理生命周期）
    std::vector<std::unique_ptr<Slice>> storage;
    // 访问视图（裸指针，排序/快速访问用，不负责释放）
    std::vector<Slice*> slices;
    // 辅助查找：laser_idx -> Slice*
    std::unordered_map<int, Slice*> laser2slice;

    void addSlice(std::unique_ptr<Slice> s) {
        slices.push_back(s.get());
        storage.push_back(std::move(s));
    }
    void buildLaserToSlice() {
        laser2slice.clear();
        for (auto* s : slices) laser2slice[s->laser_idx] = s;
    }
};

// ============================================================
// 评分信息
// ============================================================
struct ScoreInfo {
    float score;       // 最终加权得分
    float dis_mode;    // 距离众数
    float dis_mean;    // 距离均值
    float dis_stddev;  // 距离标准差
    float norm_census; // 归一化 Census 代价
};

// ============================================================
// DP 状态节点
// ============================================================
struct DpNode {
    float cost;
    int prev_l, prev_r, prev_p;
};

// ============================================================
// Cost Matrix 单元
// ============================================================
struct CostEntry {
    ScoreInfo info;
    int r_idx;
};

// ============================================================
// RLE 分析（光平面连续段）
// ============================================================
struct Run {
    int p_idx;      // 光平面 ID
    int start_idx;  // 在 group 中的起始下标
    int len;        // 长度
};

// ============================================================
// 最终匹配结果
// ============================================================
struct MatchResult {
    int band_id;    // -> Band
    int l_slice_id; // -> 左侧 Slice
    int r_slice_id; // -> 右侧 Slice
    int l_idx;      // -> laser_l 激光线索引
    int p_idx;      // -> QuadSurface 光平面索引
    int r_idx;      // -> laser_r 激光线索引（-1 表示无匹配）

    ScoreInfo info; // 详细评分
};

// ============================================================
// 匹配统计（用于计算匹配完成率）
// ============================================================
struct MatchStats {
    int total_l_slices   = 0; // 参与匹配的左线切片总数
    int matched_l_slices = 0; // 已匹配左切片数
    int total_r_slices   = 0; // 参与匹配的右线切片总数
    int matched_r_slices = 0; // 已匹配右切片数
};

// ============================================================
// 匹配处理器
// ============================================================
class MatchProcessor {
public:
    std::vector<MatchResult> match(
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r,
        MatchStats& stats           // [output] 匹配统计，用于计算完成率
    );

    void filterMatches(std::vector<MatchResult>& matches,
                       const std::vector<Band>& bands_l,
                       const std::vector<Band>& bands_r);

    std::vector<cv::Point3f> findIntersection(
        const cv::Point3f& point,
        const cv::Point3f& normal,
        const cv::Mat& Coeff6x1);

    int computeCensus(const cv::Mat& img1, const cv::Point2f& pt1,
                      const cv::Mat& img2, const cv::Point2f& pt2,
                      int window_size = 5);

    // 生成条带切片（public，供外部生成可视化图调用）
    std::vector<Band> generateBands(const std::vector<LaserLine>& laser,
                                    const cv::Mat& P,
                                    int img_rows);

    // 可视化切片：将两侧条带绘制在拼接图上，按激光线着色，标注切片 ID
    // 若有匹配结果，则将匹配的左右对使用同一颜色
    cv::Mat visualizeSlices(
        const std::vector<LaserLine>& laser_l,
        const std::vector<LaserLine>& laser_r,
        const cv::Mat& rectify_l,
        const cv::Mat& rectify_r,
        const std::vector<MatchResult>& results);

private:
    const float precision_    = 0.5f;
    const float D_thresh_     = 10.0f;
    const float S_thresh_     = 4.0f;
    const float EPS_          = 1e-4f;
    const int   MIN_LEN_      = 160;
    const int   SLICE_HEIGHT_ = 15;
    const float COST_INVALID_ = 100.0f;

    inline float alignToPrecision(float y) {
        return std::round(y / precision_) * precision_;
    }

    ScoreInfo computeCost(const Slice* slice_l, const Slice* slice_r,
                          const cv::Mat& coef,
                          const cv::Mat& rectify_l, const cv::Mat& rectify_r);

    inline cv::Scalar getBandColor(int band_idx);
};