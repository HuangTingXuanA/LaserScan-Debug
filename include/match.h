#pragma once
#include "type.h"
#include "configer.h"
#include <numeric>
#include <fmt/core.h>
#include <fstream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>

// 左图采样点（预计算射线方向）
struct LeftPoint {
    float x;
    float y;
    cv::Point3f ray;  // 归一化射线方向
    
    LeftPoint(float x_, float y_, cv::Point3f ray_) : x(x_), y(y_), ray(ray_) {}
};

// 切片
struct Slice {
    int id;                 // 全局唯一ID (= all_slices数组下标)
    int laser_idx;          // 原始激光线连通域索引 (laser_l或laser_r的下标)
    int band_idx;           // 属于第几行扫描带 (用于调试分层)
    
    cv::Point2f center_pt;  // 切片重心
    std::vector<LeftPoint> pts; // 切片内所有像素点坐标(x,y)
};

// 扫描条带
struct Band {
    int idx;
    
    // 1. 所有权存储：使用 unique_ptr 自动管理 Slice 内存生命周期
    std::vector<std::unique_ptr<Slice>> storage; 
    
    // 2. 访问视图：存储裸指针，用于排序和快速访问
    // 注意：不要手动 delete 这里的指针，它们由 slices_storage 管理
    std::vector<Slice*> slices; 

    // 3.辅助查找表: laser_idx -> Slice*
    std::unordered_map<int, Slice*> laser2slice; 
    
    // 辅助函数：添加切片
    void addSlice(std::unique_ptr<Slice> s) {
        // 先保存裸指针用于访问
        slices.push_back(s.get());
        // 再转移所有权给 storage
        storage.push_back(std::move(s));
    }

    void buildLaserToSlice() {
        laser2slice.clear();
        for (auto* s : slices) {
            laser2slice[s->laser_idx] = s;
        }
    }
};

// 评分信息
struct ScoreInfo {
    float score;          // 最终加权得分
    float dis_mode;       // 距离众数
    float dis_mean;       // 距离均值
    float dis_stddev;     // 距离标准差
    float norm_census;    // 归一化 Census 代价
};

// DP 状态节点
struct DpNode {
    float cost;
    int prev_l, prev_r, prev_p;
};

// 存储 Cost Matrix 的单元
struct CostEntry {
    ScoreInfo info;
    int r_idx;
};

// RLE分析
struct Run {
    int p_idx;      // 光平面 ID
    int start_idx;  // 在 group 中的起始下标
    int len;        // 长度
};

// 最终匹配结果
struct MatchResult {
    int band_id;     // -> 索引到 Band
    int l_slice_id;  // -> 索引到 左侧 Slice
    int r_slice_id;  // -> 索引到 右侧 Slice
    int l_idx;       // -> 索引到 laser_l (冗余但方便)
    int p_idx;       // -> 索引到 Surfaces (光平面)
    int r_idx;       // -> 索引到 laser_r (匹配到的最佳右线，若无则为-1)
    
    ScoreInfo info;  // 详细评分
};

class MatchProcessor {
public:
    std::vector<MatchResult> match(
        const std::vector<LaserLine> &laser_l, const std::vector<LaserLine> &laser_r,
        const cv::Mat &rectify_l, const cv::Mat &rectify_r
    );
    void filterMatches(std::vector<MatchResult>& matches, const std::vector<Band>& bands_l, const std::vector<Band>& bands_r);
    std::vector<cv::Point3f> findIntersection(const cv::Point3f &point, const cv::Point3f &normal, const cv::Mat &Coeff6x1);
    int computeCensus(const cv::Mat& img1, const cv::Point2f& pt1,
                         const cv::Mat& img2, const cv::Point2f& pt2, int window_size = 5);

private:
    const float precision_ = 0.5f;
    const float D_thresh_ = 10.0f;
    const float S_thresh_ = 4.0f;
    const float EPS_ = 1e-4f;
    const int MIN_LEN_ = 160;
    const int SLICE_HEIGHT_ = 30;
    const float COST_INVALID_ = 100.0f;  // 提高无效代价阈值
    
    // 辅助函数：将y坐标对齐到精度网格
    inline float alignToPrecision(float y) {
        return std::round(y / precision_) * precision_;
    }

    std::vector<Band> generateBands(const std::vector<LaserLine>& laser, const cv::Mat& P, int img_rows);
    ScoreInfo computeCost(const Slice* slice_l, const Slice* slice_r, const cv::Mat& coef,
                const cv::Mat& rectify_l, const cv::Mat& rectify_r);
    
    inline cv::Scalar getBandColor(int band_idx);
};