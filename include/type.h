#pragma once
#include <omp.h>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <tuple>

// ============================================================
// 双目标定结果
// ============================================================
struct CalibrationResult {
    // 0-左相机，1-右相机
    std::array<cv::Mat, 2> camera_matrix;      // 单目相机内参矩阵
    std::array<cv::Mat, 2> dist_coeffs;        // 单目相机畸变系数
    std::array<std::vector<cv::Mat>, 2> rvecs; // 左/右相机的旋转向量
    std::array<std::vector<cv::Mat>, 2> tvecs; // 左/右相机的平移向量
    cv::Mat R;                  // 双目旋转矩阵
    cv::Mat T;                  // 双目平移向量
    cv::Mat E;                  // 本质矩阵
    cv::Mat F;                  // 基础矩阵
    cv::Mat rectify_matrix[2];  // 极线校正后的3x3内参矩阵
    cv::Mat P[2];               // 极线校正后的3x4投影矩阵
    cv::Mat rectify_R[2];       // 极线校正旋转矩阵
    cv::Mat Q;                  // 视差图到深度图映射矩阵
    cv::Rect valid_roi[2];      // 极线校正后有效区域
    cv::Rect common_roi;        // 双目公共有效区域
    std::vector<std::array<cv::Mat, 2>> remap; // 重映射表 remap[cam][0]=map1, [1]=map2
    double baseline;            // 双目基线距离（毫米）
    double l_rmse;              // 左相机标定精度
    double r_rmse;              // 右相机标定精度
    double rmse;                // 双目标定精度
    double l_reproj_avge;       // 左相机重投影平均误差（像素）
    double r_reproj_avge;       // 右相机重投影平均误差（像素）
};

// ============================================================
// 激光线中心点
// ============================================================
struct LaserPoint {
    float x, y;

    LaserPoint(float x_, float y_) : x(x_), y(y_) {}
    bool operator==(const LaserPoint& other) const {
        return x == other.x && y == other.y;
    }
};

// ============================================================
// 激光线（有序点列，支持二分查找）
// ============================================================
struct LaserLine {
    std::vector<LaserPoint> points;  // 激光点数据（与 y_coords 一一对应）
    std::vector<float>      y_coords; // y 坐标，保持升序以支持二分查找
    float                   angle_deg = 0.0f; // 激光线大概角度（度）

    // 预留容量以减少重分配
    void reserve(size_t capacity) {
        points.reserve(capacity);
        y_coords.reserve(capacity);
    }

    // 单点有序插入
    void addPoint(const LaserPoint& point, float y) {
        auto it = std::lower_bound(y_coords.begin(), y_coords.end(), y);
        size_t pos = std::distance(y_coords.begin(), it);
        points.insert(points.begin() + pos, point);
        y_coords.insert(it, y);
    }

    // 批量添加（直接移动，调用方需保证外部已排序）
    void addPoints(std::vector<LaserPoint>&& newPoints, std::vector<float>&& coords) {
        if (newPoints.size() != coords.size())
            throw std::invalid_argument("Points and coordinates size mismatch");
        if (points.empty()) {
            points   = std::move(newPoints);
            y_coords = std::move(coords);
            return;
        }
        points.insert(points.end(),
                      std::make_move_iterator(newPoints.begin()),
                      std::make_move_iterator(newPoints.end()));
        y_coords.insert(y_coords.end(),
                        std::make_move_iterator(coords.begin()),
                        std::make_move_iterator(coords.end()));
    }

    // 二分查找最近点（容差 eps）
    const LaserPoint* findPoint(float y, float eps = 1e-4f) const {
        if (y_coords.empty()) return nullptr;
        auto it = std::lower_bound(y_coords.begin(), y_coords.end(), y);

        size_t idx   = 0;
        float min_d  = std::numeric_limits<float>::max();

        if (it != y_coords.end()) {
            float d = std::fabs(*it - y);
            if (d < min_d) { min_d = d; idx = std::distance(y_coords.begin(), it); }
        }
        if (it != y_coords.begin()) {
            auto prev = std::prev(it);
            float d = std::fabs(*prev - y);
            if (d < min_d) { min_d = d; idx = std::distance(y_coords.begin(), prev); }
        }
        return (min_d < eps) ? &points[idx] : nullptr;
    }

    size_t size()  const { return points.size(); }
    bool   empty() const { return points.empty(); }
    void   clear()       { points.clear(); y_coords.clear(); }
};

// ============================================================
// 激光曲面标定结果（二次曲面）
// ============================================================
struct QuadSurface {
    cv::Mat coefficients;  // 6x1 系数矩阵 [a, b, c, d, e, f]
    float   rmse = 0.0f;   // 曲面拟合均方根误差
};