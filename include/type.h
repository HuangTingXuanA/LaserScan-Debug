#pragma once
#include <omp.h>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <tuple>

// 双目标定将结果
struct CalibrationResult {
    // 0-左相机，1-右相机
    std::array<cv::Mat, 2> camera_matrix;      // 单目相机内参矩阵
    std::array<cv::Mat, 2> dist_coeffs;        // 单目相机畸变系数
    std::array<std::vector<cv::Mat>, 2> rvecs; // 左/右相机的旋转向量（每个图像一个）
    std::array<std::vector<cv::Mat>, 2> tvecs; // 左/右相机的平移向量（每个图像一个）
    cv::Mat R;                  // 双目旋转矩阵
    cv::Mat T;                  // 双目平移矩阵
    cv::Mat E;                  // 双目本质矩阵
    cv::Mat F;                  // 双目基础矩阵
    cv::Mat rectify_matrix[2];  // 极线校正后的3*3内参矩阵
    cv::Mat P[2];               // 极线校正后的3*4投影矩阵，P2 的基线非零（单位：毫米）
    cv::Mat rectify_R[2];
    cv::Mat Q;                  // 视差图到深度图的映射矩阵
    cv::Rect valid_roi[2];      // 极线校正后的有效区域
    cv::Rect common_roi;        // 双目图像裁剪后的公共有效区域
    std::vector<std::array<cv::Mat, 2>> remap;
    double baseline;            // 双目基线距离（单位：毫米）
    double l_rmse;                // 左侧相机标定精度
    double r_rmse;                // 右侧相机标定精度
    double rmse;                  // 双目标定精度
    double l_reproj_avge;        // 左相机重投影平均误差
    double r_reproj_avge;        // 右相机重投影平均误差
};

// 激光线中心点
struct LaserPoint {
    float x;
    float y;

    LaserPoint() : x(0), y(0) {}
    LaserPoint(float x_, float y_) : x(x_), y(y_) {}
    // 相等性判断
    friend bool operator==(const LaserPoint& a, const LaserPoint& b) {
        return a.x == b.x && a.y == b.y; // 仅比较x,y坐标
    }
};

// 激光线,extractLine2
struct LaserLine {
    // points<y, point>
    std::map<float, LaserPoint> points;
    
    // 批量添加点
    void addPoints(const std::map<float, LaserPoint>& newPoints) {
        points.insert(newPoints.begin(), newPoints.end());
    }
    
    // 获取点数量
    size_t size() const { return points.size(); }
};

// 对应extractLine3
struct LaserLine2 {
    std::vector<LaserPoint> points;     // 激光点数据
    std::vector<float> y_coords;        // 对应的y坐标，保持有序以支持二分查找
    
    // 预留容量以减少重分配
    void reserve(size_t capacity) {
        points.reserve(capacity);
        y_coords.reserve(capacity);
    }

    // 单个点添加
    void addPoint(const LaserPoint& point, float y) {
        // 找到插入位置以保持有序
        auto it = std::lower_bound(y_coords.begin(), y_coords.end(), y);
        size_t pos = std::distance(y_coords.begin(), it);
        
        points.insert(points.begin() + pos, point);
        y_coords.insert(it, y);
    }

    // 批量添加点
    void addPoints(std::vector<LaserPoint>&& newPoints, std::vector<float>&& coords) {
        if (newPoints.size() != coords.size()) {
            throw std::invalid_argument("Points and coordinates size mismatch");
        }
        
        // 如果当前为空，直接移动
        if (points.empty()) {
            points = std::move(newPoints);
            y_coords = std::move(coords);
            return;
        }
        
        // 合并并保持有序
        size_t old_size = points.size();
        points.insert(points.end(), 
                     std::make_move_iterator(newPoints.begin()),
                     std::make_move_iterator(newPoints.end()));
        y_coords.insert(y_coords.end(),
                       std::make_move_iterator(coords.begin()),
                       std::make_move_iterator(coords.end()));
    }

    // 二分精度范围内查找 
    const LaserPoint* findPoint(float y) const {
        if (y_coords.empty()) return nullptr;
        auto it = std::lower_bound(y_coords.begin(), y_coords.end(), y);

        size_t idx = 0;
        float min_dist = std::numeric_limits<float>::max();
        const float eps = 1e-4f;

        // 检查it
        if (it != y_coords.end()) {
            float dist = std::fabs(*it - y);
            if (dist < min_dist) {
                min_dist = dist;
                idx = std::distance(y_coords.begin(), it);
            }
        }
        // 检查prev(it)
        if (it != y_coords.begin()) {
            auto it_prev = std::prev(it);
            float dist = std::fabs(*it_prev - y);
            if (dist < min_dist) {
                min_dist = dist;
                idx = std::distance(y_coords.begin(), it_prev);
            }
        }
        if (min_dist < eps) {
            return &points[idx];
        }
        return nullptr;
    }
       
    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }
    void clear() { points.clear(); y_coords.clear(); }
};



// 激光平面标定结果
struct Plane {
    cv::Vec3f normal;
    float d;
    float mean_error;
    float max_error;

    inline bool isValid() const {
        float norm = cv::norm(normal);
        return (norm > 1e-6) && (std::abs(normal[0]) > 1e-6 || std::abs(normal[1]) > 1e-6);
    }
};

// 激光曲面标定结果
struct QuadSurface {
    cv::Mat coefficients;  // 6x1的系数矩阵 [a, b, c, d, e, f]
    float rmse = 0.0;      // 曲面拟合的均方根误差
};