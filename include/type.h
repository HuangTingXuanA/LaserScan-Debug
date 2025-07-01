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
    float gx = 0;   // 对应处的梯度 X 分量
    float gy = 0;   // 对应处的梯度 Y 分量

    // 梯度幅值和方向
    float mag() const { return std::hypot(gx, gy); }
    float dir() const { return std::atan2(gy, gx); }
    
    // 相等性判断
    friend bool operator==(const LaserPoint& a, const LaserPoint& b) {
        return a.x == b.x && a.y == b.y; // 仅比较x,y坐标
    }
};

// 激光线
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