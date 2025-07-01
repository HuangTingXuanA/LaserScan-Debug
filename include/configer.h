#pragma once
#include "type.h"
#include <shared_mutex>
#include <unordered_map>

class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }

    //================ 专用标定结果接口 ================//
    CalibrationResult getCalibInfo() const {
        std::shared_lock lock(calib_mutex_);
        return calibration_;
    }

    template<typename F>
    void updateCalibInfo(F&& modifier) {
        std::unique_lock lock(calib_mutex_);
        modifier(calibration_);
    }

    //================ 平面方程接口 ================//
    std::vector<Plane> getPlane() const {
        std::shared_lock lock(plane_mutex_);
        return plane_;
    }

    void setPlane(const std::vector<Plane>& plane) {
        std::unique_lock lock(plane_mutex_);
        plane_.assign(plane.begin(), plane.end());
    }

    //================ 曲面方程接口 ================//
    std::vector<QuadSurface> getQuadSurfaces() const {
        std::shared_lock lock(quad_surface_mutex_);
        return quad_surfaces_;
    }

    void setQuadSurfaces(const std::vector<QuadSurface>& surfaces) {
        std::unique_lock lock(quad_surface_mutex_);
        quad_surfaces_ = surfaces;
    }

private:
    // 标定数据成员
    CalibrationResult calibration_;
    mutable std::shared_mutex calib_mutex_;

    // 平面方程数据成员
    std::vector<Plane> plane_;
    mutable std::shared_mutex plane_mutex_;

    // 二次曲面数据成员
    std::vector<QuadSurface> quad_surfaces_;
    mutable std::shared_mutex quad_surface_mutex_;
};