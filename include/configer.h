#pragma once
#include "type.h"
#include <shared_mutex>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// 全局路径与运行模式（由 main 解析命令行后赋值）
// ============================================================
inline std::vector<std::pair<cv::Mat, cv::Mat>> laser_imgs_;   // 图像对列表
inline std::vector<std::string>                 image_names_;  // 图像对名称

inline fs::path calib_info_dir_ = fs::current_path() / "calib_info_0926"; // 标定信息目录
inline fs::path laser_imgs_dir_ = fs::current_path() / "laser_0926";      // 图像目录
inline fs::path debug_img_dir   = fs::current_path() / "debug_img";       // debug 输出目录

inline bool debug_mode_     = false; // -d：开启 Debug 模式
inline int  inspect_laser_id_ = -1; // -i：查看指定左线 ID 的重投影（-1 表示未指定）

// ============================================================
// 配置管理器（线程安全的标定数据管理）
// ============================================================
class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }

    // 标定结果接口
    CalibrationResult getCalibInfo() const {
        std::shared_lock lock(calib_mutex_);
        return calibration_;
    }
    template<typename F>
    void updateCalibInfo(F&& modifier) {
        std::unique_lock lock(calib_mutex_);
        modifier(calibration_);
    }

    // 二次曲面接口
    std::vector<QuadSurface> getQuadSurfaces() const {
        std::shared_lock lock(quad_surface_mutex_);
        return quad_surfaces_;
    }
    void setQuadSurfaces(const std::vector<QuadSurface>& surfaces) {
        std::unique_lock lock(quad_surface_mutex_);
        quad_surfaces_ = surfaces;
    }

private:
    CalibrationResult        calibration_;
    mutable std::shared_mutex calib_mutex_;

    std::vector<QuadSurface>  quad_surfaces_;
    mutable std::shared_mutex quad_surface_mutex_;
};