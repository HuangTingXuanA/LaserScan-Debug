#include "main.h"
#include "twopass.h"
#include "laser.h"
#include "match.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// 打印帮助信息
// ============================================================
static void printHelp(const char* prog) {
    printf("用法: %s -f <目录> [-d] [-i <左线ID>]\n\n", prog);
    printf("  -h            显示此帮助信息并退出\n");
    printf("  -f <目录>     指定测试文件夹（需包含 left/ 和 right/ 子目录，\n");
    printf("                文件名相同的图像视为配对）\n");
    printf("  -d            开启 Debug 模式：\n");
    printf("                  · 将 match5 可视化图像输出到 debug_img/<图像名>.jpg\n");
    printf("                  · 将切片匹配详情写入 debug_img/<图像名>_slices.txt\n");
    printf("  -i <左线ID>   查看指定左图激光线 ID 的所有光曲面重投影线\n");
    printf("                （必须同时开启 -d）\n");
    printf("\n示例:\n");
    printf("  %s -f laser_0926\n", prog);
    printf("  %s -f laser_0926 -d\n", prog);
    printf("  %s -f laser_0926 -d -i 3\n", prog);
}

// ============================================================
// NOTE: -d 模式可视化改用 mproc.visualizeSlices()，尝试generateDebugImg并不在这里实现
// ============================================================


// ============================================================
// 生成 -i 模式图像：对指定左线 ID 绘制所有光曲面的重投影线
// ============================================================
static cv::Mat generateInspectImg(
    int target_l_id,
    const std::vector<LaserLine>& laser_l,
    const cv::Mat& rectify_l,
    const cv::Mat& rectify_r)
{
    const auto calib  = ConfigManager::getInstance().getCalibInfo();
    const auto planes = ConfigManager::getInstance().getQuadSurfaces();

    double fx_l = calib.P[0].at<double>(0,0), fy_l = calib.P[0].at<double>(1,1);
    double cx_l = calib.P[0].at<double>(0,2), cy_l = calib.P[0].at<double>(1,2);
    double fx_r = calib.P[1].at<double>(0,0), fy_r = calib.P[1].at<double>(1,1);
    double cx_r = calib.P[1].at<double>(0,2), cy_r = calib.P[1].at<double>(1,2);
    double baseline = -calib.P[1].at<double>(0,3) / fx_r;

    cv::Mat canvas;
    cv::Mat color_l, color_r;
    cv::cvtColor(rectify_l, color_l, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectify_r, color_r, cv::COLOR_GRAY2BGR);
    cv::hconcat(color_l, color_r, canvas);
    int off = rectify_l.cols;

    if (target_l_id < 0 || target_l_id >= (int)laser_l.size()) {
        printf("[警告] 左线 ID=%d 不存在（有效范围 0~%d）\n",
               target_l_id, (int)laser_l.size()-1);
        return canvas;
    }

    const auto& line = laser_l[target_l_id];

    // 用 cv::circle 绘制左线（白色，半径 2，易于观察）
    for (size_t i = 0; i < line.size(); ++i) {
        int x = cvRound(line.points[i].x), y = cvRound(line.y_coords[i]);
        if (x >= 1 && x < rectify_l.cols-1 && y >= 1 && y < rectify_l.rows-1)
            cv::circle(canvas, cv::Point(x, y), 2, cv::Scalar(255,255,255), -1, cv::LINE_AA);
    }

    // 调色板（每个光曲面一种颜色）
    static const std::vector<cv::Scalar> palette = {
        {0,255,0},{0,0,255},{255,0,0},{255,255,0},
        {255,0,255},{0,255,255},{128,255,0},{255,128,0}
    };

    LaserProcessor lp;

    for (int p_idx = 0; p_idx < (int)planes.size(); ++p_idx) {
        const auto& coef = planes[p_idx].coefficients;
        cv::Scalar col = palette[p_idx % palette.size()];

        // 重投影点列表（用于绘制连线和找中心）
        std::vector<cv::Point2f> reproj_pts;

        for (size_t i = 0; i < line.size(); ++i) {
            float y_f = line.y_coords[i], x_f = line.points[i].x;
            cv::Point3f ray(float((x_f-cx_l)/fx_l), float((y_f-cy_l)/fy_l), 1.f);
            ray *= 1.f / cv::norm(ray);
            auto ips = lp.findIntersection({0,0,0}, ray, coef);
            if (ips.empty()) continue;
            cv::Point3f pt3;
            bool ok = false;
            for (auto& q : ips) if (q.z>100&&q.z<1500) { pt3=q; ok=true; break; }
            if (!ok) continue;
            cv::Point3f pr(pt3.x-float(baseline), pt3.y, pt3.z);
            float xr = float(fx_r*pr.x/pr.z+cx_r);
            float yr = float(fy_r*pr.y/pr.z+cy_r);
            if (xr>=0&&xr<rectify_r.cols&&yr>=0&&yr<rectify_r.rows)
                reproj_pts.emplace_back(xr + off, yr);
        }

        // 用 cv::circle 绘制重投影点（半径 2）
        for (auto& p : reproj_pts) {
            int px = cvRound(p.x), py = cvRound(p.y);
            if (px >= 1 && px < canvas.cols-1 && py >= 1 && py < canvas.rows-1)
                cv::circle(canvas, cv::Point(px, py), 2, col, -1, cv::LINE_AA);
        }

        // 在重投影线中心标注曲面 ID
        if (!reproj_pts.empty()) {
            auto& cp = reproj_pts[reproj_pts.size()/2];
            cv::putText(canvas, "P"+std::to_string(p_idx), cp,
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, col, 2);
        }
    }
    return canvas;
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
    // --- 参数解析 ---
    bool show_help = false;
    std::string folder_arg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h") {
            show_help = true;
        } else if (arg == "-f" && i+1 < argc) {
            folder_arg = argv[++i];
        } else if (arg == "-d") {
            debug_mode_ = true;
        } else if (arg == "-i" && i+1 < argc) {
            inspect_laser_id_ = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "[错误] 未知参数: %s\n", arg.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    if (show_help) { printHelp(argv[0]); return 0; }

    // -i 只能在 -d 下使用
    if (inspect_laser_id_ >= 0 && !debug_mode_) {
        fprintf(stderr, "[错误] -i 参数必须与 -d 同时使用\n");
        printHelp(argv[0]);
        return 1;
    }

    if (folder_arg.empty()) {
        fprintf(stderr, "[错误] 请用 -f 指定测试文件夹\n");
        printHelp(argv[0]);
        return 1;
    }

    // -f 指定的目录同时作为：图像目录（含 left/ right/）和标定文件目录（含 stereo_calib.yml / quad_surface.yml）
    fs::path base_dir = fs::absolute(folder_arg);
    if (!fs::exists(base_dir)) {
        fprintf(stderr, "[错误] 路径不存在: %s\n", base_dir.c_str());
        return 1;
    }
    laser_imgs_dir_ = base_dir;
    calib_info_dir_ = base_dir;

    // debug_img 目录
    if (debug_mode_) {
        fs::create_directories(debug_img_dir);
    }

    // --- 加载标定和图像 ---
    bool ok = loadStereoCalibInfo();
    if (!ok) { fprintf(stderr, "[错误] 无法加载双目标定文件\n"); return 1; }
    ok = loadQuadSurfaceInfo();
    if (!ok) { fprintf(stderr, "[错误] 无法加载曲面标定文件\n"); return 1; }
    loadImgs();

    // --- 逐图像对处理 ---
    for (size_t img_idx = 0; img_idx < laser_imgs_.size(); ++img_idx) {
        const std::string& img_name = image_names_[img_idx];
        printf("\n图像对 %s\n", img_name.c_str());

        cv::Mat img_l, img_r;
        std::tie(img_l, img_r) = laser_imgs_[img_idx];

        // 1. 极线校正
        auto rectified = getEpipolarRectifyImage(img_l, img_r);
        cv::Mat rect_l = rectified[0].clone();
        cv::Mat rect_r = rectified[1].clone();

        // 2. 预处理（二值化）
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat bin_l = processImg2(rect_l, 0, true);
        cv::Mat bin_r = processImg2(rect_r, 1, true);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms_pre = std::chrono::duration<float,std::milli>(t1-t0).count();
        printf("预处理耗时：%.2fms\n", ms_pre);

        // 3. 生成连通区域
        std::vector<std::vector<std::pair<cv::Point,cv::Point>>> contours_l, contours_r;
        Two_PassNew4(bin_l, contours_l, img_idx);
        Two_PassNew4(bin_r, contours_r, img_idx);
        auto t2 = std::chrono::high_resolution_clock::now();
        float ms_conn = std::chrono::duration<float,std::milli>(t2-t1).count();
        printf("生成连通区域耗时：%.2fms\n", ms_conn);

        // 4. 激光线中心点提取（仅保留激光区域的原始像素）
        cv::Mat laser_l = cv::Mat::zeros(rect_l.size(), CV_8UC1);
        cv::Mat laser_r = cv::Mat::zeros(rect_r.size(), CV_8UC1);
        rect_l.copyTo(laser_l, bin_l);
        rect_r.copyTo(laser_r, bin_r);

        LaserProcessor lproc;
        auto lines_l = lproc.extractLine2(laser_l, contours_l, img_idx);
        auto lines_r = lproc.extractLine2(laser_r, contours_r, img_idx);
        auto t3 = std::chrono::high_resolution_clock::now();
        float ms_extract = std::chrono::duration<float,std::milli>(t3-t2).count();
        printf("激光线中心点提取耗时：%.2fms\n", ms_extract);

        // 按点数降序排列
        std::sort(lines_l.begin(), lines_l.end(),
                  [](const auto& a, const auto& b){ return a.size() > b.size(); });
        std::sort(lines_r.begin(), lines_r.end(),
                  [](const auto& a, const auto& b){ return a.size() > b.size(); });

        // 5. 匹配
        MatchStats stats;
        MatchProcessor mproc;
        auto results = mproc.match(lines_l, lines_r, rect_l, rect_r, stats);

        // 计算匹配完成率
        int l_pct = (stats.total_l_slices > 0)
                    ? int(100.f * stats.matched_l_slices / stats.total_l_slices) : 0;
        int r_pct = (stats.total_r_slices > 0)
                    ? int(100.f * stats.matched_r_slices / stats.total_r_slices) : 0;
        printf("匹配完成率：L-%d%%，R-%d%%\n", l_pct, r_pct);

        // 6. Debug 模式输出
        if (debug_mode_) {
            // 6a. 生成切片可视化图（按激光线着色，标注 S<id>）
            cv::Mat debug_img = mproc.visualizeSlices(lines_l, lines_r, rect_l, rect_r, results);
            fs::path img_path = debug_img_dir / (img_name + ".jpg");
            cv::imwrite(img_path.string(), debug_img);

            // 6b. 写切片匹配详情到 txt
            fs::path txt_path = debug_img_dir / (img_name + "_slices.txt");
            std::ofstream ofs(txt_path);
            ofs << "图像对 " << img_name << "\n";
            for (const auto& res : results) {
                ofs << "slice L_ID=" << res.l_slice_id
                    << "-R_ID="  << res.r_slice_id
                    << "-P_ID="  << res.p_idx
                    << " | Score="      << std::fixed << std::setprecision(2) << res.info.score
                    << ", dis_mean="    << res.info.dis_mean
                    << ", dis_mode="    << res.info.dis_mode
                    << ", dis_stddev="  << res.info.dis_stddev
                    << ", norm_census=" << res.info.norm_census
                    << "\n";
            }
            ofs.close();

            // 6c. -i 模式：绘制指定左线所有光曲面的重投影线
            if (inspect_laser_id_ >= 0) {
                cv::Mat inspect_img = generateInspectImg(
                    inspect_laser_id_, lines_l, rect_l, rect_r);
                fs::path ip_path = debug_img_dir /
                    (img_name + "_inspect_L" + std::to_string(inspect_laser_id_) + ".jpg");
                cv::imwrite(ip_path.string(), inspect_img);
            }
        }
    }

    printf("\n处理完成。\n");
    return 0;
}
