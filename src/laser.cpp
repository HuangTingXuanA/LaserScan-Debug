#include "laser.h"
#include <fstream>
#define DEBUG_PLANE_MATCH
#define DEBUG_PLANE_MATCH_FINAL
// #define DEBUG_CENTER_FIND

/************************************** Extract Centers
 * **************************************/
int LaserProcessor::convert_to_odd_number(float num) {
  int rounded = static_cast<int>(std::round(num)); // 四舍五入到整数

  if (rounded % 2 != 0) {
    return rounded; // 若结果本身是奇数，直接返回
  } else {
    float dist_prev = std::abs(num - (rounded - 1)); // 与下一个较小奇数的距离
    float dist_next = std::abs(num - (rounded + 1)); // 与下一个较大奇数的距离

    // 选择更近的奇数（距离相同则取较大的）
    if (dist_prev < dist_next) {
      return rounded - 1;
    } else if (dist_next < dist_prev) {
      return rounded + 1;
    } else {
      return rounded + 1; // 距离相等时优先较大值
    }
  }
}

std::tuple<cv::Mat, cv::Mat, int>
LaserProcessor::computeGaussianDerivatives(float sigma, float angle_rad,
                                           bool h_is_long_edge) {
  int ksize = cvRound(sigma * 3 * 2) + 1;
  ksize = std::max(ksize, 3) > 31 ? 31 : std::max(ksize, 3); // 限制最大核尺寸
  ksize = ksize % 2 == 0 ? ksize + 1 : ksize;                // 确保奇数

  // 创建标准高斯导数核
  cv::Mat dx_std, dy_std;
  cv::getDerivKernels(dx_std, dy_std, 1, 0, ksize, true, CV_32F);
  cv::multiply(dx_std, -1.0 / (sigma * sigma), dx_std);

  cv::Mat dx_temp, dy_temp;
  cv::getDerivKernels(dx_temp, dy_temp, 0, 1, ksize, true, CV_32F);
  cv::multiply(dy_temp, -1.0 / (sigma * sigma), dy_temp);

  // 计算旋转角度的正弦和余弦
  float vx, vy;
  if (h_is_long_edge) {
    // θ 是从 y 轴到长边 ⇒ 主轴方向为 (sinθ, -cosθ)
    vx = std::sin(angle_rad);
    vy = -std::cos(angle_rad);
  } else {
    // θ 是从 y 轴到短边 ⇒ 主轴方向为 (cosθ, sinθ)
    vx = std::cos(angle_rad);
    vy = std::sin(angle_rad);
  }

  // 创建旋转后的核
  cv::Mat dx_kernel = dx_std * vx + dy_temp * vy;
  cv::Mat dy_kernel = -dx_std * vy + dy_temp * vx;

  return {dx_kernel, dy_kernel, ksize};
}

cv::Mat LaserProcessor::computeGaussianDerivatives(float sigma, float epsilon) {
  // 确定核半径（文档2.3节）
  float x0 = sigma * std::sqrt(-2 * std::log(epsilon / 2));
  int radius = std::ceil(x0);
  int ksize = 2 * radius + 1;

  cv::Mat kernel(ksize, 1, CV_32F);
  float sum = 0.0f;

  // 高斯累积分布函数（文档公式9）
  auto phi = [](float x, float sigma) {
    return 0.5f * (1.0f + std::erf(x / (sigma * std::sqrt(2.0f))));
  };

  // 构建离散积分核（文档公式18）
  for (int i = 0; i < ksize; ++i) {
    float x = i - radius;
    float val = phi(x + 0.5f, sigma) - phi(x - 0.5f, sigma);
    kernel.at<float>(i) = val;
    sum += val;
  }

  // 归一化处理（文档2.3节）
  kernel /= sum;
  return kernel;
}

cv::Mat LaserProcessor::getSafeROI(const cv::Mat &img, int x, int y, int size) {
  int r = size / 2;
  int x0 = std::max(0, x - r);
  int y0 = std::max(0, y - r);
  int x1 = std::min(img.cols - 1, x + r);
  int y1 = std::min(img.rows - 1, y + r);

  // 保证最小3x3区域
  if (x1 - x0 < 3)
    x0 = std::max(0, x - 1), x1 = std::min(img.cols - 1, x + 1);
  if (y1 - y0 < 3)
    y0 = std::max(0, y - 1), y1 = std::min(img.rows - 1, y + 1);

  return img(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)).clone();
}

cv::Point2f LaserProcessor::computeStegerCenter(const cv::Mat &img, int x,
                                                int y, float sigma) {
  // 1. 生成积分高斯核（文档公式16-18）
  float x0 = sigma * std::sqrt(-2 * std::log(1e-4f));
  int radius = std::ceil(x0);
  int ksize = 2 * radius + 1;
  cv::Mat kernel(ksize, 1, CV_32F);

  // 高斯累积函数（文档公式9）
  auto phi = [](float x, float s) {
    return 0.5f * (1.0f + std::erf(x / (s * std::sqrt(2.0f))));
  };
  for (int i = 0; i < ksize; ++i) {
    float pos = i - radius;
    kernel.at<float>(i) = phi(pos + 0.5f, sigma) - phi(pos - 0.5f, sigma);
  }
  kernel /= cv::sum(kernel)[0]; // 归一化

  // 2. 计算一阶导数核（文档公式4）
  cv::Mat dx_kernel = cv::Mat::zeros(ksize, 1, CV_32F);
  for (int i = 1; i < ksize - 1; ++i) {
    dx_kernel.at<float>(i) = kernel.at<float>(i + 1) - kernel.at<float>(i - 1);
  }

  // 3. 计算二阶导数核（文档公式5）
  cv::Mat dxx_kernel = cv::Mat::zeros(ksize, 1, CV_32F);
  for (int i = 1; i < ksize - 1; ++i) {
    dxx_kernel.at<float>(i) =
        dx_kernel.at<float>(i + 1) - dx_kernel.at<float>(i - 1);
  }

  // 4. 提取局部ROI（考虑边界安全）
  cv::Rect roi(std::max(0, x - radius), std::max(0, y - radius),
               std::min(ksize, img.cols - x + radius),
               std::min(ksize, img.rows - y + radius));
  if (roi.width < 3 || roi.height < 3)
    return cv::Point2f(-1, -1);

  cv::Mat patch = img(roi).clone();

  // 5. 计算偏导数（文档§2.3）
  cv::Mat I_x, I_y, I_xx, I_xy, I_yy;
  cv::filter2D(patch, I_x, CV_32F, dx_kernel);     // x方向一阶导
  cv::filter2D(patch, I_y, CV_32F, dx_kernel.t()); // y方向一阶导
  cv::filter2D(I_x, I_xx, CV_32F, dxx_kernel);     // xx二阶导
  cv::filter2D(I_x, I_xy, CV_32F, dx_kernel.t());  // xy混合导
  cv::filter2D(I_y, I_yy, CV_32F, dxx_kernel.t()); // yy二阶导

  // 6. 中心点Hessian矩阵（文档公式21）
  int cx = patch.cols / 2, cy = patch.rows / 2;
  float a = I_xx.at<float>(cy, cx);
  float b = I_xy.at<float>(cy, cx);
  float c = I_yy.at<float>(cy, cx);
  cv::Matx22f H(a, b, b, c);

  // 7. 特征分解求法向量（文档§2.4）
  cv::Vec2f eigenvalues;
  cv::Matx22f eigenvectors;
  if (!cv::eigen(H, eigenvalues, eigenvectors))
    return cv::Point2f(-1, -1);

  // 选取最大特征值对应向量
  int max_idx = (std::abs(eigenvalues[0]) > std::abs(eigenvalues[1])) ? 0 : 1;
  cv::Vec2f n(eigenvectors(max_idx, 0), eigenvectors(max_idx, 1));

  // 8. 泰勒展开求解亚像素偏移（文档公式23）
  cv::Vec2f grad(I_x.at<float>(cy, cx), I_y.at<float>(cy, cx));
  float numerator = grad.dot(n);
  float denominator = n[0] * (H(0, 0) * n[0] + H(0, 1) * n[1]) +
                      n[1] * (H(1, 0) * n[0] + H(1, 1) * n[1]);

  if (std::abs(denominator) < 1e-6)
    return cv::Point2f(-1, -1);

  float t = -numerator / denominator;

  // 9. 约束偏移量（文档§2.4）
  float max_t = 0.5f / std::max(std::abs(n[0]), std::abs(n[1]));
  t = std::clamp(t, -max_t, max_t);

  // 10. 计算最终坐标
  return cv::Point2f(x + t * n[0], y + t * n[1]);
}

float LaserProcessor::interpolateChannel(const cv::Mat &img, float x, float y) {
  int xi = std::clamp(static_cast<int>(x), 0, img.cols - 2);
  int yi = std::clamp(static_cast<int>(y), 0, img.rows - 2);

  float dx = x - xi;
  float dy = y - yi;

  const float *row0 = img.ptr<float>(yi);
  const float *row1 = img.ptr<float>(yi + 1);

  float a = row0[xi];
  float b = row0[xi + 1];
  float c = row1[xi];
  float d = row1[xi + 1];

  return a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy +
         d * dx * dy;
}

double LaserProcessor::interpolateChannel(const cv::Mat &img, double x,
                                          double y) {
  int xi = std::clamp(static_cast<int>(x), 0, img.cols - 2);
  int yi = std::clamp(static_cast<int>(y), 0, img.rows - 2);

  double dx = x - xi;
  double dy = y - yi;

  const float *row0 = img.ptr<float>(yi);
  const float *row1 = img.ptr<float>(yi + 1);

  double a = row0[xi];
  double b = row0[xi + 1];
  double c = row1[xi];
  double d = row1[xi + 1];

  return static_cast<double>(a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) +
                             c * (1 - dx) * dy + d * dx * dy);
}

float LaserProcessor::findSymmetricCenter(const cv::Mat &img, float x, float y,
                                          cv::Vec2f dir, float search_range) {
  // 阶段1：梯度预检测快速定位
  const float coarse_step = 0.2f;
  float max_grad = -FLT_MAX;
  float peak_t = 0;

  // 梯度检测（中心差分法）
  for (float t = -search_range; t <= search_range; t += coarse_step) {
    float val_prev = interpolateChannel(img, x + (t - 0.1f) * dir[0],
                                        y + (t - 0.1f) * dir[1]);
    float val_next = interpolateChannel(img, x + (t + 0.1f) * dir[0],
                                        y + (t + 0.1f) * dir[1]);
    float grad = val_next - val_prev;
    if (grad > max_grad) {
      max_grad = grad;
      peak_t = t;
    }
  }

  // 阶段2：精细采样
  const float fine_range = search_range * 0.6f; // 根据线宽自适应
  std::vector<std::pair<float, float>> profile;
  profile.reserve(static_cast<int>(2 * fine_range / 0.1f) + 2);

  // 获取当前剖面数据并计算max_val
  float max_val = -FLT_MAX;
  for (float t = peak_t - fine_range; t <= peak_t + fine_range; t += 0.02f) {
    float val = interpolateChannel(img, x + t * dir[0], y + t * dir[1]);
    profile.emplace_back(t, val);
    if (val > max_val)
      max_val = val;
  }

  // 方法A：梯度对称点检测（一阶导数过零点）
  float sym_center = FLT_MAX;
  {
    std::vector<float> deriv1;
    for (size_t i = 1; i < profile.size() - 1; ++i) {
      deriv1.push_back((profile[i + 1].second - profile[i - 1].second) /
                       0.2f); // 中心差分
    }

    // 寻找正到负的过零点
    for (size_t i = 1; i < deriv1.size(); ++i) {
      if (deriv1[i - 1] > 0 && deriv1[i] < 0) {
        // 线性插值求精确过零点
        float alpha = deriv1[i] / (deriv1[i] - deriv1[i - 1]);
        sym_center = profile[i].first - 0.1f * alpha;
        break;
      }
    }
  }

  // 方法B：加权质心法（动态阈值）
  float centroid = FLT_MAX;
  {
    const float threshold = max_val * 0.7f; // 自适应阈值
    float sum_wt = 0.0f, sum_w = 0.0f;

    for (const auto &p : profile) {
      if (p.second < threshold)
        continue;
      float w = p.second - threshold;
      sum_wt += w * p.first;
      sum_w += w;
    }

    if (sum_w > 1e-6f) {
      centroid = sum_wt / sum_w;
    }
  }

  // 结果融合策略
  float final_center = FLT_MAX;

  // 情况1：两种方法均有效时取加权平均
  if (sym_center != FLT_MAX && centroid != FLT_MAX) {
    if (std::abs(sym_center - centroid) < 0.2f) { // 结果一致性检查
      final_center = 0.4f * sym_center + 0.6f * centroid;
    } else {
      final_center = centroid; // 优先质心法
    }
  }
  // 情况2：仅一种方法有效
  else if (centroid != FLT_MAX) {
    final_center = centroid;
  } else if (sym_center != FLT_MAX) {
    final_center = sym_center;
  }

  return final_center;
}

float LaserProcessor::findSymmetricCenter2(const cv::Mat &img, float x, float y,
                                           cv::Vec2f dir, float search_range,
                                           size_t is_right) {
#ifdef DEBUG_CENTER_FIND
  cv::namedWindow("Gray Profile Visualization", cv::WINDOW_NORMAL);
#endif
  // 1. 搜索区间
  std::vector<std::pair<float, float>> profile; // (位置t, 灰度值val)

  // 更细致的步长
  float step = 0.1f;

  // 在法向量方向采样灰度剖面
  for (float t = -search_range; t <= search_range; t += step) {
    float px = x + t * dir[0];
    float py = y + t * dir[1];
    float val = interpolateChannel(img, px, py);
    profile.emplace_back(t, val);
  }

  // 2. 找到灰度峰值区域
  float max_val = -FLT_MAX;
  for (const auto &p : profile) {
    max_val = std::max(max_val, p.second);
  }

  // 提取接近峰值的所有点（考虑到灰度相同的情况）
  const float tolerance = 0.01f; // 灰度值容差
  std::vector<float> peak_positions;

  for (const auto &p : profile) {
    if (std::fabs(p.second - max_val) <= tolerance) {
      peak_positions.push_back(p.first);
    }
  }

  float res1 = FLT_MAX, res2 = FLT_MAX, res3 = FLT_MAX;
  std::vector<float> fit_t, fit_val;
  fit_t.resize(7), fit_val.resize(7);

  // 3. 如果有多个灰度相同的点，找到峰值区域的中心
  if (peak_positions.size() > 1) {
    // 连续同值区域分组
    std::vector<std::vector<float>> regions;
    std::vector<float> current_region = {peak_positions[0]};

    for (size_t i = 1; i < peak_positions.size(); ++i) {
      // 如果与前一个点连续（考虑采样步长），归为同一区域
      if (std::fabs(peak_positions[i] - peak_positions[i - 1]) <= 1.5f * step) {
        current_region.push_back(peak_positions[i]);
      } else {
        regions.push_back(current_region);
        current_region = {peak_positions[i]};
      }
    }

    if (!current_region.empty()) {
      regions.push_back(current_region);
    }

    // 选择最长的连续区域（可能是线条的中心平台）
    auto longest_region = std::max_element(
        regions.begin(), regions.end(),
        [](const std::vector<float> &a, const std::vector<float> &b) {
          return a.size() < b.size();
        });

    if (longest_region != regions.end()) {
      // 返回该区域的中心位置
      float sum = 0;
      for (float pos : *longest_region) {
        sum += pos;
      }
      res1 = sum / longest_region->size();

#ifdef DEBUG_CENTER_FIND

      fit_t[0] = res1;
      fit_val[0] = 0.9f;
#endif
    }
  }

  // 4. 如果没有明显的平台区域，使用所有峰值的中心
  if (!peak_positions.empty()) {
    float sum = 0;
    for (float pos : peak_positions) {
      sum += pos;
    }
    res2 = sum / peak_positions.size();

#ifdef DEBUG_CENTER_FIND
    fit_t[1] = res2;
    fit_val[1] = 0.9f;
#endif
  }

  // 5. 抛物线拟合
  auto max_it = std::max_element(
      profile.begin(), profile.end(),
      [](const std::pair<float, float> &a, const std::pair<float, float> &b) {
        return a.second < b.second;
      });

  size_t max_idx = std::distance(profile.begin(), max_it);

  // 抛物线五点拟合 - 改进版
  float a_coef, b_coef, c_coef;
  if (profile.size() >= 5) {
    cv::Mat A(5, 3, CV_32F);
    cv::Mat b(5, 1, CV_32F);

    // 使用基于分布的采样点选择策略
    std::vector<size_t> selected_indices;

    // 确保包含最大值点
    selected_indices.push_back(max_idx);

    // 计算合适的最小间隔（基于profile的总跨度）
    float total_range = profile.back().first - profile.front().first;
    // 将最小间隔从1/10改为1/30，使采样更密集
    float min_distance = total_range / 30.0f; // 最小间隔为总范围的三十分之一

    // 创建非对称间隔策略，使峰值附近采样更密集
    std::vector<float> distance_factors = {0.5f, 0.75f, 1.5f,
                                           2.5f}; // 不同距离因子

    // 向左右两侧各选择两个点，峰值附近间隔更小
    for (int direction = -1; direction <= 1; direction += 2) {
      float last_t = profile[max_idx].first;
      int idx = max_idx;

      for (int count = 0; count < 2; count++) {
        bool found = false;
        // 根据离峰值的距离动态调整最小间隔
        float current_min_distance = min_distance * distance_factors[count];

        while (true) {
          idx += direction;
          if (idx < 0 || idx >= profile.size())
            break;

          if (std::abs(profile[idx].first - last_t) >= current_min_distance) {
            selected_indices.push_back(idx);
            last_t = profile[idx].first;
            found = true;
            break;
          }
        }
        if (!found)
          break;
      }
    }

    // 如果没有选出足够的点，使用备选策略
    if (selected_indices.size() < 5) {
      selected_indices.clear();

      // 备选策略：均匀分布选择
      for (int i = 0; i < 5; i++) {
        size_t idx =
            static_cast<size_t>(i * (profile.size() - 1) / 4); // 均匀分布
        selected_indices.push_back(idx);
      }

      // 确保包含最大值点
      bool has_max = false;
      for (size_t idx : selected_indices) {
        if (idx == max_idx) {
          has_max = true;
          break;
        }
      }

      if (!has_max) {
        // 用中间点替换为max_idx
        selected_indices[2] = max_idx;
      }
    }

    // 排序确保索引有序
    std::sort(selected_indices.begin(), selected_indices.end());

    // 使用选择的点进行拟合
    for (int i = 0; i < selected_indices.size(); i++) {
      const auto &p = profile[selected_indices[i]];
      A.at<float>(i, 0) = p.first * p.first;
      A.at<float>(i, 1) = p.first;
      A.at<float>(i, 2) = 1;
      b.at<float>(i, 0) = p.second;

#ifdef DEBUG_CENTER_FIND
      if (i + 2 < fit_t.size()) {
        fit_t[i + 2] = p.first;
        fit_val[i + 2] = p.second;
      }
#endif
    }

    cv::Mat coeffs;
    if (cv::solve(A, b, coeffs, cv::DECOMP_SVD)) {
      a_coef = coeffs.at<float>(0);
      b_coef = coeffs.at<float>(1);
      c_coef = coeffs.at<float>(2);

      if (a_coef < 0) { // 仅当抛物线开口向下时才接受拟合结果
        res3 = -b_coef / (2 * a_coef);

        // 计算拟合点两侧灰度值
        float left_intensity = interpolateChannel(img, x + (res3 - 3) * dir[0],
                                                  y + (res3 - 3) * dir[1]);

        float right_intensity =
            interpolateChannel(img, x + (res3 + search_range / 2) * dir[0],
                               y + (res3 + search_range / 2) * dir[1]);
        float intensity_change = std::fabs(left_intensity - right_intensity);

        // 验证梯度幅值是否达标
        if (intensity_change < 0.1) {
          res3 = FLT_MAX; // 拒绝平缓区域
        }
      }
    }
  }

  /** 三点拟合
  if (max_idx > 0 && max_idx < profile.size() - 1) {
      // 取三个点进行抛物线拟合
      float t0 = profile[max_idx-1].first;
      float t1 = profile[max_idx].first;
      float t2 = profile[max_idx+1].first;

      float y0 = profile[max_idx-1].second;
      float y1 = profile[max_idx].second;
      float y2 = profile[max_idx+1].second;

      // 抛物线拟合公式: y = a*t^2 + b*t + c
      float denom = (t0-t1)*(t0-t2)*(t1-t2);
      if (std::fabs(denom) > 1e-6f) {
          float a = ((t2*(y1-y0) + t1*(y0-y2) + t0*(y2-y1)) / denom);
          float b = ((t2*t2*(y0-y1) + t1*t1*(y2-y0) + t0*t0*(y1-y2)) / denom);

          // 计算抛物线峰值位置: t = -b/(2a)
          if (std::fabs(a) > 1e-6f) {
              return -b/(2*a);
          }
      }
  }
  */

#ifdef DEBUG_CENTER_FIND
  // 可视化灰度剖面和抛物线拟合（每隔一定步长进行一次，避免过多）
  // if (is_right == 0 && x > 720 && x < 815 && y > 300 && y < 370) {
  // 创建FreeType2对象并加载中文字体
  cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
  ft2->loadFontData("./simsun.ttc", 0);

  // 创建灰度剖面图，增大宽度以容纳图例
  cv::Mat profile_img(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));

  // 绘制坐标轴
  cv::line(profile_img, cv::Point(50, 350), cv::Point(550, 350),
           cv::Scalar(0, 0, 0), 1);
  cv::line(profile_img, cv::Point(50, 50), cv::Point(50, 350),
           cv::Scalar(0, 0, 0), 1);

  // 添加水平轴刻度和标签
  int num_ticks = 5; // t轴上的刻度数量(每边)
  for (int i = -num_ticks; i <= num_ticks; i++) {
    float tick_t = i * search_range / num_ticks;
    int tick_x = 50 + static_cast<int>((tick_t + search_range) /
                                       (2 * search_range) * 500);

    // 绘制刻度线
    cv::line(profile_img, cv::Point(tick_x, 350), cv::Point(tick_x, 355),
             cv::Scalar(0, 0, 0), 1);

    // 添加刻度值
    std::string tick_label = cv::format("%.1f", tick_t);
    ft2->putText(profile_img, tick_label, cv::Point(tick_x - 10, 370), 14,
                 cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
  }

  // 添加垂直轴刻度和标签
  int v_num_ticks = 5; // intensity轴上的刻度数量
  for (int i = 0; i <= v_num_ticks; i++) {
    float tick_val = static_cast<float>(i) / v_num_ticks;
    int tick_y = 350 - static_cast<int>(tick_val * 300);

    // 绘制刻度线
    cv::line(profile_img, cv::Point(45, tick_y), cv::Point(50, tick_y),
             cv::Scalar(0, 0, 0), 1);

    // 添加刻度值
    std::string tick_label = cv::format("%.1f", tick_val);
    ft2->putText(profile_img, tick_label, cv::Point(25, tick_y + 5), 14,
                 cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
  }

  // 添加坐标轴标签
  ft2->putText(profile_img, "t", cv::Point(550, 365), 16, cv::Scalar(0, 0, 0),
               -1, cv::LINE_AA, true);
  ft2->putText(profile_img, "intensity", cv::Point(30, 40), 16,
               cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);

  // 映射函数将灰度剖面值映射到图像坐标
  auto mapToImg = [&](float t, float val) -> cv::Point {
    int img_x =
        50 + static_cast<int>((t + search_range) / (2 * search_range) * 500);
    int img_y = 350 - static_cast<int>(val * 300);
    return cv::Point(img_x, img_y);
  };

  // 绘制灰度剖面点
  for (size_t i = 0; i < profile.size(); ++i) {
    cv::Point pt = mapToImg(profile[i].first, profile[i].second);
    cv::circle(profile_img, pt, 2, cv::Scalar(0, 0, 255), -1);

    // 连接相邻点
    if (i > 0) {
      cv::Point prev_pt = mapToImg(profile[i - 1].first, profile[i - 1].second);
      cv::line(profile_img, prev_pt, pt, cv::Scalar(0, 0, 255), 1);
    }
  }

  // 绘制抛物线拟合结果
  for (float t = -search_range; t <= search_range; t += 0.1f) {
    float val = a_coef * t * t + b_coef * t + c_coef;
    if (val >= 0 && val <= 1) {
      cv::Point pt = mapToImg(t, val);
      cv::circle(profile_img, pt, 1, cv::Scalar(0, 255, 0), -1);
    }
  }

  // 标记拟合使用的点
  for (size_t i = 2; i < fit_t.size(); ++i) {
    cv::Point pt = mapToImg(fit_t[i], fit_val[i]);
    cv::circle(profile_img, pt, 4, cv::Scalar(255, 0, 0), 2);
  }

  // 标记计算的峰值位置
  if (res1 != FLT_MAX) {
    cv::Point pt_res1 = mapToImg(fit_t[0], fit_val[0]);
    cv::circle(profile_img, pt_res1, 5, cv::Scalar(255, 255, 0), -1); // 青色
  }
  if (res2 != FLT_MAX) {
    cv::Point pt_res2 = mapToImg(fit_t[1], fit_val[1]);
    cv::circle(profile_img, pt_res2, 5, cv::Scalar(30, 105, 210),
               -1); // 巧克力色
  }
  if (res3 != FLT_MAX) {
    cv::Point peak_pt =
        mapToImg(res3, a_coef * res3 * res3 + b_coef * res3 + c_coef);
    cv::circle(profile_img, peak_pt, 5, cv::Scalar(0, 255, 255), -1); // 黄色
  }

  // 添加文字说明
  ft2->putText(
      profile_img,
      "Profile at (" + std::to_string(x) + "," + std::to_string((int)y) + ")",
      cv::Point(50, 30), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA, true);
  ft2->putText(profile_img,
               "a=" + std::to_string(a_coef) + ", b=" + std::to_string(b_coef) +
                   ", c=" + std::to_string(c_coef) +
                   ", res1=" + (res1 == FLT_MAX ? "no" : std::to_string(res1)) +
                   ", res2=" + (res2 == FLT_MAX ? "no" : std::to_string(res2)) +
                   ", res3=t_peak",
               cv::Point(50, 390), 16, cv::Scalar(0, 0, 0), -1, cv::LINE_AA,
               true);

  // 添加右侧图例
  int legend_x = 590;
  int legend_y = 80;
  int legend_spacing = 40;

  // 图例标题
  ft2->putText(profile_img, "图例说明", cv::Point(legend_x, legend_y), 20,
               cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
  legend_y += 30;

  // 原始灰度剖面
  cv::circle(profile_img, cv::Point(legend_x, legend_y), 2,
             cv::Scalar(0, 0, 255), -1);
  cv::line(profile_img, cv::Point(legend_x - 10, legend_y),
           cv::Point(legend_x + 10, legend_y), cv::Scalar(0, 0, 255), 1);
  ft2->putText(profile_img, "灰度剖面采样点",
               cv::Point(legend_x + 20, legend_y + 5), 20, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);

  // 抛物线拟合
  legend_y += legend_spacing;
  cv::circle(profile_img, cv::Point(legend_x, legend_y), 2,
             cv::Scalar(0, 255, 0), -1);
  cv::line(profile_img, cv::Point(legend_x - 10, legend_y),
           cv::Point(legend_x + 10, legend_y), cv::Scalar(0, 255, 0), 1);
  ft2->putText(profile_img, "抛物线拟合曲线",
               cv::Point(legend_x + 20, legend_y + 5), 20, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);

  // 拟合使用的点
  legend_y += legend_spacing;
  cv::circle(profile_img, cv::Point(legend_x, legend_y), 4,
             cv::Scalar(255, 0, 0), 2);
  ft2->putText(profile_img, "拟合使用的采样点",
               cv::Point(legend_x + 20, legend_y + 5), 20, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);

  // 计算的峰值位置
  legend_y += legend_spacing;
  cv::circle(profile_img, cv::Point(legend_x, legend_y), 5,
             cv::Scalar(0, 255, 255), -1);
  ft2->putText(profile_img, "计算的线中心位置",
               cv::Point(legend_x + 20, legend_y + 5), 20, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);

  // 添加计算方法说明
  legend_y += legend_spacing;
  ft2->putText(profile_img, "中心点计算方法:", cv::Point(legend_x, legend_y),
               20, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA, true);
  legend_y += 25;
  ft2->putText(profile_img, "1. y = at²+bt+c",
               cv::Point(legend_x + 10, legend_y), 18, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);
  legend_y += 20;
  ft2->putText(profile_img, "2. t_peak = -b/(2a)",
               cv::Point(legend_x + 10, legend_y), 18, cv::Scalar(0, 0, 0),
               cv::FILLED, cv::LINE_AA, true);
  legend_y += 20;
  if (res3 == FLT_MAX)
    ft2->putText(profile_img, "3. t_peak = 无",
                 cv::Point(legend_x + 10, legend_y), 20, cv::Scalar(0, 0, 0),
                 cv::FILLED, cv::LINE_AA, true);
  else
    ft2->putText(profile_img, "3. t_peak = " + cv::format("%.4f", res3),
                 cv::Point(legend_x + 10, legend_y), 18, cv::Scalar(0, 0, 0),
                 cv::FILLED, cv::LINE_AA, true);

  // 显示图像
  const std::string profile_name = "./gray_profile/(" + std::to_string(x) +
                                   "_" + std::to_string(y) + ").png";
  // cv::imwrite(profile_name, profile_img);
  while (true) {
    cv::imshow("Gray Profile Visualization", profile_img);
    if (cv::waitKey(0) == 27)
      break;
  }
  // }
#endif

  // 如果所有方法都失败，说明是极端边缘点
  float final_res = FLT_MAX;
  if (res1 != FLT_MAX)
    final_res = res1;
  else if (res2 != FLT_MAX)
    final_res = res2;
  else if (res3 != FLT_MAX)
    final_res = res3;
  return final_res;
}

float LaserProcessor::findSymmetricCenter3(const cv::Mat &img, float x, float y,
                                           cv::Vec2f dir, float R) {
  // 采样与拟合参数
  const float step = 0.1f; // t 方向采样步长
#ifdef DEBUG_CENTER_FIND
  // 准备可视化窗口
  cv::namedWindow("GrayProfile", cv::WINDOW_NORMAL);
#endif

  float extend_range = R;
  // if (R < 5) extend_range += extend_range * 0.8f;
  // else if (R < 10) extend_range += extend_range * 0.7f;
  // else if (R > 10) extend_range += 5.0f;

  // 1. 全范围采样
  std::vector<double> Ts, Vs;
  for (float t = -2.0f; t <= extend_range; t += step) {
    Ts.push_back(t);
    Vs.push_back(interpolateChannel(img, double(x + t * dir[0]),
                                    double(y + t * dir[1])));
  }
  int N = (int)Ts.size();
  if (N < 3)
    return FLT_MAX;

  // 2. 找到第一个峰值
  double maxVal = -FLT_MAX;
  int peakIdx = -1;
  bool isRising = false;
  for (int i = 0; i < N; ++i) {
    // 只考虑 t <= R 的点
    if (Vs[i] > maxVal && Ts[i] <= R) {
      maxVal = Vs[i];
      peakIdx = i;
    }
  }
  double t_peak = Ts[peakIdx];
  if (Vs[peakIdx] < 0.5f)
    return FLT_MAX;

  // 确定第一个最大峰的上升和下降沿边界
  int leftBound = peakIdx - 1;  // 初始化为峰顶位置
  int rightBound = peakIdx + 1; // 初始化为峰顶位置
  while (leftBound > 0) {
    // 检测灰度值上升：表示上升沿开始
    float diff = Vs[leftBound + 1] - Vs[leftBound];
    if ((diff <= 1e-3f && Vs[leftBound] <= 0.05) || Ts[leftBound] < -R - 1)
      break;
    leftBound--;
  }
  leftBound += 2;
  while (rightBound < N) {
    // 检测灰度值上升：表示下降沿结束
    float diff = Vs[rightBound] - Vs[rightBound - 1];
    if ((diff <= 1e-3f && Vs[rightBound] <= 0.05) || Ts[rightBound] > R + 1)
      break;
    rightBound++;
  }
  rightBound -= 2;

  // 3. 平台检测 → res1
  double res1 = FLT_MAX;
  {
    double v0 = Vs[peakIdx];
    int L = peakIdx, Rg = peakIdx;
    while (L > 0 && std::fabs(Vs[L - 1] - v0) < 1e-5f)
      --L;
    while (Rg + 1 < rightBound && std::fabs(Vs[Rg + 1] - v0) < 1e-5f)
      ++Rg;
    if (Rg - L + 1 >= 10) { // 大于1个像素
      res1 = 0.5f * (Ts[L] + Ts[Rg]);
    }
  }
#ifndef DEBUG_CENTER_FIND
  // if (res1 != FLT_MAX) {
  //     return res1;
  // }
#endif

  // 4. 抛物线拟合 → res2
  float res2 = FLT_MAX;
  {
    // 确定左右两侧长度
    int leftLen = peakIdx - leftBound - 1;   // 左侧点个数（不包括峰值）
    int rightLen = rightBound - 1 - peakIdx; // 右侧点个数（不包括峰值）

    // 判断哪边更短，并获取短边末端索引和灰度值
    int shortEnd = 0;
    float targetGray;
    int searchStart, searchEnd;

    if (leftLen < rightLen) {
      // 左侧较短
      shortEnd = leftBound;
      targetGray = Vs[leftBound];
      searchStart = peakIdx + 1;
      searchEnd = rightBound;
    } else {
      // 右侧较短
      shortEnd = rightBound;
      targetGray = Vs[rightBound];
      searchStart = leftBound;
      searchEnd = peakIdx - 1;
    }

    // 在长边寻找灰度最接近的点
    int matchIdx = -1;
    float minDiff = FLT_MAX;
    for (int i = searchStart; i <= searchEnd; i++) {
      float diff = std::fabs(Vs[i] - targetGray);
      if (diff < minDiff) {
        minDiff = diff;
        matchIdx = i;
      }
    }

    // 确定拟合点集范围
    int startIdx, endIdx;
    if (leftLen < rightLen) {
      startIdx = shortEnd; // 左侧末端 (0)
      endIdx = matchIdx;   // 右侧匹配点
    } else {
      startIdx = matchIdx; // 左侧匹配点
      endIdx = shortEnd;   // 右侧末端 (N-1)
    }
    int numPoints = endIdx - startIdx + 1;

    // 至少需要3个点才能拟合抛物线
    if (matchIdx != -1 && numPoints >= 3) {
      cv::Mat A(numPoints, 3, CV_32F), Y(numPoints, 1, CV_32F);
      for (int i = 0; i < numPoints; ++i) {
        int idx = startIdx + i;
        float t = Ts[idx];
        A.at<float>(i, 0) = t * t;
        A.at<float>(i, 1) = t;
        A.at<float>(i, 2) = 1.0f;
        Y.at<float>(i, 0) = Vs[idx];
      }

      cv::Mat coeff;
      if (cv::solve(A, Y, coeff, cv::DECOMP_SVD)) {
        float a = coeff.at<float>(0, 0);
        float b = coeff.at<float>(1, 0);
        if (a < 0) { // 确保是开口向下的抛物线
          float tp = -b / (2 * a);
          if (tp >= -extend_range && tp <= extend_range) {
            res2 = tp;
          }
        }
      }
    }
  }
#ifndef DEBUG_CENTER_FIND
  if (res2 != FLT_MAX) {
    return res2;
  }
#endif

  // 泰勒展开法 → res3
  float dx = (Vs[peakIdx + 1] - Vs[peakIdx - 1]) / (2 * step);
  float dxx =
      (Vs[peakIdx + 1] - 2 * Vs[peakIdx] + Vs[peakIdx - 1]) / (step * step);
  float offset = -dx / dxx;
  float res3 = Ts[peakIdx] + offset;
#ifndef DEBUG_CENTER_FIND
  // return res3;
#endif

  float left_bound = Ts[leftBound];
  float right_bound = Ts[rightBound];
  double res4 = t_peak;

#ifdef DEBUG_CENTER_FIND
  //
  // —— 可视化部分 ——
  //
  int W = 800, H = 500;
  cv::Mat prof_img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  int x0 = 60, y0 = H - 60;
  int x1 = W - 200, y1 = 60;
  // 0) 目前的点
  cv::putText(prof_img,
              cv::format("%s%.3f%s%.3f%s", "Profile at (", x, ",", y, ")"),
              cv::Point(50, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 1);

  // 1) 坐标轴
  cv::line(prof_img, {x0, y0}, {x1, y0}, {0, 0, 0}, 1);
  cv::line(prof_img, {x0, y0}, {x0, y1}, {0, 0, 0}, 1);

  // 2) 刻度与标签
  int n_xt = 5;
  for (int i = 0; i <= n_xt; ++i) {
    float t_tick = -extend_range + 2 * extend_range * i / n_xt;
    int xt = x0 + int((t_tick + extend_range) / (2 * extend_range) * (x1 - x0));
    cv::line(prof_img, {xt, y0 - 5}, {xt, y0 + 5}, {0, 0, 0}, 1);
    cv::putText(prof_img, cv::format("%.2f", t_tick), {xt - 20, y0 + 25},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  }
  int n_yt = 5;
  for (int i = 0; i <= n_yt; ++i) {
    float v_tick = float(i) / n_yt;
    int yt = y0 - int(v_tick * (y0 - y1));
    cv::line(prof_img, {x0 - 5, yt}, {x0 + 5, yt}, {0, 0, 0}, 1);
    cv::putText(prof_img, cv::format("%.2f", v_tick), {x0 - 60, yt + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  }

  // 3) 全采样点 & 连线（灰）
  for (int i = 0; i < N; ++i) {
    int xx = x0 + int((Ts[i] + extend_range) / (2 * extend_range) * (x1 - x0));
    int yy = y0 - int(Vs[i] * (y0 - y1));
    cv::circle(prof_img, {xx, yy}, 3, {180, 180, 180}, -1);
    if (i > 0) {
      int xx0 =
          x0 + int((Ts[i - 1] + extend_range) / (2 * extend_range) * (x1 - x0));
      int yy0 = y0 - int(Vs[i - 1] * (y0 - y1));
      cv::line(prof_img, {xx0, yy0}, {xx, yy}, {200, 200, 200}, 1);
    }
  }

  // 4) 四条垂直标线 & 图例
  struct Rst {
    float t;
    cv::Scalar c;
    const char *lbl;
  };
  std::vector<Rst> rst = {{res1, {0, 255, 0}, "res1"},
                          {res2, {255, 0, 0}, "res2"},
                          {res3, {255, 255, 0}, "res3"},
                          {res4, {255, 111, 200}, "res4"},
                          {left_bound, {128, 0, 128}, "l_bound"},
                          {right_bound, {128, 0, 128}, "r_bound"}};
  int lx = W - 180, ly = 80, dy = 30;
  for (auto &r : rst) {
    if (r.t == FLT_MAX)
      continue;
    int xx = x0 + int((r.t + extend_range) / (2 * extend_range) * (x1 - x0));
    cv::line(prof_img, {xx, y1}, {xx, y0}, r.c, 1);
    cv::putText(prof_img, cv::format("%s: %.3f", r.lbl, r.t), {lx, ly},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, r.c, 2);
    ly += dy;
  }

  cv::imshow("GrayProfile", prof_img);
  cv::waitKey(0);
#endif

  return res4;
}

float LaserProcessor::findSymmetricCenter4(const cv::Mat &img, float x, float y,
                                           cv::Vec2f dir, float R) {
  // 采样与拟合参数
  const float step = 0.1f; // t 方向采样步长
#ifdef DEBUG_CENTER_FIND
  // 准备可视化窗口
  cv::namedWindow("GrayProfile", cv::WINDOW_NORMAL);
#endif

  float extend_range = R;

  // 1. 全范围采样
  std::vector<float> Ts, Vs;
  for (float t = -2.0f; t <= extend_range; t += step) {
    float val = img.at<float>(y + t * dir[1], x + t * dir[0]);
    Ts.push_back(t);
    Vs.push_back(val);
  }
  int N = (int)Ts.size();
  if (N < 3)
    return FLT_MAX;

  // 2. 找到第一个峰值
  float maxVal = -FLT_MAX;
  int peakIdx = -1;
  for (int i = 0; i < N; ++i) {
    // 只考虑 t <= R 的点
    if (Vs[i] > maxVal && Ts[i] <= R) {
      maxVal = Vs[i];
      peakIdx = i;
    }
  }
  float t_peak = Ts[peakIdx];
  if (Vs[peakIdx] < 125.0f)
    return FLT_MAX;

  // 3. 平台检测 → res1
  float res1 = FLT_MAX;
  {
    float v0 = Vs[peakIdx];
    int L = peakIdx, Rg = peakIdx;
    while (L > 0 && std::fabs(Vs[L - 1] - v0) <= 20)
      --L;
    while (Rg + 1 < N && std::fabs(Vs[Rg + 1] - v0) <= 20)
      ++Rg;
    if (Rg - L + 1 >= 10) { // 大于1个像素
      res1 = 0.5f * (Ts[L] + Ts[Rg]);
    }
  }
#ifndef DEBUG_CENTER_FIND
  // if (res1 != FLT_MAX) {
  //     return res1;
  // }
#endif

  // 4. 灰度值不为0的纳入求中心 → res2
  float res2 = t_peak;
  {
    std::vector<float> fit_t;
    for (int i = 0; i < Vs.size(); ++i) {
      if (Vs[i] > 0.0f)
        fit_t.push_back(Ts[i]);
    }
    res2 = 0.5f * (fit_t.front() + fit_t.back()) + 0.1f;
  }

  // 5. 灰度值不为0的纳入求质心 → res3
  float res3 = t_peak;
  {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
      float val = Vs[i];
      if (val > 0.0f) {
        num += Ts[i] * val; // 加权坐标
        den += val;         // 总“质量”
      }
    }
    res3 = static_cast<float>(num / den) + 0.1f; // 质心
  }

#ifdef DEBUG_CENTER_FIND
  //
  // —— 可视化部分 ——
  //
  int W = 800, H = 500;
  cv::Mat prof_img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
  int x0 = 60, y0 = H - 60;
  int x1 = W - 200, y1 = 60;
  // 0) 目前的点
  cv::putText(prof_img,
              cv::format("%s%.3f%s%.3f%s", "Profile at (", x, ",", y, ")"),
              cv::Point(50, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 1);

  // 1) 坐标轴
  cv::line(prof_img, {x0, y0}, {x1, y0}, {0, 0, 0}, 1);
  cv::line(prof_img, {x0, y0}, {x0, y1}, {0, 0, 0}, 1);

  // 2) 刻度与标签
  int n_xt = 5;
  for (int i = 0; i <= n_xt; ++i) {
    float t_tick = -extend_range + 2 * extend_range * i / n_xt;
    int xt = x0 + int((t_tick + extend_range) / (2 * extend_range) * (x1 - x0));
    cv::line(prof_img, {xt, y0 - 5}, {xt, y0 + 5}, {0, 0, 0}, 1);
    cv::putText(prof_img, cv::format("%.2f", t_tick), {xt - 20, y0 + 25},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  }
  int n_yt = 5;
  for (int i = 0; i <= n_yt; ++i) {
    int v_tick = (i * 255) / n_yt;
    // 计算当前刻度对应的y坐标
    int yt = y0 - static_cast<int>((v_tick / 255.0f) * (y0 - y1));
    cv::line(prof_img, {x0 - 5, yt}, {x0 + 5, yt}, {0, 0, 0}, 1);
    // 显示整数刻度标签
    cv::putText(prof_img, cv::format("%d", v_tick), {x0 - 60, yt + 5},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);
  }

  // 3) 全采样点 & 连线（灰）
  for (int i = 0; i < N; ++i) {
    int xx = x0 + int((Ts[i] + extend_range) / (2 * extend_range) * (x1 - x0));
    int yy = y0 - static_cast<int>((Vs[i] / 255.0f) * (y0 - y1));
    cv::circle(prof_img, {xx, yy}, 3, {180, 180, 180}, -1);
    if (i > 0) {
      int xx0 =
          x0 + int((Ts[i - 1] + extend_range) / (2 * extend_range) * (x1 - x0));
      int yy0 = y0 - static_cast<int>((Vs[i - 1] / 255.0f) * (y0 - y1));
      cv::line(prof_img, {xx0, yy0}, {xx, yy}, {200, 200, 200}, 1);
    }
  }

  // 4) 四条垂直标线 & 图例
  struct Rst {
    float t;
    cv::Scalar c;
    const char *lbl;
  };
  std::vector<Rst> rst = {{res1, {0, 255, 0}, "res1"},
                          {res2, {255, 0, 0}, "res2"},
                          {res3, {255, 255, 0}, "res3"}};
  int lx = W - 180, ly = 80, dy = 30;
  for (auto &r : rst) {
    if (r.t == FLT_MAX)
      continue;
    int xx = x0 + int((r.t + extend_range) / (2 * extend_range) * (x1 - x0));
    cv::line(prof_img, {xx, y1}, {xx, y0}, r.c, 1);
    cv::putText(prof_img, cv::format("%s: %.3f", r.lbl, r.t), {lx, ly},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, r.c, 2);
    ly += dy;
  }

  cv::imshow("GrayProfile", prof_img);
  cv::waitKey(0);
#endif

  return res3;
}

// 行内连续质心计算
float LaserProcessor::computeRowCentroid(const cv::Mat &img, int y, float xL,
                                         float xR) {
  if (y < 0 || y >= img.rows)
    return 0.5f * (xL + xR);
  if (xL > xR)
    std::swap(xL, xR);

  xL = std::clamp(xL, 0.0f, (float)img.cols - 1.0f);
  xR = std::clamp(xR, 0.0f, (float)img.cols - 1.0f);
  if (xR - xL < 1e-3f)
    return 0.5f * (xL + xR);

  const float *row = img.ptr<float>(y);

  double A = 0.0, M = 0.0;

  int xiL = (int)std::floor(xL);
  int xiR = (int)std::floor(xR);

  // 左端分数像素
  if (xiL < img.cols - 1) {
    float I0 = row[xiL];
    float I1 = row[xiL + 1];
    float frac = xL - xiL;
    float IL = I0 + (I1 - I0) * frac;
    float IR = row[xiL + 1];
    double dx = (xiL + 1 - xL);
    double avg = 0.5 * (IL + IR);
    A += avg * dx;
    M += avg * ((xL + (xiL + 1)) * 0.5) * dx;
  }

  // 中间整像素段
  for (int i = xiL + 1; i < xiR; ++i) {
    float I0 = row[i];
    float I1 = row[i + 1];
    double dx = 1.0;
    double avg = 0.5 * (I0 + I1);
    A += avg * dx;
    M += avg * (i + 0.5) * dx;
  }

  // 右端分数像素
  if (xiR < img.cols - 1) {
    float I0 = row[xiR];
    float I1 = row[xiR + 1];
    float frac = xR - xiR;
    float IR = I0 + (I1 - I0) * frac;
    double dx = (xR - xiR);
    double avg = 0.5 * (I0 + IR);
    A += avg * dx;
    M += avg * ((xiR + xR) * 0.5) * dx;
  }

  if (A <= 1e-9)
    return 0.5f * (xL + xR);
  return (float)(M / A);
}

std::vector<cv::Point2f>
LaserProcessor::processCenters(const std::map<float, float> &orig) {
  std::vector<cv::Point2f> out;
  if (orig.empty() || precision_ <= 0)
    return out;

  // k -> x 映射，保证有序且去重
  std::map<int, float> k2x;

  // 阶段1：收集原始对齐点
  for (auto const &[y, x] : orig) {
    int k = int(std::lround(y / precision_));
    if (std::abs(y - k * precision_) < EPS_) {
      k2x.emplace(k, x);
    }
  }

  // 阶段2：相邻对插值
  auto prev = orig.begin();
  for (auto it = std::next(orig.begin()); it != orig.end(); ++it, ++prev) {
    float yA = prev->first, xA = prev->second;
    float yB = it->first, xB = it->second;
    if (yB <= yA + EPS_)
      continue;
    if ((yB - yA) > 1.5 + EPS_)
      continue;

    int k_start = int(std::ceil((yA + EPS_) / precision_));
    int k_end = int(std::floor((yB + EPS_) / precision_));

    for (int k = k_start; k <= k_end; ++k) {
      // 已有原始或已插值过，则跳过
      if (k2x.count(k))
        continue;
      float yi = k * precision_;
      float t = (yi - yA) / (yB - yA);
      float xi = xA + t * (xB - xA);
      k2x.emplace(k, xi);
    }
  }

  // 阶段3：按 k 有序输出
  out.reserve(k2x.size());
  for (auto const &[k, x] : k2x) {
    out.emplace_back(x, k * precision_);
  }
  return out;
}

std::pair<cv::Point2f, cv::Point2f>
LaserProcessor::getAxisEndpoints(const cv::RotatedRect &rect) {
  cv::Point2f vertices[4];
  rect.points(vertices);

  // 计算所有相邻边的长度并识别最短边
  float min_len = FLT_MAX;
  int short_edge_idx = 0;
  for (int i = 0; i < 4; ++i) {
    float len = cv::norm(vertices[i] - vertices[(i + 1) % 4]);
    if (len < min_len) {
      min_len = len;
      short_edge_idx = i;
    }
  }

  // 确定对应的两条短边（对边）
  int opposite_idx = (short_edge_idx + 2) % 4;

  // 计算两条短边的中点
  cv::Point2f mid1 =
      0.5f * (vertices[short_edge_idx] + vertices[(short_edge_idx + 1) % 4]);
  cv::Point2f mid2 =
      0.5f * (vertices[opposite_idx] + vertices[(opposite_idx + 1) % 4]);

  return std::make_pair(mid1, mid2);
}

std::vector<LaserLine>
LaserProcessor::extractLine(const std::vector<cv::RotatedRect> &rois,
                            const cv::Mat &rectify_img,
                            const cv::Mat &label_img, int img_idx) {
  std::vector<LaserLine> laser_lines;
  cv::Mat rectify_img_float;
  rectify_img.convertTo(rectify_img_float, CV_32F, 1.0f / 255.0f);

  // 保留全局二值化用于后续方向判断（非轮廓提取）
  cv::Mat bin;
  cv::threshold(rectify_img, bin, 80, 255, cv::THRESH_BINARY);

  // 可视化边缘
  cv::Mat direct_vis;
  cv::cvtColor(rectify_img, direct_vis, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < rois.size(); ++i) {
    const auto &roi = rois[i];
    const auto &roi_w = std::min(roi.size.width, roi.size.height);

    // 1. 获取ROI四个顶点
    cv::Point2f vertices[4];
    roi.points(vertices);

    // 2. 创建ROI区域mask
    cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8UC1);
    std::vector<cv::Point> roi_poly;
    for (int i = 0; i < 4; ++i) {
      roi_poly.push_back(
          cv::Point(cvRound(vertices[i].x), cvRound(vertices[i].y)));
    }
    std::vector<std::vector<cv::Point>> polys = {roi_poly};
    cv::fillPoly(mask, polys, cv::Scalar(255));

    // 3. 提取ROI区域内的标签图
    cv::Mat roi_labels(label_img.size(), CV_32SC1, cv::Scalar(0));
    label_img.copyTo(roi_labels, mask); // 仅复制ROI区域的标签

    // 4. 在ROI区域内选择面积最大的连通域
    int max_label = 0;
    double max_area = 0.0;

    // 统计ROI内各标签面积
    std::map<int, double> label_areas;
    for (int r = 0; r < roi_labels.rows; ++r) {
      const int *ptr = roi_labels.ptr<int>(r);
      for (int c = 0; c < roi_labels.cols; ++c) {
        int label_val = ptr[c];
        if (label_val > 1) { // 忽略背景(0)和未标记(1)
          label_areas[label_val] += 1.0;
        }
      }
    }

    // 寻找最大面积的标签
    for (const auto &[label, area] : label_areas) {
      if (area > max_area) {
        max_area = area;
        max_label = label;
      }
    }

    // 5. 创建最大连通域的二值图像并提取轮廓
    cv::Mat max_blob = (roi_labels == max_label);
    max_blob.convertTo(max_blob, CV_8U, 255); // 转换为8UC1格式

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(max_blob, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);

    // 6. 收集轮廓点（后续逻辑保持不变）
    if (contours.empty())
      throw std::logic_error("no contours found in max blob");
    auto max_it = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
          return a.size() < b.size();
        });

    // 7. 主轴投影区域
    std::vector<std::vector<cv::Point>> max_contour{*max_it};
    std::unordered_map<int, int> edge_maxY_map;
    std::unordered_map<int, int> edge_minY_map;
    auto [p1, p2] = getAxisEndpoints(roi);
    cv::Point2f axis = p2 - p1;
    float axis_len = cv::norm(axis);
    cv::Point2f axis_dir = axis / axis_len;
    float min_proj = 0.02f * axis_len;
    float max_proj = 0.98f * axis_len;
    for (const auto &pt : max_contour[0]) {
      cv::Point2f pt_f(pt.x, pt.y);
      cv::Point2f vec = pt_f - p1;
      float proj = vec.dot(axis_dir);
      if (proj >= min_proj && proj <= max_proj) {
        // 找Y最大值
        if (edge_maxY_map.find(pt.x) == edge_maxY_map.end())
          edge_maxY_map[pt.x] = pt.y;
        else if (pt.y > edge_maxY_map[pt.x])
          edge_maxY_map[pt.x] = pt.y;

        // 找Y最小值
        if (edge_minY_map.find(pt.x) == edge_minY_map.end())
          edge_minY_map[pt.x] = pt.y;
        else if (pt.y < edge_minY_map[pt.x])
          edge_minY_map[pt.x] = pt.y;
      }
    }

    // 8. 可视化边缘方向
    for (const auto &[x, y] : edge_minY_map)
      direct_vis.at<cv::Vec3b>(cv::Point(x, y)) =
          cv::Vec3b(0, 0, 255); // 红色表示上边沿
    for (const auto &[x, y] : edge_maxY_map)
      direct_vis.at<cv::Vec3b>(cv::Point(x, y)) =
          cv::Vec3b(255, 0, 0); // 蓝色表示下边沿

    cv::Mat laser_center_vis;
    cv::cvtColor(rectify_img, laser_center_vis, cv::COLOR_GRAY2BGR);
    std::map<float, float> orign_centers;
    std::unordered_map<int, bool> x_used;
    float max_search_range = -FLT_MAX;
    for (const auto &[x, y] : edge_minY_map) {
      if (x_used[x])
        continue;

      auto it = edge_maxY_map.find(x);
      if (it == edge_maxY_map.end())
        continue;
      float search_range = (it->second - y + 1);
      if (search_range < 3 || search_range > 17)
        continue;
      cv::Vec2f dir(0, 1);
      x_used[x] = true;
      if (search_range > max_search_range)
        max_search_range = search_range;

      // if (x == 1755 && y == 647)
      // puts("");

      // float t_peak = FLT_MAX;
      // if ((x >= 992 && x <= 1037) && (y >= 610 && y <= 640))
      // t_peak = findSymmetricCenter3(rectify_img_float, x, y, dir,
      // search_range);

      float t_peak =
          findSymmetricCenter3(rectify_img_float, x, y, dir, search_range);
      // float t_peak = findSymmetricCenter2(rectify_img_float, x, y, dir,
      // roi_w, 0);
      if (t_peak == FLT_MAX || t_peak <= 0)
        continue;
      float center_x = x + t_peak * dir[0], center_y = y + t_peak * dir[1];

      orign_centers[center_y] = center_x;

      // 可视化激光线中心点
      cv::Point2f center(center_x, center_y);
      laser_center_vis.at<cv::Vec3b>(cv::Point(center.x, center.y)) =
          cv::Vec3b(0, 255, 0); // 绿色表示中心点
    }

    // 9. 同一条线相邻中心点插值为整数
    cv::Mat new_centers_vis;
    cv::cvtColor(rectify_img, new_centers_vis, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point2f> new_centers = processCenters(orign_centers);
    for (const auto &p : new_centers)
      new_centers_vis.at<cv::Vec3b>(cv::Point(std::round(p.x), p.y)) =
          cv::Vec3b(0, 255, 0); // 绿色表示中心点

    // 10. 存储结果
    // const float roi_angle = roi.angle * CV_PI / 180.0f;
    // const float laser_width = max_search_range * sin(roi_angle);
    // const float sigma_val = convert_to_odd_number(laser_width / (2 *
    // std::sqrt(3.0f))); const bool h_is_long_edge = roi.size.height >=
    // roi.size.width; auto [dx_kernel, dy_kernel, ksize] =
    // computeGaussianDerivatives(sigma_val, roi_angle, h_is_long_edge); const
    // float laser_width = max_search_range; const float sigma_val =
    // convert_to_odd_number(laser_width / (2 * std::sqrt(3.0f))); auto
    // [dx_kernel, dy_kernel, ksize] = computeGaussianDerivatives(sigma_val, 90
    // * CV_PI / 180.0f, true); cv::Mat dx, dy; cv::filter2D(rectify_img_float,
    // dx, CV_32F, dx_kernel); cv::filter2D(rectify_img_float, dy, CV_32F,
    // dy_kernel);
    std::map<float, LaserPoint> best_points;
    for (const auto &p : new_centers) {
      // float gx = interpolateChannel(dx, p.x, p.y);
      // float gy = interpolateChannel(dy, p.x, p.y);
      LaserPoint lp(p.x, p.y);
      best_points[p.y] = lp;
    }
    LaserLine best_line;
    best_line.addPoints(best_points);
    laser_lines.emplace_back(best_line);
  }

  static int vis_img_cnt = 0;
  if (vis_img_cnt % 2 == 0)
    cv::imwrite(debug_img_dir /
                    ("direct_l_" + std::to_string(img_idx) + ".jpg"),
                direct_vis);
  else
    cv::imwrite(debug_img_dir /
                    ("direct_r_" + std::to_string(img_idx) + ".jpg"),
                direct_vis);
  vis_img_cnt++;

  return laser_lines;
}

std::vector<LaserLine> LaserProcessor::extractLine2(
    const cv::Mat &rectify_img,
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
    int img_idx) {
  std::vector<LaserLine> laser_lines;
  cv::Mat rectify_img_float;
  rectify_img.convertTo(rectify_img_float, CV_32F);

  // debug查看原始中心点分布
  int scale = 1;
  cv::Size expand_size(rectify_img.cols * scale, rectify_img.rows * scale);
  cv::Mat orign_centers_vis;
  cv::cvtColor(rectify_img, orign_centers_vis, cv::COLOR_GRAY2BGR);
  cv::resize(orign_centers_vis, orign_centers_vis, expand_size, 0, 0,
             cv::INTER_NEAREST);

  for (size_t i = 0; i < contours.size(); ++i) {
    const auto &edge_pair = contours[i];

    // 1. 求最小外接矩形 ROI
    std::vector<cv::Point> contour_points;
    for (const auto &p : edge_pair) {
      contour_points.push_back(p.first);
      contour_points.push_back(p.second);
    }
    cv::RotatedRect roi = cv::minAreaRect(contour_points);

    // 2. 主轴投影区域
    auto [p1, p2] = getAxisEndpoints(roi);
    cv::Point2f axis = p2 - p1;
    float axis_len = cv::norm(axis);
    cv::Point2f axis_dir = axis / axis_len;
    float min_proj = 0.05f * axis_len;
    float max_proj = 0.95f * axis_len;

    // 3. 激光线中心点提取
    std::map<float, float> orign_centers;
    float max_search_range = -FLT_MAX;
    for (const auto &p : edge_pair) {
      cv::Point2f vec = cv::Point2f(p.first.x, p.first.y) - p1;
      float proj = vec.dot(axis_dir);
      if (proj <= min_proj || proj >= max_proj)
        continue;

      float search_range = (p.second.x - p.first.x + 1);
      cv::Vec2f dir(1, 0);
      if (search_range > max_search_range)
        max_search_range = search_range;

      // if (x == 1755 && y == 647)
      // puts("");

      // float t_peak = FLT_MAX;
      // if ((p.first.x >= 850 && p.first.x <= 885) && (p.first.y >= 563 &&
      // p.first.y <= 563)) t_peak = findSymmetricCenter3(rectify_img_float,
      // p.first.x, p.first.y, dir, search_range);

      // float t_peak = findSymmetricCenter3(rectify_img_float, p.first.x,
      // p.first.y, dir, search_range);
      float t_peak = findSymmetricCenter4(rectify_img_float, p.first.x,
                                          p.first.y, dir, search_range);
      if (t_peak == FLT_MAX)
        continue;
      float center_x = p.first.x + t_peak * dir[0],
            center_y = p.first.y + t_peak * dir[1];

      orign_centers[center_y] = center_x;
    }
    if (orign_centers.empty())
      continue;

    // debug查看原始中心点分布
    for (const auto &[y, x] : orign_centers)
      orign_centers_vis.at<cv::Vec3b>(static_cast<int>(y * scale),
                                      static_cast<int>(x * scale)) =
          cv::Vec3b(255, 0, 255);

    // 4. 同一条线相邻中心点插值为整数
    std::vector<cv::Point2f> new_centers = processCenters(orign_centers);

    // 5. 存储结果
    // const float laser_width = max_search_range;
    // const float sigma_val = convert_to_odd_number(laser_width / (2 *
    // std::sqrt(3.0f))); auto [dx_kernel, dy_kernel, ksize] =
    // computeGaussianDerivatives(sigma_val); cv::Mat dx, dy;
    // cv::filter2D(rectify_img_float, dx, CV_32F, dx_kernel);
    // cv::filter2D(rectify_img_float, dy, CV_32F, dy_kernel);
    std::map<float, LaserPoint> best_points;
    for (const auto &p : new_centers) {
      // float gx = interpolateChannel(dx, p.x, p.y);
      // float gy = interpolateChannel(dy, p.x, p.y);
      best_points[p.y] = {p.x, p.y};
    }
    LaserLine best_line;
    best_line.addPoints(best_points);
    laser_lines.emplace_back(best_line);
  }

  // cv::imwrite(debug_img_dir / ("laser_orign_img" + std::to_string(img_idx) +
  // ".bmp"), orign_centers_vis);

  return laser_lines;
}

std::vector<LaserLine2> LaserProcessor::extractLine3(
    const cv::Mat &rectify_img,
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> contours,
    int img_idx) {
  std::vector<LaserLine2> laser_lines;
  cv::Mat rectify_img_float;
  rectify_img.convertTo(rectify_img_float, CV_32F);

  // debug查看原始中心点分布
  int scale = 10;
  cv::Size expand_size(rectify_img.cols * scale, rectify_img.rows * scale);
  cv::Mat orign_centers_vis;
  cv::cvtColor(rectify_img, orign_centers_vis, cv::COLOR_GRAY2BGR);
  cv::resize(orign_centers_vis, orign_centers_vis, expand_size, 0, 0,
             cv::INTER_NEAREST);

  for (size_t i = 0; i < contours.size(); ++i) {
    const auto &edge_pair = contours[i];

    // 1. 求最小外接矩形 ROI
    std::vector<cv::Point> contour_points;
    for (const auto &p : edge_pair) {
      contour_points.push_back(p.first);
      contour_points.push_back(p.second);
    }
    cv::RotatedRect roi = cv::minAreaRect(contour_points);

    // 2. 主轴投影区域
    auto [p1, p2] = getAxisEndpoints(roi);
    cv::Point2f axis = p2 - p1;
    float axis_len = cv::norm(axis);
    cv::Point2f axis_dir = axis / axis_len;
    float min_proj = 0.05f * axis_len;
    float max_proj = 0.95f * axis_len;

    // 3. 激光线中心点提取
    std::map<float, float> orign_centers;
    float max_search_range = -FLT_MAX;
    for (const auto &p : edge_pair) {
      cv::Point2f vec = cv::Point2f(p.first.x, p.first.y) - p1;
      float proj = vec.dot(axis_dir);
      if (proj <= min_proj || proj >= max_proj)
        continue;

      float search_range = (p.second.x - p.first.x + 1);
      cv::Vec2f dir(1, 0);
      if (search_range > max_search_range)
        max_search_range = search_range;

      // if (x == 1755 && y == 647)
      // puts("");

      // float t_peak = FLT_MAX;
      // if ((p.first.x >= 850 && p.first.x <= 885) && (p.first.y >= 563 &&
      // p.first.y <= 563)) t_peak = findSymmetricCenter3(rectify_img_float,
      // p.first.x, p.first.y, dir, search_range);

      // float t_peak = findSymmetricCenter3(rectify_img_float, p.first.x,
      // p.first.y, dir, search_range);
      float t_peak = findSymmetricCenter4(rectify_img_float, p.first.x,
                                          p.first.y, dir, search_range);
      if (t_peak == FLT_MAX)
        continue;
      float center_x = p.first.x + t_peak * dir[0],
            center_y = p.first.y + t_peak * dir[1];

      orign_centers[center_y] = center_x;
    }
    if (orign_centers.empty())
      continue;

    // debug查看原始中心点分布
    // for (const auto& [y, x] : orign_centers)
    //     orign_centers_vis.at<cv::Vec3b>(static_cast<int>(y * scale),
    //     static_cast<int>(x * scale)) = cv::Vec3b(255, 0, 255);

    // 4. 同一条线相邻中心点插值为整数
    std::vector<cv::Point2f> new_centers = processCenters(orign_centers);

    // 5. 结果封装
    std::vector<LaserPoint> best_points;
    std::vector<float> y_coords;
    best_points.reserve(new_centers.size());
    y_coords.reserve(new_centers.size());

    for (const auto &p : new_centers) {
      best_points.emplace_back(p.x, p.y);
      y_coords.emplace_back(p.y);
    }
    LaserLine2 best_line;
    best_line.addPoints(std::move(best_points), std::move(y_coords));
    laser_lines.emplace_back(std::move(best_line));
  }

  // cv::imwrite(debug_img_dir / ("laser_orign_img" + std::to_string(img_idx) +
  // ".bmp"), orign_centers_vis);

  return laser_lines;
}

/***********************************************************************************
 */

/************************************** Quad Surface Operation
 * **************************************/
std::vector<cv::Point3f>
LaserProcessor::findIntersection(const cv::Point3f &point,
                                 const cv::Point3f &normal,
                                 const cv::Mat &Coeff6x1) {
  float a = Coeff6x1.at<float>(0);
  float b = Coeff6x1.at<float>(1);
  float c = Coeff6x1.at<float>(2);
  float d = Coeff6x1.at<float>(3);
  float e = Coeff6x1.at<float>(4);
  float f = Coeff6x1.at<float>(5);

  std::vector<cv::Point3f> intersections;

  // 计算二次方程的系数
  float A = a * normal.y * normal.y + b * normal.y * normal.z +
            c * normal.z * normal.z;

  float B = 2 * a * point.y * normal.y +
            b * (point.y * normal.z + point.z * normal.y) +
            2 * c * point.z * normal.z + d * normal.y + e * normal.z - normal.x;

  float C = a * point.y * point.y + b * point.y * point.z +
            c * point.z * point.z + d * point.y + e * point.z + f - point.x;

  // 解二次方程 A*t² + B*t + C = 0
  float discriminant = B * B - 4 * A * C;

  if (std::abs(A) < 1e-6) { // 线性情况
    if (std::abs(B) > 1e-6) {
      float t = -C / B;
      intersections.push_back(point + t * normal);
    }
  } else if (discriminant > 0) { // 两个实数解
    float sqrt_discriminant = std::sqrt(discriminant);
    float t1 = (-B + sqrt_discriminant) / (2 * A);
    float t2 = (-B - sqrt_discriminant) / (2 * A);
    intersections.push_back(point + t1 * normal);
    intersections.push_back(point + t2 * normal);
  } else if (std::abs(discriminant) < 1e-6) { // 一个实数解
    float t = -B / (2 * A);
    intersections.push_back(point + t * normal);
  }
  // 判别式小于0无实数解，返回空vector

  return intersections;
}

double LaserProcessor::evaluateQuadSurf(const cv::Mat &Coeff6x1,
                                        const cv::Point3f &p) {
  float a = Coeff6x1.at<float>(0);
  float b = Coeff6x1.at<float>(1);
  float c = Coeff6x1.at<float>(2);
  float d = Coeff6x1.at<float>(3);
  float e = Coeff6x1.at<float>(4);
  float f = Coeff6x1.at<float>(5);
  float x_eval =
      a * p.y * p.y + b * p.y * p.z + c * p.z * p.z + d * p.y + e * p.z + f;
  float res = abs(x_eval - p.x);

  cv::Point3f norm(1, -2.0 * a * p.y - b * p.z - d,
                   -b * p.y - 2.0 * c * p.z - e);
  auto vec_intersec = findIntersection(p, norm, Coeff6x1);
  for (const auto &p_inter : vec_intersec) {
    float res_i = cv::norm(p_inter - p);
    res = MIN(res, res_i);
  }
  return res;
}

double
LaserProcessor::evaluateQuadSurf(const cv::Mat &Coeff6x1,
                                 const std::vector<cv::Point3f> &points) {
  std::vector<float> vec_res;
  for (const auto &p : points)
    vec_res.push_back(evaluateQuadSurf(Coeff6x1, p));
  float sum_of_squares = 0.0f;
  for (float value : vec_res) {
    sum_of_squares += std::pow(value, 2.0f); // 或者 value * value
  }
  float mean_square = sum_of_squares / (float)vec_res.size();

  // 3. 计算平方根
  float rmse = std::sqrt(mean_square);
  return rmse;
}

/***************************************************************************************
 */

void LaserProcessor::LabelColor(const cv::Mat &labelImg,
                                cv::Mat &colorLabelImg) {
  int num = 0;
  if (labelImg.empty() || labelImg.type() != CV_32SC1) {
    return;
  }

  std::map<int, cv::Scalar> colors;

  int rows = labelImg.rows;
  int cols = labelImg.cols;

  colorLabelImg.release();
  colorLabelImg.create(rows, cols, CV_8UC3);
  colorLabelImg = cv::Scalar::all(0);

  for (int i = 0; i < rows; i++) {
    const int *data_src = (int *)labelImg.ptr<int>(i);
    uchar *data_dst = colorLabelImg.ptr<uchar>(i);
    for (int j = 0; j < cols; j++) {
      int pixelValue = data_src[j];
      if (pixelValue > 1) {
        if (colors.count(pixelValue) <= 0) {
          colors[pixelValue] = GetRandomColor();
          num++;
        }

        cv::Scalar color = colors[pixelValue];
        *data_dst++ = color[0];
        *data_dst++ = color[1];
        *data_dst++ = color[2];
      } else {
        data_dst++;
        data_dst++;
        data_dst++;
      }
    }
  }
}

void LaserProcessor::Two_PassNew(const cv::Mat &img, cv::Mat &labImg) {
  cv::Mat bwImg;
  cv::threshold(img, bwImg, 100, 255, cv::THRESH_BINARY);
  assert(bwImg.type() == CV_8UC1);
  labImg.create(bwImg.size(), CV_32SC1); // bwImg.convertTo( labImg, CV_32SC1 );
  labImg = cv::Scalar(0);
  labImg.setTo(cv::Scalar(1), bwImg);
  assert(labImg.isContinuous());
  const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
  int label = 1;
  std::vector<int> labelSet;
  labelSet.push_back(0);
  labelSet.push_back(1);
  // the first pass
  int *data_prev =
      (int *)labImg.data; // 0-th row : int* data_prev = labImg.ptr<int>(i-1);
  int *data_cur =
      (int *)(labImg.data +
              labImg.step); // 1-st row : int* data_cur = labImg.ptr<int>(i);
  for (int i = 1; i < Rows; i++) {
    data_cur++;
    data_prev++;
    for (int j = 1; j < Cols; j++, data_cur++, data_prev++) {
      if (*data_cur != 1)
        continue;
      int left = *(data_cur - 1);
      int up = *data_prev;
      int neighborLabels[2];
      int cnt = 0;
      if (left > 1)
        neighborLabels[cnt++] = left;
      if (up > 1)
        neighborLabels[cnt++] = up;
      if (!cnt) {
        labelSet.push_back(++label);
        labelSet[label] = label;
        *data_cur = label;
        continue;
      }
      int smallestLabel = neighborLabels[0];
      if (cnt == 2 && neighborLabels[1] < smallestLabel)
        smallestLabel = neighborLabels[1];
      *data_cur = smallestLabel;
      // 保存最小等价表
      for (int k = 0; k < cnt; k++) {
        int tempLabel = neighborLabels[k];
        int &oldSmallestLabel =
            labelSet[tempLabel]; // 这里的&不是取地址符号,而是引用符号
        if (oldSmallestLabel > smallestLabel) {
          labelSet[oldSmallestLabel] = smallestLabel;
          oldSmallestLabel = smallestLabel;
        } else if (oldSmallestLabel < smallestLabel)
          labelSet[smallestLabel] = oldSmallestLabel;
      }
    }
    data_cur++;
    data_prev++;
  }
  // 更新等价队列表,将最小标号给重复区域
  for (size_t i = 2; i < labelSet.size(); i++) {
    int curLabel = labelSet[i];
    int prelabel = labelSet[curLabel];
    while (prelabel != curLabel) {
      curLabel = prelabel;
      prelabel = labelSet[prelabel];
    }
    labelSet[i] = curLabel;
  }
  // second pass
  data_cur = (int *)labImg.data;
  for (int i = 0; i < Rows; i++) {
    for (int j = 0; j < bwImg.cols - 1; j++, data_cur++)
      *data_cur = labelSet[*data_cur];
    data_cur++;
  }
}

cv::Scalar LaserProcessor::GetRandomColor() {
  uchar r = 255 * (rand() / (1.0 + RAND_MAX));
  uchar g = 255 * (rand() / (1.0 + RAND_MAX));
  uchar b = 255 * (rand() / (1.0 + RAND_MAX));
  return cv::Scalar(b, g, r);
}

/************************************** ComputeScore
 * ***************************************/

float LaserProcessor::computeCompScore(float avgDist, float coverage, float wD,
                                       float wC) {
  return wD * avgDist + wC * (1.0f - coverage);
}

float LaserProcessor::computeEnhancedScore(
    const std::vector<std::pair<float, float>> &distance_pairs,
    int pts_repro_cnt, float &coverage, float &std_dev) {

  if (distance_pairs.empty() || pts_repro_cnt <= 0)
    return FLT_MAX;

  // 提取距离
  std::vector<float> distances;
  distances.reserve(distance_pairs.size());
  for (const auto &[y, d] : distance_pairs)
    distances.push_back(d);

  // 1. 距离中位数
  std::nth_element(distances.begin(), distances.begin() + distances.size() / 2,
                   distances.end());
  float median_dist = distances[distances.size() / 2];

  // 2. 标准差（表示距离分布是否均匀）
  float sum_dist = std::accumulate(distances.begin(), distances.end(), 0.0f);
  float mean = sum_dist / distances.size();
  float variance = 0.0f;
  for (float d : distances) {
    float diff = d - mean;
    variance += diff * diff;
  }
  variance /= distances.size();
  std_dev = std::sqrt(variance);

  // 3. 覆盖率
  coverage = static_cast<float>(distance_pairs.size()) / pts_repro_cnt;

  // 4. 线长惩罚（惩罚过短的线）
  float short_penalty = 0.0f;
  const int min_len_thresh = 250;
  if (pts_repro_cnt < min_len_thresh) {
    short_penalty = (min_len_thresh - pts_repro_cnt) * 0.15f; // 可调节
  }

  // 5. 综合评分（越小越好）
  // 加权组合各项评分
  const float w_median = 0.25f; // 中位距离权重
  const float w_stddev =
      (int)distance_pairs.size() > min_len_thresh ? 1.5 : 2.5; // 距离一致性权重
  const float w_coverage = 23.0f;                              // 覆盖率惩罚权重
  const float w_short = 1.0f;                                  // 线长惩罚权重

  float score = w_median * median_dist + w_stddev * std_dev +
                w_coverage * (0.9f - coverage) + w_short * short_penalty;

  // printf("median_dist: %.2f / std_dev: %.2f / cov_pen: %.2f / match_p_cnt: %d
  // / short_pen: %.2f / score: %.2f\n",
  //         median_dist, std_dev, (0.9f - coverage),
  //         (int)distance_pairs.size(), short_penalty, score);

  return score;
}

float LaserProcessor::computeEnhancedScoreV2(
    const std::vector<std::pair<float, float>> &distance_pairs,
    int left_point_count, int right_point_count, float &coverage,
    float &std_dev) {

  if (distance_pairs.empty())
    return FLT_MAX;

  // 提取距离
  std::vector<float> distances;
  distances.reserve(distance_pairs.size());
  for (const auto &[y, d] : distance_pairs)
    distances.push_back(d);

  // 1. 距离中位数
  std::nth_element(distances.begin(), distances.begin() + distances.size() / 2,
                   distances.end());
  float median_dist = distances[distances.size() / 2];

  // 2. MAD (Median Absolute Deviation) - 更鲁棒的离散度度量
  std::vector<float> abs_deviations;
  abs_deviations.reserve(distances.size());
  for (float d : distances) {
    abs_deviations.push_back(std::abs(d - median_dist));
  }
  std::nth_element(abs_deviations.begin(),
                   abs_deviations.begin() + abs_deviations.size() / 2,
                   abs_deviations.end());
  float mad = abs_deviations[abs_deviations.size() / 2];

  // 保持std_dev输出兼容性
  std_dev = mad * 1.4826f; // MAD到标准差的近似转换系数

  // 3. 改进的覆盖率计算
  int matched = static_cast<int>(distance_pairs.size());
  int norm_len = std::min(left_point_count, right_point_count);
  float coverage_norm = static_cast<float>(matched) / norm_len;
  coverage = coverage_norm; // 输出兼容性

  // 4. 长度一致性惩罚
  float length_ratio = static_cast<float>(left_point_count) / right_point_count;
  float length_penalty = std::abs(1.0f - length_ratio) * 2.0f;

  // 5. 短匹配惩罚（基于归一化覆盖率）
  float short_penalty =
      std::max(0.0f, std::min(1.0f, 1.0f - coverage_norm)) * 5.0f;

  // 6. 统一评分函数
  const float alpha = 0.3f;   // 几何精度权重
  const float beta = 1.0f;    // 几何一致性权重
  const float gamma = 6.0f;  // 覆盖率惩罚权重
  const float delta = 1.2f;   // 短匹配惩罚权重
  const float epsilon = 1.0f; // 长度一致性惩罚权重

  float score = alpha * median_dist + beta * mad +
                gamma * (1.0f - coverage_norm) + delta * short_penalty +
                epsilon * length_penalty;

  return score;
}

/***************************************************************************************
 */

/************************************** Match Four
 * ***************************************/
struct MatchCandidate {
  int l_idx;
  int plane_idx;
  int r_idx;
  float score;
  float avg_dist;
  float coverage;
};

std::vector<std::tuple<int, int, int>>
LaserProcessor::match4(const std::vector<std::map<float, float>> &sample_points,
                       const std::vector<LaserLine> &laser_r,
                       const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto planes = ConfigManager::getInstance().getQuadSurfaces();
  const double fx_l = calib.P[0].at<double>(0, 0),
               fy_l = calib.P[0].at<double>(1, 1);
  const double cx_l = calib.P[0].at<double>(0, 2),
               cy_l = calib.P[0].at<double>(1, 2);
  const double fx_r = calib.P[1].at<double>(0, 0),
               fy_r = calib.P[1].at<double>(1, 1);
  const double cx_r = calib.P[1].at<double>(0, 2),
               cy_r = calib.P[1].at<double>(1, 2);
  const double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  int L = sample_points.size();
  std::vector<std::vector<MatchCandidate>> allCands(L);
  std::set<int> used_r;
  std::vector<bool> locked(L, false);
  std::vector<std::tuple<int, int, int>> match_results;

  // 收集候选
  for (int l = 0; l < L; ++l) {
    const auto &pts = sample_points[l];
    for (int p = 0; p < (int)planes.size(); ++p) {
      int inCnt = 0;
      std::map<int, int> hits;
      std::map<int, std::pair<float, int>> acc;
      auto &coef = planes[p].coefficients;
      for (auto [y, x] : pts) {
        cv::Point3f ray((x - cx_l) / fx_l, (y - cy_l) / fy_l, 1.0f);
        ray *= 1.0f / cv::norm(ray);
        auto ips = findIntersection({0, 0, 0}, ray, coef);
        if (ips.empty())
          continue;
        cv::Point3f pt3;
        bool ok = false;
        for (auto &q : ips)
          if (q.z > 100 && q.z < 1200) {
            pt3 = q;
            ok = true;
            break;
          }
        if (!ok)
          continue;
        cv::Point3f pr(pt3.x - baseline, pt3.y, pt3.z);
        float xr = fx_r * pr.x / pr.z + cx_r;
        int yr = cvRound(fy_r * pr.y / pr.z + cy_r);
        if (xr < 0 || xr >= rectify_r.cols || yr < 0 || yr >= rectify_r.rows)
          continue;
        ++inCnt;
        for (int r = 0; r < (int)laser_r.size(); ++r) {
          if (used_r.count(r))
            continue;
          auto it = laser_r[r].points.find(yr);
          if (it == laser_r[r].points.end())
            continue;
          float d = hypot(it->second.x - xr, it->second.y - yr);
          hits[r]++;
          acc[r].first += d;
          acc[r].second++;
        }
      }
      if (!inCnt)
        continue;
      for (auto &[r, c] : hits) {
        auto pr = acc[r];
        float avg = pr.first / pr.second;
        float cov = float(c) / inCnt;
        float comp = computeCompScore(avg, cov);
        allCands[l].push_back({l, p, r, comp, avg, cov});
      }
    }
  }

  // 第一次统一可视化
#ifdef DEBUG_PLANE_MATCH
  cv::Mat vis_global;
  cv::hconcat(rectify_l, rectify_r, vis_global);
  if (vis_global.channels() == 1)
    cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
  int off = rectify_l.cols;
  cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
  cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

  for (int l = 0; l < L; ++l) {
    if (allCands[l].empty())
      continue;

#ifdef DEBUG_PLANE_MATCH
    cv::Mat vis = vis_global.clone();
    // 左激光线点
    for (auto [y, x] : sample_points[l])
      cv::circle(vis, cv::Point2f(x, y), 1.5, cv::Scalar(0, 255, 0), -1);

    // 左激光线ID
    int l_middle = sample_points[l].size() / 2;
    auto l_mid_it = sample_points[l].begin();
    std::advance(l_mid_it, l_middle);
    cv::putText(vis, "L" + std::to_string(l),
                cv::Point2f(l_mid_it->second, l_mid_it->first),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
#endif
    // 按 plane_idx 提取最佳候选
    std::map<int, MatchCandidate> bestOfPlane;
    for (auto &m : allCands[l]) {
      auto &b = bestOfPlane[m.plane_idx];
      if (b.score == 0 || m.score < b.score)
        b = m;
    }
    // 全局最佳
    MatchCandidate globalBest = bestOfPlane.begin()->second;
    for (auto &kv : bestOfPlane)
      if (kv.second.score < globalBest.score)
        globalBest = kv.second;

#ifdef DEBUG_PLANE_MATCH
    // 重投影点（红）
    for (auto &kv : bestOfPlane) {
      auto m = kv.second;
      auto &coef = planes[m.plane_idx].coefficients;
      for (auto [y, x] : sample_points[l]) {
        cv::Point3f ray((x - cx_l) / fx_l, (y - cy_l) / fy_l, 1.0f);
        ray *= 1.0f / cv::norm(ray);
        auto ips = findIntersection({0, 0, 0}, ray, coef);
        if (ips.empty())
          continue;
        cv::Point3f pt3;
        bool ok = false;
        for (auto &q : ips)
          if (q.z > 100 && q.z < 1200) {
            pt3 = q;
            ok = true;
            break;
          }
        if (!ok)
          continue;
        cv::Point3f pr(pt3.x - baseline, pt3.y, pt3.z);
        float xr = fx_r * pr.x / pr.z + cx_r;
        int yr = cvRound(fy_r * pr.y / pr.z + cy_r);
        if (xr < 0 || xr >= rectify_r.cols || yr < 0 || yr >= rectify_r.rows)
          continue;
        cv::circle(vis, cv::Point2f(xr + off, yr), 1.5, cv::Scalar(0, 0, 255),
                   -1);
      }
    }

    // 绘制右激光线点和ID
    for (auto &kv : bestOfPlane) {
      auto m = kv.second;
      bool isGlobal = (m.r_idx == globalBest.r_idx);
      cv::Scalar col = used_r.count(m.r_idx) ? cv::Scalar(0, 255, 255)
                       : isGlobal            ? cv::Scalar(128, 0, 128)
                                             : cv::Scalar(0, 180, 0);
      for (auto &[y, pt] : laser_r[m.r_idx].points)
        cv::circle(vis, cv::Point2f(pt.x + off, y), 1.5, col, -1);

      if (l == 4 && m.r_idx == 3) {
        auto it_begin = sample_points[l].begin();
        auto it_end = sample_points[l].end();

        // 在begin点的y坐标处画水平线
        cv::line(vis, cv::Point(0, it_begin->first),
                 cv::Point(vis.cols - 1, it_begin->first),
                 cv::Scalar(0, 0, 255), 2);

        // 在end点的y坐标处画水平线
        auto it_last = std::prev(it_end);
        cv::line(vis, cv::Point(0, it_last->first),
                 cv::Point(vis.cols - 1, it_last->first), cv::Scalar(255, 0, 0),
                 2);
      }

      auto it = laser_r[m.r_idx].points.begin();
      int r_middle = laser_r[m.r_idx].points.size() / 2;
      std::advance(it, r_middle);
      if (it == laser_r[m.r_idx].points.end())
        it = std::prev(laser_r[m.r_idx].points.end());
      cv::putText(vis, "R" + std::to_string(m.r_idx),
                  cv::Point2f(it->second.x + off, it->first),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2);
    }

    // 状态文本
    char buf[128];
    snprintf(buf, sizeof(buf), "Round1: L%d->R%d (P %d / S %.2f)",
             globalBest.l_idx, globalBest.r_idx, globalBest.plane_idx,
             globalBest.score);
    cv::putText(vis, buf, {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.3,
                cv::Scalar(0, 255, 255), 2);
    cv::imshow("vis_img", vis);
    cv::waitKey(0);
#endif
  }

  // 多轮严格锁定
  bool prog = true;
  while (prog) {
    prog = false;
    for (int l = 0; l < L; ++l)
      if (!locked[l]) {
        std::vector<MatchCandidate> av;
        for (auto &m : allCands[l])
          if (!used_r.count(m.r_idx))
            av.push_back(m);
        if (av.empty())
          continue;
        std::sort(av.begin(), av.end(),
                  [](auto &a, auto &b) { return a.score < b.score; });
        float b0 = av[0].score, b1 = av.size() > 1 ? av[1].score : FLT_MAX;
        if (b0 <= 10.0f && (av.size() == 1 || b1 - b0 >= 10.0f)) {
          locked[l] = true;
          used_r.insert(av[0].r_idx);
          match_results.emplace_back(l, av[0].plane_idx, av[0].r_idx);
          prog = true;
          printf("   [锁定匹配] 左%d 平面%d→右%d 平均距离:%.2f 与次优差距:%.2f "
                 "综合得分:%.2f 覆盖率:%.2f%%\n",
                 l, av[0].plane_idx, av[0].r_idx, av[0].avg_dist, b1 - b0,
                 av[0].score, av[0].coverage * 100.0f);
        }
      }
  }

#ifdef DEBUG_PLANE_MATCH
  // 全局可视化锁定结果
  static const std::vector<cv::Scalar> palette30 = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},  {255, 255, 0}, {255, 0, 255},
      {0, 255, 255}, {128, 0, 0},   {0, 128, 0},  {0, 0, 128},   {128, 128, 0},
      {128, 0, 128}, {0, 128, 128}, {64, 0, 128}, {128, 64, 0},  {0, 128, 64},
      {64, 128, 0},  {0, 64, 128},  {128, 0, 64}, {192, 192, 0}, {192, 0, 192},
      {64, 255, 0},  {255, 64, 0},  {0, 64, 255}, {0, 255, 64},  {255, 0, 64},
      {64, 0, 255},  {192, 0, 64},  {64, 192, 0}, {0, 192, 64},  {64, 0, 192}};
  // 标注左线ID
  for (int l = 0; l < L; ++l) {
    int mid = sample_points[l].size() / 2;
    auto it = sample_points[l].begin();
    std::advance(it, mid);
    cv::putText(vis_global, "L" + std::to_string(l),
                cv::Point2f(it->second, it->first), cv::FONT_HERSHEY_SIMPLEX,
                1.5, cv::Scalar(0, 255, 0), 2);
  }
  // 标注右线ID
  for (int r = 0; r < (int)laser_r.size(); ++r) {
    int mid = laser_r[r].points.size() / 2;
    auto it = laser_r[r].points.begin();
    std::advance(it, mid);
    cv::putText(vis_global, "R" + std::to_string(r),
                cv::Point2f(it->second.x + off, it->first),
                cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 255, 255), 2);
  }

  // 仅为配对线条着色（不绘制连线）
  for (int idx = 0; idx < match_results.size(); ++idx) {
    auto [l, p, r] = match_results[idx];
    cv::Scalar clr = palette30[idx % palette30.size()];
    // 左线点着色
    for (auto &[y, x] : sample_points[l]) {
      cv::circle(vis_global, cv::Point(cvRound(x), y), 3, clr, -1);
    }
    // 右线点着色
    for (auto &[y, pt] : laser_r[r].points) {
      cv::circle(vis_global, cv::Point(cvRound(pt.x) + off, y), 3, clr, -1);
    }
  }
  cv::putText(vis_global, "vis_img", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.3,
              cv::Scalar(255, 0, 255), 2);
  cv::imshow("vis_img", vis_global);
  cv::waitKey(0);
  cv::destroyWindow("vis_img");
#endif

  return match_results;
}
/***************************************************************************************
 */

// 匈牙利算法（最小化版，方阵 n x n）
// 输入：cost 矩阵 n x n（double），输出 matchL: size n，matchL[i] = j 表示行 i
// 被分配到列 j，若未分配则 = -1 实现参考标准 Kuhn-Munkres (O(n^3))
static void hungarian1(const std::vector<std::vector<double>> &cost,
                       std::vector<int> &matchL) {
  const double INF = 1e100;
  int n = (int)cost.size();
  matchL.assign(n, -1);
  std::vector<int> u(n + 1, 0), v(n + 1, 0), p(n + 1, 0), way(n + 1, 0);
  // 1-based indexing in typical implementation
  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, INF);
    std::vector<char> used(n + 1, false);
    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      double delta = INF;
      for (int j = 1; j <= n; j++)
        if (!used[j]) {
          double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      for (int j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else
          minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] != 0);
    // augmenting
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  // p[j] = i matched to j
  std::vector<int> ans(n, -1);
  for (int j = 1; j <= n; j++) {
    if (p[j] != 0 && p[j] <= n && j <= n) {
      ans[p[j] - 1] = j - 1;
    }
  }
  matchL = ans;
}

static void hungarian2(const std::vector<std::vector<float>> &cost,
                       std::vector<int> &matchL) {
  int n = (int)cost.size();
  matchL.assign(n, -1);

  std::vector<double> u(n + 1, 0), v(n + 1, 0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);

  for (int i = 1; i <= n; i++) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, 1e18);
    std::vector<char> used(n + 1, false);

    do {
      used[j0] = true;
      int i0 = p[j0], j1 = 0;
      double delta = 1e18;

      for (int j = 1; j <= n; j++)
        if (!used[j]) {
          double cur = (double)cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }

      for (int j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> ans(n, -1);
  for (int j = 1; j <= n; j++) {
    if (p[j] != 0 && p[j] <= n && j <= n) {
      ans[p[j] - 1] = j - 1;
    }
  }
  matchL = ans;
}

/************************************** Match
 * **************************************/

static void merge_intervals_vec(std::vector<Interval> &ivs) {
  if (ivs.empty())
    return;
  std::sort(ivs.begin(), ivs.end(), [](const Interval &a, const Interval &b) {
    return a.y_start < b.y_start;
  });
  std::vector<Interval> out;
  out.reserve(ivs.size());
  Interval cur = ivs[0];
  for (size_t i = 1; i < ivs.size(); ++i) {
    if (ivs[i].y_start <= cur.y_end + 1e-3f) {
      cur.y_end = std::max(cur.y_end, ivs[i].y_end);
      cur.count += ivs[i].count;
    } else {
      out.push_back(cur);
      cur = ivs[i];
    }
  }
  out.push_back(cur);
  ivs.swap(out);
}

std::vector<IntervalMatch>
LaserProcessor::match5(const std::vector<std::map<float, float>> &sample_points,
                       const std::vector<LaserLine> &laser_r,
                       const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
  // 1. 相机与阈值初始化
  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto planes = ConfigManager::getInstance().getQuadSurfaces();
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  constexpr float D_thresh_ = 20.0f;
  constexpr float S_thresh_ = 20.0f;
  constexpr float Delta_thresh = 3.0f;
  constexpr int MIN_LEN_ =
      80; // 存在问题：大一些会过滤掉正确导致匹配错误，小一些容易配错

  int L = (int)sample_points.size();
  int R = (int)laser_r.size();

  // 结果与锁定区间
  std::vector<IntervalMatch> final_matches;
  std::vector<std::vector<Interval>> locked_l(L), locked_r(R);

  // lambda: 合并已锁定区间
  auto merge_intervals = [&](std::vector<Interval> &v) {
    if (v.empty())
      return;
    std::sort(v.begin(), v.end(),
              [](auto &a, auto &b) { return a.y_start < b.y_start; });
    std::vector<Interval> merged;
    Interval cur = v[0];
    for (int i = 1; i < (int)v.size(); ++i) {
      auto &n = v[i];
      if (n.y_start <= cur.y_end + EPS_) {
        cur.y_end = std::max(cur.y_end, n.y_end);
        cur.count += n.count;
      } else {
        merged.push_back(cur);
        cur = n;
      }
    }
    merged.push_back(cur);
    v.swap(merged);
  };

#ifdef DEBUG_PLANE_MATCH_FINAL
  // 全局可视化底图
  cv::Mat vis_global;
  cv::hconcat(rectify_l, rectify_r, vis_global);
  if (vis_global.channels() == 1)
    cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
  int off = rectify_l.cols;
  static const cv::Scalar proc_interval_col(0, 255, 255); // 候选区间颜色
  static const cv::Scalar r_laser_col(255, 0, 0);         // 整条右线颜色
  cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
  cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

  // --- 阶段：唯一候选先轮完
  bool progress = true;
  while (progress) {
    progress = false;
    for (int l = 0; l < L; ++l) {
      const auto &pts = sample_points[l];
      int repro_cnt_total = 0;
      std::vector<IntervalMatch> cands;

      // 针对每个平面+右线构建支持集合
      for (int p = 0; p < (int)planes.size(); ++p) {
        int repro_cnt = 0;
        const auto &coef = planes[p].coefficients;
        std::map<int, std::vector<std::pair<float, float>>>
            support; // r_idx->(y,distance)
        // if (l == 5 && p == 1)
        //     puts("");
        for (auto [y_f, x_f] : pts) {
          // 左侧已锁区跳过
          bool skipL = false;
          for (auto &iv : locked_l[l]) {
            if (y_f >= iv.y_start - EPS_ && y_f <= iv.y_end + EPS_) {
              skipL = true;
              break;
            }
          }
          if (skipL)
            continue;
          // 重投影
          cv::Point3f ray((x_f - cx_l) / fx_l, (y_f - cy_l) / fy_l, 1.0f);
          ray *= 1.0f / cv::norm(ray);
          auto ips = findIntersection({0, 0, 0}, ray, coef);
          if (ips.empty())
            continue;
          cv::Point3f pt3;
          bool ok = false;
          for (auto &q : ips)
            if (q.z > 100 && q.z < 1500) {
              pt3 = q;
              ok = true;
              break;
            }
          if (!ok)
            continue;
          cv::Point3f pr(pt3.x - baseline, pt3.y, pt3.z);
          float xr = fx_r * pr.x / pr.z + cx_r;
          float yr = alignToPrecision(fy_r * pr.y / pr.z + cy_r);
          if (xr < 0 || xr >= rectify_r.cols || yr < 0 || yr >= rectify_r.rows)
            continue;
          repro_cnt++;
          repro_cnt_total++;
          // 遍历右线
          for (int r = 0; r < R; ++r) {
            // if (l == 5 && p ==1 && r == 5)
            //     puts("");
            auto it = laser_r[r].points.lower_bound(yr);
            if (it != laser_r[r].points.begin()) {
              auto prev = std::prev(it);
              if (it == laser_r[r].points.end() ||
                  fabs(prev->first - yr) < fabs(it->first - yr))
                it = prev;
            }
            if (fabs(it->first - yr) > EPS_)
              continue;
            float d = hypot(it->second.x - xr, it->second.y - yr);
            // if (r == 5 && p == 1 && l == 5)
            //     puts("");
            if (d > D_thresh_)
              continue;
            bool skipR = false;
            for (auto &iv : locked_r[r]) {
              if (yr >= iv.y_start - EPS_ && yr <= iv.y_end + EPS_) {
                skipR = true;
                break;
              }
            }
            if (!skipR)
              support[r].emplace_back(yr, d);
          }
        }
        if (support.empty())
          continue;
        // 对每个右线构建子段集
        for (auto &ent : support) {
          auto &vec = ent.second;
          std::sort(vec.begin(), vec.end(),
                    [](auto &a, auto &b) { return a.first < b.first; });
          // 拆分子段
          std::vector<Interval> segs;
          int start = 0;
          for (int i = 1; i < (int)vec.size(); ++i) {
            float gap = vec[i].first - vec[i - 1].first;
            if (gap > 2 * precision_ + EPS_) {
              Interval iv{alignToPrecision(vec[start].first),
                          alignToPrecision(vec[i - 1].first), i - start};
              segs.push_back(iv);
              start = i;
            }
          }
          Interval ivlast{alignToPrecision(vec[start].first),
                          alignToPrecision(vec.back().first),
                          (int)vec.size() - start};
          segs.push_back(ivlast);
          if (segs.empty())
            continue;
          // 汇总此 (l,p,r)所有子段
          int total_count = 0;
          for (auto &iv : segs)
            total_count += iv.count;
          if (total_count < MIN_LEN_)
            continue;
          // 计算距离集合
          std::vector<std::pair<float, float>> allpd;
          for (auto &pd : vec) {
            for (auto &iv : segs) {
              if (pd.first >= iv.y_start - EPS_ &&
                  pd.first <= iv.y_end + EPS_) {
                allpd.push_back(pd);
                break;
              }
            }
          }
          // 计算标准差
          float std_dev = 0.0f, coverage = 0.0f;
          // 统一打分
          float score =
              computeEnhancedScore(allpd, repro_cnt, coverage, std_dev);
          if (score <= S_thresh_) {
            cands.push_back({l, p, ent.first, segs, score, coverage, std_dev});
          }
        }
      }
      if (cands.empty())
        continue;

      // 如果唯一候选
      bool lock = false;
      auto &m = cands[0];
      if (cands.size() == 1 && m.score <= 7.0f && m.coverage >= 0.65)
        lock = true;
      else if (cands.size() == 1 && m.score <= 11.0f && m.coverage >= 0.70)
        lock = true;
      else if (cands.size() == 1 && m.score <= 14.5f && m.coverage >= 0.85)
        lock = true;
      else if (cands.size() == 1 && m.score <= 17.0f && m.coverage >= 0.91 &&
               m.std_dev < 5.0f)
        lock = true;
      else if (cands.size() > 1) {
        std::sort(cands.begin(), cands.end(),
                  [](auto &a, auto &b) { return a.score < b.score; });
        auto &mm = cands[1];
        if (m.p_idx == mm.p_idx && fabs(mm.score - m.score) <= 1.5 &&
            fabs(m.std_dev - mm.std_dev) < 1.0f)
          lock = true;
        else if (fabs(mm.score - m.score) >= 10.0f && m.score <= 13.0f &&
                 m.coverage >= 0.65)
          lock = true;
        else if (fabs(mm.score - m.score) <= 3.0f &&
                 fabs(m.std_dev - mm.std_dev) < 1.0f) {
          if (mm.coverage > m.coverage)
            std::swap(m, mm);
          if (m.coverage - mm.coverage > 0.6)
            lock = true;
        }
      }
      if (lock) {
        final_matches.push_back(m);
        locked_l[m.l_idx].insert(locked_l[m.l_idx].end(), m.intervals.begin(),
                                 m.intervals.end());
        merge_intervals(locked_l[m.l_idx]);
        locked_r[m.r_idx].insert(locked_r[m.r_idx].end(), m.intervals.begin(),
                                 m.intervals.end());
        merge_intervals(locked_r[m.r_idx]);
        progress = true;
      }

      // 过程可视化
#ifdef DEBUG_PLANE_MATCH
      cv::Mat vis = vis_global.clone();
      // 左图激光点
      for (auto [y, x] : pts)
        cv::circle(vis, cv::Point2f(x, y), 1.5, {0, 255, 0}, -1);
      // 左线ID
      auto l_mid = pts.begin();
      std::advance(l_mid, pts.size() / 2);
      cv::putText(vis, "L" + std::to_string(l),
                  cv::Point2f(l_mid->second, l_mid->first),
                  cv::FONT_HERSHEY_SIMPLEX, 1.3, {0, 255, 0}, 2);
      // 重投影点（红）及候选区间（黄）
      cv::Point2i test_point(10, 40);
      for (auto &c : cands) {
        // 蓝色整条右线
        for (auto &p_r : laser_r[c.r_idx].points)
          cv::circle(vis, cv::Point2f(p_r.second.x + off, p_r.second.y), 1.5,
                     r_laser_col, -1);
        // 红色投影
        const auto &coef_c = planes[c.p_idx].coefficients;
        for (auto [y_f, x_f] : pts) {
          cv::Point3f ray((x_f - cx_l) / fx_l, (y_f - cy_l) / fy_l, 1.0f);
          ray *= 1.0f / cv::norm(ray);
          auto ips = findIntersection({0, 0, 0}, ray, coef_c);
          if (ips.empty())
            continue;
          cv::Point3f pt3;
          bool ok = false;
          for (auto &q : ips)
            if (q.z > 100 && q.z < 1200) {
              pt3 = q;
              ok = true;
              break;
            }
          if (!ok)
            continue;
          cv::Point3f pr(pt3.x - baseline, pt3.y, pt3.z);
          float xr = fx_r * pr.x / pr.z + cx_r;
          float yr = alignToPrecision(fy_r * pr.y / pr.z + cy_r);
          if (xr >= 0 && xr < rectify_r.cols && yr >= 0 && yr < rectify_r.rows)
            cv::circle(vis, cv::Point2f(xr + off, yr), 1.5, {0, 0, 255}, -1);
        }
        // 黄色区间
        for (auto &iv : c.intervals)
          for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
            auto it_r = laser_r[c.r_idx].points.lower_bound(y0);
            if (it_r != laser_r[c.r_idx].points.begin()) {
              auto prev = std::prev(it_r);
              if (it_r == laser_r[c.r_idx].points.end() ||
                  fabs(prev->first - y0) < fabs(it_r->first - y0)) {
                it_r = prev;
              }
            }
            if (fabs(it_r->first - y0) > EPS_)
              continue;
            cv::circle(vis, cv::Point2f(it_r->second.x + off, y0), 1.5,
                       proc_interval_col, -1);
          }
        // 右线ID
        auto r_mid = laser_r[c.r_idx].points.begin();
        std::advance(r_mid, laser_r[c.r_idx].points.size() / 2);
        cv::putText(vis, "R" + std::to_string(c.r_idx),
                    cv::Point2f(r_mid->second.x + off, r_mid->second.y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.3, {0, 255, 0}, 2);
        // 文本：候选数
        char text_buf[128];
        snprintf(text_buf, sizeof(text_buf),
                 "L%d->P%d->R%d (S %.2f / C %.2f%%)", c.l_idx, c.p_idx, c.r_idx,
                 c.score, c.coverage * 100);
        cv::putText(vis, text_buf, test_point, cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    {0, 255, 255}, 2);
        test_point.y += 45;
      }
      cv::imshow("vis_img", vis);
      cv::waitKey(0);
#endif
    }
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  cv::Mat vis_all;
  cv::hconcat(rectify_l, rectify_r, vis_all);
  if (vis_all.channels() == 1)
    cv::cvtColor(vis_all, vis_all, cv::COLOR_GRAY2BGR);
  int off_all = rectify_l.cols;
  static const std::vector<cv::Scalar> palette30 = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},  {255, 255, 0}, {255, 0, 255},
      {0, 255, 255}, {128, 0, 0},   {0, 128, 0},  {0, 0, 128},   {128, 128, 0},
      {128, 0, 128}, {0, 128, 128}, {64, 0, 128}, {128, 64, 0},  {0, 128, 64},
      {64, 128, 0},  {0, 64, 128},  {128, 0, 64}, {192, 192, 0}, {192, 0, 192},
      {64, 255, 0},  {255, 64, 0},  {0, 64, 255}, {0, 255, 64},  {255, 0, 64},
      {64, 0, 255},  {192, 0, 64},  {64, 192, 0}, {0, 192, 64},  {64, 0, 192}};
  for (int idx = 0; idx < (int)final_matches.size(); ++idx) {
    auto &m = final_matches[idx];
    cv::Scalar col = palette30[idx % palette30.size()];
    // 左图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itL = sample_points[m.l_idx].lower_bound(y0);
        if (itL != sample_points[m.l_idx].begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itL);
          if (itL == sample_points[m.l_idx].end() ||
              fabs(prev->first - y0) < fabs(itL->first - y0)) {
            itL = prev;
          }
        }
        if (fabs(itL->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itL->second, y0), 2, col, -1);
      }
    // 右图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itR = laser_r[m.r_idx].points.lower_bound(y0);
        if (itR != laser_r[m.r_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itR);
          if (itR == laser_r[m.r_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itR->first - y0)) {
            itR = prev;
          }
        }
        if (fabs(itR->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itR->second.x + off_all, y0), 2, col,
                   -1);
      }
  }
  cv::imshow("vis_img", vis_all);
  cv::waitKey(0);
  cv::destroyWindow("vis_img");
#endif

  return final_matches;
}

std::vector<IntervalMatch>
LaserProcessor::match6(const std::vector<std::map<float, float>>
                           &sample_points,                   // 左激光线: y -> x
                       const std::vector<LaserLine> &laser_r // 右激光线
) {
  // ---------- 配置与常量 ----------
  // 从配置获取相机内参与平面集合（保持与你原始实现一致）
  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();

  // 相机内参（左/右）
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  // 阈值（你可以按需调整）
  constexpr float D_thresh_ = 20.0f;
  constexpr float S_thresh_ = 20.0f;
  constexpr float Delta_thresh = 3.0f;
  constexpr int MIN_LEN_ = 80;

  // 容差与插值精度（可按工程设定修改）
  const float EPS_ = 1e-4f;
  const float precision_ = 1.0f;

  // 结构尺寸
  const int L = static_cast<int>(sample_points.size());
  const int R = static_cast<int>(laser_r.size());
  const int P = static_cast<int>(surfaces.size());

  // 结果容器与锁区（最终会返回 final_matches）
  std::vector<IntervalMatch> final_matches;
  std::vector<std::vector<Interval>> locked_l(L);
  std::vector<std::vector<Interval>> locked_r(R);

  // ---------- 预处理（串行） ----------
  // 1) 将右激光线的 map => vector 以便使用二分查找（提高缓存局部性）
  std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
  for (int r = 0; r < R; ++r) {
    const auto &mp = laser_r[r].points;
    right_vec[r].reserve(mp.size());
    for (const auto &kv : mp)
      right_vec[r].emplace_back(kv.first, kv.second);
    // map 本身就是按 y 有序，故无需额外排序
  }

  // 2) 为每条左线按 y 顺序构建采样点数组和预计算射线（归一化）
  //    这样不同平面间可以复用 ray，避免重复归一化计算
  struct LeftSample {
    std::vector<float> ys;         // y 值（与下面 xs/x 对应）
    std::vector<float> xs;         // x 值
    std::vector<cv::Point3f> rays; // 归一化射线方向
  };
  std::vector<LeftSample> left_samples(L);
  for (int l = 0; l < L; ++l) {
    const auto &mp = sample_points[l]; // map<float,float>
    auto &ls = left_samples[l];
    ls.ys.reserve(mp.size());
    ls.xs.reserve(mp.size());
    ls.rays.reserve(mp.size());
    for (const auto &kv : mp) {
      float y_f = kv.first;
      float x_f = kv.second;
      ls.ys.push_back(y_f);
      ls.xs.push_back(x_f);
      cv::Point3f ray(
          (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
          (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
      float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      if (rn > 0.0f)
        ray *= (1.0f / rn);
      ls.rays.push_back(ray);
    }
  }

  // ---------- 多轮并行候选生成 + 串行贪心上锁 ----------
  bool progress = true;
  while (progress) {
    progress = false;

    // 记录快照供并行阶段只读使用，避免并发冲突
    auto snap_locked_l = locked_l;
    auto snap_locked_r = locked_r;

    // 并行候选容器（线程安全）
    tbb::concurrent_vector<IntervalMatch> all_cands;

    // 扁平化 (l,p) 范围并行化：索引 [0, L*P)
    const int total_tasks = L * P;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, total_tasks),
        [&](const tbb::blocked_range<int> &range) {
          for (int idx = range.begin(); idx != range.end(); ++idx) {
            int l = idx / P;
            int p = idx % P;

            // 若该左线没有采样点则跳过
            if (left_samples[l].ys.empty())
              continue;

            // 本任务的局部支持集合和候选（局部缓冲，减少并发 push）
            std::unordered_map<int, std::vector<std::pair<float, float>>>
                support; // r -> vector<(yr, d)>
            std::vector<IntervalMatch> local_cands;
            local_cands.reserve(4);

            // 当前平面系数
            const cv::Mat &coef = surfaces[p].coefficients;

            // reprojection 计数（按 (l,p) 统计）
            int repro_cnt = 0;

            // 遍历左线每个采样点（使用预计算 ray）
            const LeftSample &ls = left_samples[l];
            for (size_t i = 0; i < ls.ys.size(); ++i) {
              float y_f = ls.ys[i];
              float x_f = ls.xs[i];

              // 如果 y 在左侧已锁区间内（快照），跳过
              bool skipL = false;
              for (const auto &iv : snap_locked_l[l]) {
                if (y_f >= iv.y_start - EPS_ && y_f <= iv.y_end + EPS_) {
                  skipL = true;
                  break;
                }
              }
              if (skipL)
                continue;

              // 使用预计算射线与当前平面求交
              const cv::Point3f &ray = ls.rays[i];
              auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
              if (ips.empty())
                continue;

              // 取第一个满足深度范围的点（与原实现一致）
              cv::Point3f pt3;
              bool ok = false;
              for (const auto &q : ips) {
                if (q.z > 100.0f && q.z < 1500.0f) {
                  pt3 = q;
                  ok = true;
                  break;
                }
              }
              if (!ok)
                continue;

              // 投影到右图像
              cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y,
                             pt3.z);
              float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
              float yr = alignToPrecision(
                  static_cast<float>(fy_r * pr.y / pr.z + cy_r));
              if (!(xr >= 0 && xr < 2048 && yr >= 0 && yr < 1200))
                continue;

              ++repro_cnt;

              // 对每条右线查找最接近的 y（使用二分法在 right_vec）
              for (int r = 0; r < R; ++r) {
                const auto &rv = right_vec[r];
                if (rv.empty())
                  continue;
                // 快速范围剪枝：若 yr
                // 小于第一或大于最后，则可能仍可匹配（但需检查 nearest） 使用
                // lower_bound 在 rv 上二分（按 y 排序）
                auto it = std::lower_bound(
                    rv.begin(), rv.end(), yr,
                    [](const std::pair<float, LaserPoint> &a, float value) {
                      return a.first < value;
                    });
                // 选择更接近的元素
                if (it != rv.begin()) {
                  auto prev = it - 1;
                  if (it == rv.end() ||
                      std::fabs(prev->first - yr) < std::fabs(it->first - yr))
                    it = prev;
                }
                if (it == rv.end())
                  continue;
                if (std::fabs(it->first - yr) > EPS_)
                  continue;
                const LaserPoint &lp = it->second;
                float d = std::hypot(lp.x - xr, lp.y - yr);
                if (d > D_thresh_)
                  continue;

                // 如果 yr 落在右侧已锁区间（快照）则跳过
                bool skipR = false;
                for (const auto &iv : snap_locked_r[r]) {
                  if (yr >= iv.y_start - EPS_ && yr <= iv.y_end + EPS_) {
                    skipR = true;
                    break;
                  }
                }
                if (skipR)
                  continue;

                // 符合条件则记录 (yr, d) 到 support[r]
                support[r].emplace_back(yr, d);
              } // end for r
            } // end for left points

            // 若 support 为空，直接跳过
            if (support.empty())
              continue;

            // 对每个右线的支持点集合进行排序并拆分为子段，计算 score 并产生候选
            for (auto &ent : support) {
              int r_idx = ent.first;
              auto vec = std::move(ent.second);
              if (vec.empty())
                continue;
              std::sort(vec.begin(), vec.end(),
                        [](const auto &a, const auto &b) {
                          return a.first < b.first;
                        });

              // 拆分为连续子段，间隔 > 2*precision_ 则分割
              std::vector<Interval> segs;
              segs.reserve(4);
              int start = 0;
              for (int i = 1; i < static_cast<int>(vec.size()); ++i) {
                float gap = vec[i].first - vec[i - 1].first;
                if (gap > 2.0f * precision_ + EPS_) {
                  Interval iv{alignToPrecision(vec[start].first),
                              alignToPrecision(vec[i - 1].first), i - start};
                  segs.push_back(iv);
                  start = i;
                }
              }
              Interval ivlast{alignToPrecision(vec[start].first),
                              alignToPrecision(vec.back().first),
                              static_cast<int>(vec.size()) - start};
              segs.push_back(ivlast);

              // 过滤掉长度不足的子段
              int total_count = 0;
              for (const auto &iv : segs)
                total_count += iv.count;
              if (total_count < MIN_LEN_)
                continue;

              // 收集子段内的 (y, d) 对用于打分
              std::vector<std::pair<float, float>> allpd;
              allpd.reserve(vec.size());
              for (const auto &pd : vec) {
                for (const auto &iv : segs) {
                  if (pd.first >= iv.y_start - EPS_ &&
                      pd.first <= iv.y_end + EPS_) {
                    allpd.push_back(pd);
                    break;
                  }
                }
              }

              // 计算 score, coverage, std_dev（外部函数）
              float std_dev = 0.0f, coverage = 0.0f;
              float score =
                  computeEnhancedScore(allpd, repro_cnt, coverage, std_dev);
              if (score <= S_thresh_) {
                IntervalMatch im;
                im.l_idx = l;
                im.p_idx = p;
                im.r_idx = r_idx;
                im.intervals = std::move(segs);
                im.score = score;
                im.coverage = coverage;
                im.std_dev = std_dev;
                local_cands.push_back(std::move(im));
              }
            } // end for each r in support

            // 将本任务产生的 local_cands 写入全局并发向量
            for (auto &c : local_cands)
              all_cands.push_back(std::move(c));
          } // end for idx in range
        }); // end parallel_for

    // ---------- 串行归约阶段（决定性排序 + 贪心上锁） ----------
    // 将并发容器转为 std::vector
    std::vector<IntervalMatch> cands;
    cands.reserve(all_cands.size());
    for (const auto &c : all_cands)
      cands.push_back(c);

    // 确定性排序：score小优先，coverage高优先，std_dev小优先，最后按
    // l_idx,r_idx,p_idx 保证稳定性
    std::sort(cands.begin(), cands.end(),
              [](const IntervalMatch &a, const IntervalMatch &b) {
                if (a.score != b.score)
                  return a.score < b.score;
                if (a.coverage != b.coverage)
                  return a.coverage > b.coverage;
                if (a.std_dev != b.std_dev)
                  return a.std_dev < b.std_dev;
                if (a.l_idx != b.l_idx)
                  return a.l_idx < b.l_idx;
                if (a.r_idx != b.r_idx)
                  return a.r_idx < b.r_idx;
                return a.p_idx < b.p_idx;
              });

    // 再逐个候选尝试贪心上锁（串行执行以保证确定性）
    for (auto &m : cands) {
      // 如果与已锁定的该左线或右线区间冲突则跳过
      auto conflict_with_locked =
          [&](const std::vector<Interval> &add,
              const std::vector<Interval> &locked) -> bool {
        if (add.empty() || locked.empty())
          return false;
        // 两集合排序后两指针检测重叠（实现上简单化使用 O(n log n) 排序+双指针）
        auto a = add;
        auto b = locked;
        std::sort(a.begin(), a.end(), [](const Interval &x, const Interval &y) {
          return x.y_start < y.y_start;
        });
        std::sort(b.begin(), b.end(), [](const Interval &x, const Interval &y) {
          return x.y_start < y.y_start;
        });
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
          const Interval &A = a[i];
          const Interval &B = b[j];
          if (A.y_end + EPS_ < B.y_start) {
            ++i;
            continue;
          }
          if (B.y_end + EPS_ < A.y_start) {
            ++j;
            continue;
          }
          return true; // 有交叠
        }
        return false;
      };

      if (conflict_with_locked(m.intervals, locked_l[m.l_idx]))
        continue;
      if (conflict_with_locked(m.intervals, locked_r[m.r_idx]))
        continue;

      // 不冲突则写入锁区并合并
      auto &Ll = locked_l[m.l_idx];
      Ll.insert(Ll.end(), m.intervals.begin(), m.intervals.end());
      // 合并区间函数（按 EPS）
      std::sort(Ll.begin(), Ll.end(), [](const Interval &a, const Interval &b) {
        if (a.y_start != b.y_start)
          return a.y_start < b.y_start;
        return a.y_end < b.y_end;
      });
      std::vector<Interval> mergedL;
      mergedL.reserve(Ll.size());
      if (!Ll.empty()) {
        Interval cur = Ll[0];
        for (size_t k = 1; k < Ll.size(); ++k) {
          const auto &n = Ll[k];
          if (n.y_start <= cur.y_end + EPS_) {
            cur.y_end = std::max(cur.y_end, n.y_end);
            cur.count += n.count;
          } else {
            mergedL.push_back(cur);
            cur = n;
          }
        }
        mergedL.push_back(cur);
      }
      Ll.swap(mergedL);

      auto &Rr = locked_r[m.r_idx];
      Rr.insert(Rr.end(), m.intervals.begin(), m.intervals.end());
      std::sort(Rr.begin(), Rr.end(), [](const Interval &a, const Interval &b) {
        if (a.y_start != b.y_start)
          return a.y_start < b.y_start;
        return a.y_end < b.y_end;
      });
      std::vector<Interval> mergedR;
      mergedR.reserve(Rr.size());
      if (!Rr.empty()) {
        Interval cur = Rr[0];
        for (size_t k = 1; k < Rr.size(); ++k) {
          const auto &n = Rr[k];
          if (n.y_start <= cur.y_end + EPS_) {
            cur.y_end = std::max(cur.y_end, n.y_end);
            cur.count += n.count;
          } else {
            mergedR.push_back(cur);
            cur = n;
          }
        }
        mergedR.push_back(cur);
      }
      Rr.swap(mergedR);

      // 记录最终匹配（这里只记录匹配项，l->r 的一对一映射由上游决定是否需要）
      final_matches.push_back(m);
      progress = true;
    } // end for cands
  } // end while(progress)

  // 返回最终匹配集合
  return final_matches;
}

std::vector<IntervalMatch>
LaserProcessor::match7(const std::vector<LaserLine> &laser_l,
                       const std::vector<LaserLine> &laser_r,
                       const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
  const int img_rows = rectify_l.rows;
  const int img_cols = rectify_l.cols;

  // 结构尺寸
  const int L = static_cast<int>(laser_l.size());
  const int R = static_cast<int>(laser_r.size());

  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
  const int P = static_cast<int>(surfaces.size());

  // 相机内参（左/右）
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  // --------------- 预处理 ---------------
  std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
  for (int r = 0; r < R; ++r) {
    right_vec[r].reserve(laser_r[r].points.size());
    for (const auto &kv : laser_r[r].points)
      right_vec[r].emplace_back(kv.first, kv.second);
  }

  // 预计算每条左线的采样点数组与归一化射线
  struct LeftSample {
    std::vector<float> ys, xs;
    std::vector<cv::Point3f> rays;
  };
  std::vector<LeftSample> left_samples(L);
  for (int l = 0; l < L; ++l) {
    const auto &mp = laser_l[l].points;
    auto &ls = left_samples[l];
    ls.ys.reserve(mp.size());
    ls.xs.reserve(mp.size());
    ls.rays.reserve(mp.size());
    for (const auto &kv : mp) {
      float y_f = kv.first;
      float x_f = kv.second.x;
      ls.ys.push_back(y_f);
      ls.xs.push_back(x_f);
      cv::Point3f ray(
          (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
          (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
      float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      if (rn > 0.0f)
        ray *= (1.0f / rn);
      ls.rays.push_back(ray);
    }
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  // 全局可视化底图
  cv::Mat vis_global;
  cv::hconcat(rectify_l, rectify_r, vis_global);
  if (vis_global.channels() == 1)
    cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
  int off = rectify_l.cols;
  static const cv::Scalar proc_interval_col(0, 255, 255); // 候选区间颜色
  static const cv::Scalar r_laser_col(255, 0, 0);         // 整条右线颜色
  cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
  cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

  // --------------- 并行候选生成 ---------------
  tbb::concurrent_vector<IntervalMatch> all_cands;
  const int total_tasks = L * P; // 扁平化 (l,p) 任务，保证 P>=1

  tbb::parallel_for(
      tbb::blocked_range<int>(0, total_tasks),
      [&](const tbb::blocked_range<int> &range) {
        for (int idx = range.begin(); idx != range.end(); ++idx) {
          int l = idx / std::max(1, P);
          int p = idx % std::max(1, P);
          if (l < 0 || l >= L || p < 0 || p >= P)
            continue;
          const auto &ls = left_samples[l];
          if (ls.ys.empty())
            continue;

          // 支持集合：r -> vector<(yr,d)>
          std::unordered_map<int, std::vector<std::pair<float, float>>> support;
          int repro_cnt = 0;

          const cv::Mat &coef = surfaces[p].coefficients;
          for (size_t i = 0; i < ls.ys.size(); ++i) {
            float y_f = ls.ys[i];
            float x_f = ls.xs[i];

            // 跳过左侧已锁区（候选生成时我们没有全局锁的参考，此处保持与重构版等价：不跳）
            // （注意：匈牙利版本将基于全对候选全局分配，而不是逐轮锁定）

            // 与平面求交
            const cv::Point3f &ray = ls.rays[i];
            auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
            if (ips.empty())
              continue;
            cv::Point3f pt3;
            bool ok = false;
            for (auto &q : ips)
              if (q.z > 100 && q.z < 1500) {
                pt3 = q;
                ok = true;
                break;
              }
            if (!ok)
              continue;

            cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
            float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
            float yr =
                alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
            if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows)
              continue;

            ++repro_cnt;

            // 遍历右线
            for (int r = 0; r < R; ++r) {
              const auto &rv = right_vec[r];
              if (rv.empty())
                continue;
              auto it =
                  std::lower_bound(rv.begin(), rv.end(), yr,
                                   [](const std::pair<float, LaserPoint> &a,
                                      float v) { return a.first < v; });
              if (it != rv.begin()) {
                auto prev = it - 1;
                if (it == rv.end() ||
                    std::fabs(prev->first - yr) < std::fabs(it->first - yr))
                  it = prev;
              }
              if (it == rv.end())
                continue;
              if (std::fabs(it->first - yr) > EPS_)
                continue;
              float d = std::hypot(it->second.x - xr, it->second.y - yr);
              if (d > D_thresh_)
                continue;

              // if (l == 4 && r == 4)
              //     puts("");

              support[r].emplace_back(yr, d);
            }
          } // end for left points

          // 支持集合转子段并打分
          for (auto &ent : support) {
            int r_idx = ent.first;
            auto vec = std::move(ent.second);
            if (vec.empty())
              continue;
            std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
              return a.first < b.first;
            });

            // 拆分子段
            std::vector<Interval> segs;
            int start = 0;
            for (int i = 1; i < (int)vec.size(); ++i) {
              float gap = vec[i].first - vec[i - 1].first;
              if (gap > 2.0f * precision_ + EPS_) {
                Interval iv{alignToPrecision(vec[start].first),
                            alignToPrecision(vec[i - 1].first), i - start};
                segs.push_back(iv);
                start = i;
              }
            }
            Interval ivlast{alignToPrecision(vec[start].first),
                            alignToPrecision(vec.back().first),
                            (int)vec.size() - start};
            segs.push_back(ivlast);

            int total_count = 0;
            for (const auto &iv : segs)
              total_count += iv.count;
            if (total_count < MIN_LEN_)
              continue;

            // 收集该子段内的 (y,d) 用于评分
            std::vector<std::pair<float, float>> allpd;
            allpd.reserve(vec.size());
            for (auto &pd : vec) {
              for (auto &iv : segs) {
                if (pd.first >= iv.y_start - EPS_ &&
                    pd.first <= iv.y_end + EPS_) {
                  allpd.push_back(pd);
                  break;
                }
              }
            }

            float std_dev = 0.0f, coverage = 0.0f;
            float score =
                computeEnhancedScore(allpd, repro_cnt, coverage, std_dev);
            if (score <= S_thresh_) {
              IntervalMatch c;
              c.l_idx = l;
              c.p_idx = p;
              c.r_idx = r_idx;
              c.intervals = std::move(segs);
              c.score = score;
              c.coverage = coverage;
              c.std_dev = std_dev;
              all_cands.push_back(std::move(c));
            }
            if (l == 8)
              printf("l%d-r%d, score: %.3f\n", l, r_idx, score);
          } // end support->segs
        } // end idx loop
      }); // end parallel_for

  // --------------- 对 (l,r) 做归约：保留每对最优候选 ---------------
  const double INF_COST = 1e9;
  int n = std::max(L, R); // cost_matrix dims n x n & n = max(L,R)
  std::vector<std::vector<double>> cost(n, std::vector<double>(n, INF_COST));
  std::vector<std::vector<IntervalMatch>> best_pair(
      L, std::vector<IntervalMatch>(R));
  std::vector<std::vector<bool>> has_best(L, std::vector<bool>(R, false));

  for (const auto &c : all_cands) {
    if (c.l_idx < 0 || c.l_idx >= L || c.r_idx < 0 || c.r_idx >= R)
      continue;
    if (!has_best[c.l_idx][c.r_idx] ||
        c.score < best_pair[c.l_idx][c.r_idx].score) {
      best_pair[c.l_idx][c.r_idx] = c;
      has_best[c.l_idx][c.r_idx] = true;
    }
  }

  // 填充 cost 矩阵（i->l 行, j->r 列）
  for (int l = 0; l < L; ++l) {
    for (int r = 0; r < R; ++r) {
      if (has_best[l][r]) {
        cost[l][r] = static_cast<double>(best_pair[l][r].score);
      } else {
        cost[l][r] = INF_COST; // 不可分配
      }
    }
  }

  // --------------- 匈牙利求解（最小成本） ---------------
  std::vector<int> matchL;
  hungarian1(cost, matchL);

  // --------------- 根据匹配结果构造最终 IntervalMatch 输出 ---------------
  std::vector<IntervalMatch> final_matches;
  for (int i = 0; i < n; ++i) {
    int j = matchL[i];
    if (i < L && j >= 0 && j < R) {
      if (!has_best[i][j])
        continue; // 可能为 INF 分配（但是匈牙利应避免）
      const IntervalMatch &c = best_pair[i][j];
      // 额外阈值检查：coverage 以及 score 上限（可按需调整）
      if (c.coverage < 0.05f)
        continue; // 极低覆盖忽略（经验阈值）
      IntervalMatch im;
      im.l_idx = c.l_idx;
      im.p_idx = c.p_idx;
      im.r_idx = c.r_idx;
      im.intervals = c.intervals;
      im.score = c.score;
      im.coverage = c.coverage;
      im.std_dev = c.std_dev;
      // 合并区间以保持输出整洁
      merge_intervals_vec(im.intervals);
      final_matches.push_back(std::move(im));
    }
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  cv::Mat vis_all;
  cv::hconcat(rectify_l, rectify_r, vis_all);
  if (vis_all.channels() == 1)
    cv::cvtColor(vis_all, vis_all, cv::COLOR_GRAY2BGR);
  int off_all = rectify_l.cols;
  static const std::vector<cv::Scalar> palette30 = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},  {255, 255, 0}, {255, 0, 255},
      {0, 255, 255}, {128, 0, 0},   {0, 128, 0},  {0, 0, 128},   {128, 128, 0},
      {128, 0, 128}, {0, 128, 128}, {64, 0, 128}, {128, 64, 0},  {0, 128, 64},
      {64, 128, 0},  {0, 64, 128},  {128, 0, 64}, {192, 192, 0}, {192, 0, 192},
      {64, 255, 0},  {255, 64, 0},  {0, 64, 255}, {0, 255, 64},  {255, 0, 64},
      {64, 0, 255},  {192, 0, 64},  {64, 192, 0}, {0, 192, 64},  {64, 0, 192}};
  for (int idx = 0; idx < (int)final_matches.size(); ++idx) {
    auto &m = final_matches[idx];
    cv::Scalar col = palette30[idx % palette30.size()];
    // 左图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itL = laser_l[m.l_idx].points.lower_bound(y0);
        if (itL != laser_l[m.l_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itL);
          if (itL == laser_l[m.l_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itL->first - y0)) {
            itL = prev;
          }
        }
        if (fabs(itL->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itL->second.x, y0), 2, col, -1);
      }
    // 右图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itR = laser_r[m.r_idx].points.lower_bound(y0);
        if (itR != laser_r[m.r_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itR);
          if (itR == laser_r[m.r_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itR->first - y0)) {
            itR = prev;
          }
        }
        if (fabs(itR->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itR->second.x + off_all, y0), 2, col,
                   -1);
      }

    // 计算整条线的中点位置（而不是每个区间的中点）
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // 找到所有区间的最小和最大y值
    for (auto &iv : m.intervals) {
      if (iv.y_start < min_y)
        min_y = iv.y_start;
      if (iv.y_end > max_y)
        max_y = iv.y_end;
    }

    // 计算整条线的中点
    float mid_y = (min_y + max_y) / 2.0f;

    // 左图标签
    auto itL = laser_l[m.l_idx].points.lower_bound(mid_y);
    if (itL != laser_l[m.l_idx].points.begin()) {
      auto prev = std::prev(itL);
      if (itL == laser_l[m.l_idx].points.end() ||
          fabs(prev->first - mid_y) < fabs(itL->first - mid_y)) {
        itL = prev;
      }
    }
    if (fabs(itL->first - mid_y) <= EPS_) {
      std::string labelL = "L" + std::to_string(m.l_idx);
      cv::putText(vis_all, labelL, cv::Point(itL->second.x, mid_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2.5); // 绿色粗体
    }

    // 右图标签
    auto itR = laser_r[m.r_idx].points.lower_bound(mid_y);
    if (itR != laser_r[m.r_idx].points.begin()) {
      auto prev = std::prev(itR);
      if (itR == laser_r[m.r_idx].points.end() ||
          fabs(prev->first - mid_y) < fabs(itR->first - mid_y)) {
        itR = prev;
      }
    }
    if (fabs(itR->first - mid_y) <= EPS_) {
      std::string labelR = "R" + std::to_string(m.r_idx);
      cv::putText(vis_all, labelR,
                  cv::Point(itR->second.x + off_all, mid_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2.5); // 绿色粗体
    }
  }
  cv::imshow("vis_img", vis_all);
  cv::waitKey(0);
  cv::destroyWindow("vis_img");

  static int dbg_idx = 0;
  cv::imwrite("debug_img/match_res_" + std::to_string(dbg_idx) + ".png",
              vis_all);
  dbg_idx++;
#endif

  // ---------- 返回最终匹配集合 ----------
  return final_matches;
}

/***************************************************************************************
 */

/************************************* MatchV8
 * *************************************** */

std::vector<IntervalMatch>
LaserProcessor::match8(const std::vector<LaserLine> &laser_l,
                       const std::vector<LaserLine> &laser_r,
                       const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
  const int img_rows = rectify_l.rows;
  const int img_cols = rectify_l.cols;
  const int L = static_cast<int>(laser_l.size());
  const int R = static_cast<int>(laser_r.size());

  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();
  const int P = static_cast<int>(surfaces.size());

  // 相机内参
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  // =============== 阶段1：自适应预处理 ===============

  // 1.1 高质量匹配阈值设定
  const int HIGH_QUALITY_POINT_THRESH = 1000;   // 高质量线的点数阈值
  const float HIGH_QUALITY_SCORE_THRESH = 1.5f; // 高质量线的得分阈值

  // 1.2 预处理右线数据
  std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
  for (int r = 0; r < R; ++r) {
    right_vec[r].reserve(laser_r[r].points.size());
    for (const auto &kv : laser_r[r].points) {
      right_vec[r].emplace_back(kv.first, kv.second);
    }
  }

  // 1.3 预计算左线采样数据
  struct LeftSample {
    std::vector<float> ys, xs;
    std::vector<cv::Point3f> rays;
    int original_point_count;
  };
  std::vector<LeftSample> left_samples(L);
  for (int l = 0; l < L; ++l) {
    const auto &mp = laser_l[l].points;
    auto &ls = left_samples[l];
    ls.original_point_count = static_cast<int>(mp.size());
    ls.ys.reserve(mp.size());
    ls.xs.reserve(mp.size());
    ls.rays.reserve(mp.size());

    for (const auto &kv : mp) {
      float y_f = kv.first;
      float x_f = kv.second.x;
      ls.ys.push_back(y_f);
      ls.xs.push_back(x_f);
      cv::Point3f ray(
          (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
          (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
      float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      if (rn > 0.0f)
        ray *= (1.0f / rn);
      ls.rays.push_back(ray);
    }
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  // 全局可视化底图
  cv::Mat vis_global;
  cv::hconcat(rectify_l, rectify_r, vis_global);
  if (vis_global.channels() == 1)
    cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
  int off = rectify_l.cols;
  static const cv::Scalar proc_interval_col(0, 255, 255); // 候选区间颜色
  static const cv::Scalar r_laser_col(255, 0, 0);         // 整条右线颜色
  cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
  cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

  // =============== 阶段1：增强候选生成 ===============

  struct EnhancedCandidate {
    IntervalMatch match;
    bool is_high_quality;
    float length_ratio;
  };

  tbb::concurrent_vector<EnhancedCandidate> all_enhanced_cands;
  const int total_tasks = L * P;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, total_tasks),
      [&](const tbb::blocked_range<int> &range) {
        for (int idx = range.begin(); idx != range.end(); ++idx) {
          int l = idx / std::max(1, P);
          int p = idx % std::max(1, P);
          if (l < 0 || l >= L || p < 0 || p >= P)
            continue;

          const auto &ls = left_samples[l];
          if (ls.ys.empty())
            continue;

          std::unordered_map<int, std::vector<std::pair<float, float>>> support;
          int repro_cnt = 0;

          const cv::Mat &coef = surfaces[p].coefficients;
          for (size_t i = 0; i < ls.ys.size(); ++i) {
            float y_f = ls.ys[i];
            float x_f = ls.xs[i];

            const cv::Point3f &ray = ls.rays[i];
            auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
            if (ips.empty())
              continue;

            cv::Point3f pt3;
            bool ok = false;
            for (auto &q : ips) {
              if (q.z > 100 && q.z < 1500) {
                pt3 = q;
                ok = true;
                break;
              }
            }
            if (!ok)
              continue;

            cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
            float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
            float yr =
                alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
            if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows)
              continue;

            ++repro_cnt;

            for (int r = 0; r < R; ++r) {
              const auto &rv = right_vec[r];
              if (rv.empty())
                continue;

              auto it =
                  std::lower_bound(rv.begin(), rv.end(), yr,
                                   [](const std::pair<float, LaserPoint> &a,
                                      float v) { return a.first < v; });
              if (it != rv.begin()) {
                auto prev = it - 1;
                if (it == rv.end() ||
                    std::fabs(prev->first - yr) < std::fabs(it->first - yr)) {
                  it = prev;
                }
              }
              if (it == rv.end())
                continue;
              if (std::fabs(it->first - yr) > EPS_)
                continue;

              float d = std::hypot(it->second.x - xr, it->second.y - yr);
              if (d > D_thresh_)
                continue;

              support[r].emplace_back(yr, d);
            }
          }

          // 生成增强候选
          for (auto &ent : support) {
            int r_idx = ent.first;
            auto vec = std::move(ent.second);
            if (vec.empty())
              continue;

            std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
              return a.first < b.first;
            });

            // 拆分子段
            std::vector<Interval> segs;
            int start = 0;
            for (int i = 1; i < static_cast<int>(vec.size()); ++i) {
              float gap = vec[i].first - vec[i - 1].first;
              if (gap > 2.0f * precision_ + EPS_) {
                Interval iv{alignToPrecision(vec[start].first),
                            alignToPrecision(vec[i - 1].first), i - start};
                segs.push_back(iv);
                start = i;
              }
            }
            Interval ivlast{alignToPrecision(vec[start].first),
                            alignToPrecision(vec.back().first),
                            static_cast<int>(vec.size()) - start};
            segs.push_back(ivlast);

            int total_count = 0;
            for (const auto &iv : segs)
              total_count += iv.count;
            if (total_count < MIN_LEN_)
              continue;

            // 收集评分数据
            std::vector<std::pair<float, float>> allpd;
            allpd.reserve(vec.size());
            for (auto &pd : vec) {
              for (auto &iv : segs) {
                if (pd.first >= iv.y_start - EPS_ &&
                    pd.first <= iv.y_end + EPS_) {
                  allpd.push_back(pd);
                  break;
                }
              }
            }

            // 使用改进的评分函数
            float std_dev = 0.0f, coverage = 0.0f;
            int right_point_count =
                static_cast<int>(laser_r[r_idx].points.size());
            float score =
                computeEnhancedScoreV2(allpd, ls.original_point_count,
                                       right_point_count, coverage, std_dev);

            if (score <= S_thresh_) {
              EnhancedCandidate cand;
              cand.match.l_idx = l;
              cand.match.p_idx = p;
              cand.match.r_idx = r_idx;
              cand.match.intervals = std::move(segs);
              cand.match.score = score;
              cand.match.coverage = coverage;
              cand.match.std_dev = std_dev;

              // 计算长度比
              cand.length_ratio = static_cast<float>(ls.original_point_count) /
                                  right_point_count;

              // 判断是否为高质量匹配：点数>=1000且得分<=1.5
              cand.is_high_quality =
                  (ls.original_point_count >= HIGH_QUALITY_POINT_THRESH) &&
                  (right_point_count >= HIGH_QUALITY_POINT_THRESH) &&
                  (score <= HIGH_QUALITY_SCORE_THRESH);

              all_enhanced_cands.push_back(std::move(cand));
            }

            if (l == 11 && r_idx == 4)
                printf("l%d-r%d, score: %.3f\n", l, r_idx, score);
          }
        }
      });

  // =============== 阶段2：高质量优先匹配 ===============

  std::vector<IntervalMatch> high_quality_matches;
  std::vector<bool> left_used(L, false), right_used(R, false);

  // 筛选并排序高质量候选
  std::vector<EnhancedCandidate> high_quality_cands;
  for (const auto &cand : all_enhanced_cands) {
    if (cand.is_high_quality) {
      high_quality_cands.push_back(cand);
    }
  }

  // 按得分排序，贪心选择最优匹配
  std::sort(high_quality_cands.begin(), high_quality_cands.end(),
            [](const auto &a, const auto &b) {
              return a.match.score < b.match.score;
            });

  for (const auto &cand : high_quality_cands) {
    int l_idx = cand.match.l_idx;
    int r_idx = cand.match.r_idx;
    if (!left_used[l_idx] && !right_used[r_idx]) {
      high_quality_matches.push_back(cand.match);
      left_used[l_idx] = true;
      right_used[r_idx] = true;
    }
  }

  // =============== 阶段3：剩余线约束匈牙利匹配 ===============

  const double INF_COST = 1e9;
  int n = std::max(L, R);
  std::vector<std::vector<double>> cost(n, std::vector<double>(n, INF_COST));
  std::vector<std::vector<IntervalMatch>> best_pair(
      L, std::vector<IntervalMatch>(R));
  std::vector<std::vector<bool>> has_best(L, std::vector<bool>(R, false));

  // 构建最优候选矩阵（包括所有候选，不仅仅是高质量的）
  for (const auto &cand : all_enhanced_cands) {
    int l = cand.match.l_idx;
    int r = cand.match.r_idx;
    if (l < 0 || l >= L || r < 0 || r >= R)
      continue;

    if (!has_best[l][r] || cand.match.score < best_pair[l][r].score) {
      best_pair[l][r] = cand.match;
      has_best[l][r] = true;
    }
  }

  // 填充约束成本矩阵
  for (int l = 0; l < L; ++l) {
    for (int r = 0; r < R; ++r) {
      if (left_used[l] || right_used[r]) {
        cost[l][r] = INF_COST;
        continue;
      }

      if (has_best[l][r]) {
        const auto &match = best_pair[l][r];

        // 长度一致性约束
        float left_len = static_cast<float>(laser_l[l].points.size());
        float right_len = static_cast<float>(laser_r[r].points.size());
        float ratio = left_len / right_len;

        if (ratio < 0.3f || ratio > 3.0f) {
          cost[l][r] = INF_COST; // 长度差异过大
        } else {
          // 基础得分 + 长度不一致惩罚
          float length_penalty = std::abs(1.0f - ratio) * 5.0f;
          cost[l][r] = static_cast<double>(match.score + length_penalty);
        }
      } else {
        cost[l][r] = INF_COST;
      }
    }
  }

  // 匈牙利求解
  std::vector<int> matchL;
  hungarian1(cost, matchL);

  // 收集第一轮匹配结果
  std::vector<IntervalMatch> first_round_matches = high_quality_matches;
  for (int i = 0; i < n; ++i) {
    int j = matchL[i];
    if (i < L && j >= 0 && j < R && !left_used[i] && !right_used[j]) {
      if (has_best[i][j] && best_pair[i][j].coverage >= 0.05f) {
        IntervalMatch im = best_pair[i][j];
        merge_intervals_vec(im.intervals);
        first_round_matches.push_back(std::move(im));
        left_used[i] = true;
        right_used[j] = true;
      }
    }
  }

  // =============== 阶段4：线段分割与二次匹配 ===============

  // 4.1 分割已匹配的线段，生成剩余可用段
  struct LaserSegment {
    int original_idx;
    std::map<float, LaserPoint> points;
    bool is_left;
    int original_point_count;
  };

  auto splitMatchedLines = [&](const std::vector<LaserLine> &lines,
                               const std::vector<IntervalMatch> &matches,
                               bool is_left) -> std::vector<LaserSegment> {
    std::vector<LaserSegment> segments;
    std::vector<bool> line_used(lines.size(), false);

    // 处理已匹配的线，分割出未匹配部分
    for (const auto &match : matches) {
      int line_idx = is_left ? match.l_idx : match.r_idx;
      if (line_idx < 0 || line_idx >= static_cast<int>(lines.size()))
        continue;

      line_used[line_idx] = true;
      const auto &line = lines[line_idx];
      std::map<float, LaserPoint> remaining_points = line.points;

      // 移除已匹配的区间
      for (const auto &interval : match.intervals) {
        auto it_start = remaining_points.lower_bound(interval.y_start - EPS_);
        auto it_end = remaining_points.upper_bound(interval.y_end + EPS_);
        remaining_points.erase(it_start, it_end);
      }

      // 将剩余点分割成连续段
      if (!remaining_points.empty()) {
        std::vector<std::map<float, LaserPoint>> continuous_segs;
        std::map<float, LaserPoint> current_seg;

        float prev_y = -1e6f;
        for (const auto &[y, point] : remaining_points) {
          if (y - prev_y > 2.0f * precision_ + EPS_ && !current_seg.empty()) {
            if (current_seg.size() >= MIN_LEN_) {
              continuous_segs.push_back(current_seg);
            }
            current_seg.clear();
          }
          current_seg[y] = point;
          prev_y = y;
        }

        if (current_seg.size() >= MIN_LEN_) {
          continuous_segs.push_back(current_seg);
        }

        // 添加有效段
        for (auto &seg_points : continuous_segs) {
          LaserSegment seg;
          seg.original_idx = line_idx;
          seg.points = std::move(seg_points);
          seg.is_left = is_left;
          seg.original_point_count = static_cast<int>(line.points.size());
          segments.push_back(seg);
        }
      }
    }

    // 添加完全未匹配的线
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
      if (!line_used[i] && lines[i].points.size() >= MIN_LEN_) {
        LaserSegment seg;
        seg.original_idx = i;
        seg.points = lines[i].points;
        seg.is_left = is_left;
        seg.original_point_count = static_cast<int>(lines[i].points.size());
        segments.push_back(seg);
      }
    }

    return segments;
  };

  std::vector<LaserSegment> left_segments =
      splitMatchedLines(laser_l, first_round_matches, true);
  std::vector<LaserSegment> right_segments =
      splitMatchedLines(laser_r, first_round_matches, false);

  // 4.2 对剩余段进行二次匹配
  std::vector<IntervalMatch> second_round_matches;

  if (!left_segments.empty() && !right_segments.empty()) {
    int SL = static_cast<int>(left_segments.size());
    int SR = static_cast<int>(right_segments.size());
    int sn = std::max(SL, SR);

    std::vector<std::vector<double>> seg_cost(
        sn, std::vector<double>(sn, INF_COST));
    std::vector<std::vector<IntervalMatch>> seg_best(
        SL, std::vector<IntervalMatch>(SR));
    std::vector<std::vector<bool>> seg_has_best(SL,
                                                std::vector<bool>(SR, false));

    // 为段间生成候选（使用所有光平面）
    tbb::concurrent_vector<EnhancedCandidate> seg_cands;

    tbb::parallel_for(
        tbb::blocked_range<int>(0, SL * SR * P),
        [&](const tbb::blocked_range<int> &range) {
          for (int idx = range.begin(); idx != range.end(); ++idx) {
            int sl = idx / (SR * P);
            int sr = (idx / P) % SR;
            int p = idx % P;

            if (sl >= SL || sr >= SR || p >= P)
              continue;

            const auto &left_seg = left_segments[sl];
            const auto &right_seg = right_segments[sr];
            const cv::Mat &coef = surfaces[p].coefficients;

            std::vector<std::pair<float, float>> seg_matches;
            int seg_repro_cnt = 0;

            for (const auto &[y_f, left_point] : left_seg.points) {
              float x_f = left_point.x;

              // 重投影计算
              cv::Point3f ray(
                  (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
                  (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l),
                  1.0f);
              float rn =
                  std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
              if (rn > 0.0f)
                ray *= (1.0f / rn);

              auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
              if (ips.empty())
                continue;

              cv::Point3f pt3;
              bool ok = false;
              for (auto &q : ips) {
                if (q.z > 100 && q.z < 1500) {
                  pt3 = q;
                  ok = true;
                  break;
                }
              }
              if (!ok)
                continue;

              cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y,
                             pt3.z);
              float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
              float yr = alignToPrecision(
                  static_cast<float>(fy_r * pr.y / pr.z + cy_r));
              if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows)
                continue;

              ++seg_repro_cnt;

              // 在右段中查找匹配点
              auto it = right_seg.points.lower_bound(yr - EPS_);
              if (it != right_seg.points.end() &&
                  std::fabs(it->first - yr) <= EPS_) {
                float d = std::hypot(it->second.x - xr, it->second.y - yr);
                if (d <= D_thresh_) {
                  seg_matches.emplace_back(yr, d);
                }
              }
            }

            if (seg_matches.size() >= MIN_LEN_) {
              // 生成区间
              std::sort(seg_matches.begin(), seg_matches.end());
              std::vector<Interval> seg_intervals;

              int start = 0;
              for (int i = 1; i < static_cast<int>(seg_matches.size()); ++i) {
                float gap = seg_matches[i].first - seg_matches[i - 1].first;
                if (gap > 2.0f * precision_ + EPS_) {
                  Interval iv{alignToPrecision(seg_matches[start].first),
                              alignToPrecision(seg_matches[i - 1].first),
                              i - start};
                  seg_intervals.push_back(iv);
                  start = i;
                }
              }
              Interval ivlast{alignToPrecision(seg_matches[start].first),
                              alignToPrecision(seg_matches.back().first),
                              static_cast<int>(seg_matches.size()) - start};
              seg_intervals.push_back(ivlast);

              // 评分（使用V2版本，无短线惩罚）
              float std_dev = 0.0f, coverage = 0.0f;
              float score = computeEnhancedScoreV2(
                  seg_matches, static_cast<int>(left_seg.points.size()),
                  static_cast<int>(right_seg.points.size()), coverage, std_dev);

              if (score <= S_thresh_) {
                EnhancedCandidate seg_cand;
                seg_cand.match.l_idx = left_seg.original_idx;
                seg_cand.match.r_idx = right_seg.original_idx;
                seg_cand.match.p_idx = p;
                seg_cand.match.intervals = std::move(seg_intervals);
                seg_cand.match.score = score;
                seg_cand.match.coverage = coverage;
                seg_cand.match.std_dev = std_dev;

                // 段匹配相关信息
                seg_cand.is_high_quality = false; // 段匹配不参与高质量判断
                seg_cand.length_ratio =
                    static_cast<float>(left_seg.points.size()) /
                    static_cast<float>(right_seg.points.size());

                seg_cands.push_back(std::move(seg_cand));
              }
                if (score < 3.0f)
                    printf("l%d-r%d, score: %.3f\n", left_seg.original_idx, right_seg.original_idx, score);
            }
          }
        });

    // 对段候选进行归约：保留每对(sl,sr)的最优候选
    for (const auto &cand : seg_cands) {
      // 需要将原始线索引映射回段索引
      int sl = -1, sr = -1;
      for (int i = 0; i < SL; ++i) {
        if (left_segments[i].original_idx == cand.match.l_idx) {
          sl = i;
          break;
        }
      }
      for (int i = 0; i < SR; ++i) {
        if (right_segments[i].original_idx == cand.match.r_idx) {
          sr = i;
          break;
        }
      }

      if (sl >= 0 && sr >= 0) {
        if (!seg_has_best[sl][sr] ||
            cand.match.score < seg_best[sl][sr].score) {
          seg_best[sl][sr] = cand.match;
          seg_has_best[sl][sr] = true;
        }
      }
    }

    // 填充段间成本矩阵
    for (int sl = 0; sl < SL; ++sl) {
      for (int sr = 0; sr < SR; ++sr) {
        if (seg_has_best[sl][sr]) {
          seg_cost[sl][sr] = static_cast<double>(seg_best[sl][sr].score);
        } else {
          seg_cost[sl][sr] = INF_COST;
        }
      }
    }

    // 段间匈牙利匹配
    std::vector<int> seg_matchL;
    hungarian1(seg_cost, seg_matchL);

    // 收集段间匹配结果
    for (int i = 0; i < sn; ++i) {
      int j = seg_matchL[i];
      if (i < SL && j >= 0 && j < SR && seg_has_best[i][j]) {
        IntervalMatch im = seg_best[i][j];
        merge_intervals_vec(im.intervals);
        second_round_matches.push_back(std::move(im));
      }
    }
  }

  // =============== 合并所有匹配结果 ===============

  std::vector<IntervalMatch> final_matches;
  final_matches.reserve(first_round_matches.size() +
                        second_round_matches.size());

  // 添加第一轮匹配
  for (auto &match : first_round_matches) {
    merge_intervals_vec(match.intervals);
    final_matches.push_back(std::move(match));
  }

  //   添加第二轮匹配
  for (auto &match : second_round_matches) {
    final_matches.push_back(std::move(match));
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  cv::Mat vis_all;
  cv::hconcat(rectify_l, rectify_r, vis_all);
  if (vis_all.channels() == 1)
    cv::cvtColor(vis_all, vis_all, cv::COLOR_GRAY2BGR);
  int off_all = rectify_l.cols;

  // 首先绘制所有匹配的线段（使用原有颜色）
  static const std::vector<cv::Scalar> palette30 = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},  {255, 255, 0}, {255, 0, 255},
      {0, 255, 255}, {128, 0, 0},   {0, 128, 0},  {0, 0, 128},   {128, 128, 0},
      {128, 0, 128}, {0, 128, 128}, {64, 0, 128}, {128, 64, 0},  {0, 128, 64},
      {64, 128, 0},  {0, 64, 128},  {128, 0, 64}, {192, 192, 0}, {192, 0, 192},
      {64, 255, 0},  {255, 64, 0},  {0, 64, 255}, {0, 255, 64},  {255, 0, 64},
      {64, 0, 255},  {192, 0, 64},  {64, 192, 0}, {0, 192, 64},  {64, 0, 192}};
  for (int idx = 0; idx < (int)final_matches.size(); ++idx) {
    auto &m = final_matches[idx];
    cv::Scalar col = palette30[idx % palette30.size()];
    // 左图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itL = laser_l[m.l_idx].points.lower_bound(y0);
        if (itL != laser_l[m.l_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itL);
          if (itL == laser_l[m.l_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itL->first - y0)) {
            itL = prev;
          }
        }
        if (fabs(itL->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itL->second.x, y0), 2, col, -1);
      }
    // 右图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itR = laser_r[m.r_idx].points.lower_bound(y0);
        if (itR != laser_r[m.r_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itR);
          if (itR == laser_r[m.r_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itR->first - y0)) {
            itR = prev;
          }
        }
        if (fabs(itR->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itR->second.x + off_all, y0), 2, col,
                   -1);
      }
  }

  // 为所有左图激光线添加ID标签（绿色）
  for (int i = 0; i < laser_l.size(); ++i) {
    if (laser_l[i].points.empty())
      continue;

    // 计算整条线的中点位置
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (auto &point : laser_l[i].points) {
      if (point.first < min_y)
        min_y = point.first;
      if (point.first > max_y)
        max_y = point.first;
    }

    float mid_y = (min_y + max_y) / 2.0f;

    // 找到中点对应的x坐标
    auto itL = laser_l[i].points.lower_bound(mid_y);
    if (itL != laser_l[i].points.begin()) {
      auto prev = std::prev(itL);
      if (itL == laser_l[i].points.end() ||
          fabs(prev->first - mid_y) < fabs(itL->first - mid_y)) {
        itL = prev;
      }
    }

    if (fabs(itL->first - mid_y) <= EPS_) {
      std::string labelL = "L" + std::to_string(i);
      cv::putText(vis_all, labelL, cv::Point(itL->second.x, mid_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0),
                  2.5); // 绿色
    }
  }

  // 为所有右图激光线添加ID标签（绿色）
  for (int i = 0; i < laser_r.size(); ++i) {
    if (laser_r[i].points.empty())
      continue;

    // 计算整条线的中点位置
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (auto &point : laser_r[i].points) {
      if (point.first < min_y)
        min_y = point.first;
      if (point.first > max_y)
        max_y = point.first;
    }

    float mid_y = (min_y + max_y) / 2.0f;

    // 找到中点对应的x坐标
    auto itR = laser_r[i].points.lower_bound(mid_y);
    if (itR != laser_r[i].points.begin()) {
      auto prev = std::prev(itR);
      if (itR == laser_r[i].points.end() ||
          fabs(prev->first - mid_y) < fabs(itR->first - mid_y)) {
        itR = prev;
      }
    }

    if (fabs(itR->first - mid_y) <= EPS_) {
      std::string labelR = "R" + std::to_string(i);
      cv::putText(
          vis_all, labelR, cv::Point(itR->second.x + off_all, mid_y - 5),
          cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2.5); // 绿色
    }
  }

  cv::imshow("vis_img", vis_all);
  cv::waitKey(0);
  cv::destroyWindow("vis_img");

//   static int dbg_idx = 0;
//   cv::imwrite("debug_img/match_res_" + std::to_string(dbg_idx) + ".png",
//               vis_all);
//   dbg_idx++;
#endif

  return final_matches;
}

/************************************* MatchV9
 * *************************************** */

void LaserProcessor::removeMatchedIntervals(
    LaserLine &line, const std::vector<Interval> &matched_intervals) {
  if (matched_intervals.empty() || line.points.empty())
    return;

  // 由于line.points是map<float, LaserPoint>，我们需要小心地移除区间内的点
  std::vector<float> keys_to_remove;

  // 遍历所有匹配区间
  for (const auto &interval : matched_intervals) {
    // 找到在区间[y_start, y_end]内的所有点
    auto it = line.points.lower_bound(interval.y_start - EPS_);
    while (it != line.points.end() && it->first <= interval.y_end + EPS_) {
      keys_to_remove.push_back(it->first);
      ++it;
    }
  }

  // 移除这些点
  for (float key : keys_to_remove) {
    line.points.erase(key);
  }
}

std::vector<IntervalMatch> LaserProcessor::generateCandidatesForRemaining(
    const std::vector<LaserLine> &left_lines,
    const std::vector<LaserLine> &right_lines,
    const std::vector<QuadSurface> &surfaces, const CalibrationResult &calib) {

  std::vector<IntervalMatch> candidates;
  const int img_rows = 1200; // 需要根据实际情况获取
  const int img_cols = 2048; // 需要根据实际情况获取

  // 相机内参
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  const int L = static_cast<int>(left_lines.size());
  const int R = static_cast<int>(right_lines.size());
  const int P = static_cast<int>(surfaces.size());

  // 预处理右线数据
  std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
  for (int r = 0; r < R; ++r) {
    right_vec[r].reserve(right_lines[r].points.size());
    for (const auto &kv : right_lines[r].points) {
      right_vec[r].emplace_back(kv.first, kv.second);
    }
  }

  // 预计算左线采样点和射线
  struct LeftSample {
    std::vector<float> ys, xs;
    std::vector<cv::Point3f> rays;
    int original_point_count;
  };
  std::vector<LeftSample> left_samples(L);
  for (int l = 0; l < L; ++l) {
    const auto &mp = left_lines[l].points;
    auto &ls = left_samples[l];
    ls.original_point_count = static_cast<int>(mp.size());
    ls.ys.reserve(mp.size());
    ls.xs.reserve(mp.size());
    ls.rays.reserve(mp.size());
    for (const auto &kv : mp) {
      float y_f = kv.first;
      float x_f = kv.second.x;
      ls.ys.push_back(y_f);
      ls.xs.push_back(x_f);
      cv::Point3f ray(
          (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
          (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
      float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      if (rn > 0.0f)
        ray *= (1.0f / rn);
      ls.rays.push_back(ray);
    }
  }

  // 预计算右线点数
  std::vector<int> right_point_counts(R);
  for (int r = 0; r < R; ++r) {
    right_point_counts[r] = static_cast<int>(right_lines[r].points.size());
  }

  // 生成候选 - 串行版本（因为剩余线数量较少）
  for (int l = 0; l < L; ++l) {
    for (int p = 0; p < P; ++p) {
      const auto &ls = left_samples[l];
      if (ls.ys.empty())
        continue;

      // 支持集合：r -> vector<(yr,d)>
      std::unordered_map<int, std::vector<std::pair<float, float>>> support;
      int repro_cnt = ls.original_point_count;

      const cv::Mat &coef = surfaces[p].coefficients;
      for (size_t i = 0; i < ls.ys.size(); ++i) {
        float y_f = ls.ys[i];
        float x_f = ls.xs[i];

        // 与平面求交
        const cv::Point3f &ray = ls.rays[i];
        auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
        if (ips.empty())
          continue;
        cv::Point3f pt3;
        bool ok = false;
        for (auto &q : ips) {
          if (q.z > 100 && q.z < 1500) {
            pt3 = q;
            ok = true;
            break;
          }
        }
        if (!ok)
          continue;

        cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
        float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
        float yr =
            alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
        if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows)
          continue;

        // 遍历右线寻找匹配
        for (int r = 0; r < R; ++r) {
          const auto &rv = right_vec[r];
          if (rv.empty())
            continue;

          auto it = std::lower_bound(rv.begin(), rv.end(), yr,
                                     [](const std::pair<float, LaserPoint> &a,
                                        float v) { return a.first < v; });

          // 找到最近的点
          if (it != rv.begin()) {
            auto prev = it - 1;
            if (it == rv.end() ||
                std::fabs(prev->first - yr) < std::fabs(it->first - yr)) {
              it = prev;
            }
          }
          if (it == rv.end())
            continue;
          if (std::fabs(it->first - yr) > EPS_)
            continue;

          float d = std::hypot(it->second.x - xr, it->second.y - yr);
          if (d > D_thresh_)
            continue;

          support[r].emplace_back(yr, d);
        }
      }

      // 处理支持集合
      for (auto &ent : support) {
        int r_idx = ent.first;
        auto vec = std::move(ent.second);
        if (vec.empty())
          continue;

        std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
          return a.first < b.first;
        });

        // 拆分子段
        std::vector<Interval> segs;
        int start = 0;
        for (int i = 1; i < (int)vec.size(); ++i) {
          float gap = vec[i].first - vec[i - 1].first;
          if (gap > 2.0f * precision_ + EPS_) {
            Interval iv{alignToPrecision(vec[start].first),
                        alignToPrecision(vec[i - 1].first), i - start};
            segs.push_back(iv);
            start = i;
          }
        }
        Interval ivlast{alignToPrecision(vec[start].first),
                        alignToPrecision(vec.back().first),
                        (int)vec.size() - start};
        segs.push_back(ivlast);

        int total_count = 0;
        for (const auto &iv : segs)
          total_count += iv.count;
        if (total_count < MIN_LEN_)
          continue;

        // 收集评分数据
        std::vector<std::pair<float, float>> allpd;
        allpd.reserve(vec.size());
        for (auto &pd : vec) {
          for (auto &iv : segs) {
            if (pd.first >= iv.y_start - EPS_ && pd.first <= iv.y_end + EPS_) {
              allpd.push_back(pd);
              break;
            }
          }
        }

        float std_dev = 0.0f, coverage = 0.0f;
        float score = computeEnhancedScoreV2(allpd, ls.original_point_count,
                                             right_point_counts[r_idx],
                                             coverage, std_dev);
        if (score <= S_thresh_) {
          IntervalMatch c;
          c.l_idx = l;
          c.p_idx = p;
          c.r_idx = r_idx;
          c.intervals = std::move(segs);
          c.score = score;
          c.coverage = coverage;
          c.std_dev = std_dev;
          candidates.push_back(std::move(c));
        }
      }
    }
  }

  return candidates;
}

std::vector<IntervalMatch>
LaserProcessor::match9(const std::vector<LaserLine> &laser_l,
                       const std::vector<LaserLine> &laser_r,
                       const cv::Mat &rectify_l, const cv::Mat &rectify_r) {
  const int img_rows = rectify_l.rows;
  const int img_cols = rectify_l.cols;
  const auto calib = ConfigManager::getInstance().getCalibInfo();
  const auto surfaces = ConfigManager::getInstance().getQuadSurfaces();

  // 结构尺寸
  const int L = static_cast<int>(laser_l.size());
  const int R = static_cast<int>(laser_r.size());
  const int P = static_cast<int>(surfaces.size());

  // 相机内参
  double fx_l = calib.P[0].at<double>(0, 0), fy_l = calib.P[0].at<double>(1, 1);
  double cx_l = calib.P[0].at<double>(0, 2), cy_l = calib.P[0].at<double>(1, 2);
  double fx_r = calib.P[1].at<double>(0, 0), fy_r = calib.P[1].at<double>(1, 1);
  double cx_r = calib.P[1].at<double>(0, 2), cy_r = calib.P[1].at<double>(1, 2);
  double baseline = -calib.P[1].at<double>(0, 3) / fx_r;

  // --------------- 预处理 ---------------
  std::vector<std::vector<std::pair<float, LaserPoint>>> right_vec(R);
  for (int r = 0; r < R; ++r) {
    right_vec[r].reserve(laser_r[r].points.size());
    for (const auto &kv : laser_r[r].points)
      right_vec[r].emplace_back(kv.first, kv.second);
  }

  // 预计算每条左线的采样点数组与归一化射线
  struct LeftSample {
    std::vector<float> ys, xs;
    std::vector<cv::Point3f> rays;
    int original_point_count;
  };
  std::vector<LeftSample> left_samples(L);
  for (int l = 0; l < L; ++l) {
    const auto &mp = laser_l[l].points;
    auto &ls = left_samples[l];
    ls.original_point_count = static_cast<int>(mp.size());
    ls.ys.reserve(mp.size());
    ls.xs.reserve(mp.size());
    ls.rays.reserve(mp.size());
    for (const auto &kv : mp) {
      float y_f = kv.first;
      float x_f = kv.second.x;
      ls.ys.push_back(y_f);
      ls.xs.push_back(x_f);
      cv::Point3f ray(
          (x_f - static_cast<float>(cx_l)) / static_cast<float>(fx_l),
          (y_f - static_cast<float>(cy_l)) / static_cast<float>(fy_l), 1.0f);
      float rn = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      if (rn > 0.0f)
        ray *= (1.0f / rn);
      ls.rays.push_back(ray);
    }
  }

  // 预计算右线点数
  std::vector<int> right_point_counts(R);
  for (int r = 0; r < R; ++r) {
    right_point_counts[r] = static_cast<int>(laser_r[r].points.size());
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  // 全局可视化底图
  cv::Mat vis_global;
  cv::hconcat(rectify_l, rectify_r, vis_global);
  if (vis_global.channels() == 1)
    cv::cvtColor(vis_global, vis_global, cv::COLOR_GRAY2BGR);
  int off = rectify_l.cols;
  static const cv::Scalar proc_interval_col(0, 255, 255); // 候选区间颜色
  static const cv::Scalar r_laser_col(255, 0, 0);         // 整条右线颜色
  cv::namedWindow("vis_img", cv::WINDOW_NORMAL);
  cv::resizeWindow("vis_img", vis_global.cols, vis_global.rows);
#endif

  // --------------- 步骤1: 初始全候选生成 ---------------
  tbb::concurrent_vector<IntervalMatch> all_cands;
  const int total_tasks = L * P;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, total_tasks),
      [&](const tbb::blocked_range<int> &range) {
        for (int idx = range.begin(); idx != range.end(); ++idx) {
          int l = idx / std::max(1, P);
          int p = idx % std::max(1, P);
          if (l < 0 || l >= L || p < 0 || p >= P)
            continue;
          const auto &ls = left_samples[l];
          if (ls.ys.empty())
            continue;

          // 支持集合：r -> vector<(yr,d)>
          std::unordered_map<int, std::vector<std::pair<float, float>>> support;
          int repro_cnt = ls.original_point_count;

          const cv::Mat &coef = surfaces[p].coefficients;
          for (size_t i = 0; i < ls.ys.size(); ++i) {
            float y_f = ls.ys[i];
            float x_f = ls.xs[i];

            // 与平面求交
            const cv::Point3f &ray = ls.rays[i];
            auto ips = findIntersection(cv::Point3f(0, 0, 0), ray, coef);
            if (ips.empty())
              continue;
            cv::Point3f pt3;
            bool ok = false;
            for (auto &q : ips)
              if (q.z > 100 && q.z < 1500) {
                pt3 = q;
                ok = true;
                break;
              }
            if (!ok)
              continue;

            cv::Point3f pr(pt3.x - static_cast<float>(baseline), pt3.y, pt3.z);
            float xr = static_cast<float>(fx_r * pr.x / pr.z + cx_r);
            float yr =
                alignToPrecision(static_cast<float>(fy_r * pr.y / pr.z + cy_r));
            if (xr < 0 || xr >= img_cols || yr < 0 || yr >= img_rows)
              continue;

            // 遍历右线
            for (int r = 0; r < R; ++r) {
              const auto &rv = right_vec[r];
              if (rv.empty())
                continue;
              auto it =
                  std::lower_bound(rv.begin(), rv.end(), yr,
                                   [](const std::pair<float, LaserPoint> &a,
                                      float v) { return a.first < v; });
              if (it != rv.begin()) {
                auto prev = it - 1;
                if (it == rv.end() ||
                    std::fabs(prev->first - yr) < std::fabs(it->first - yr))
                  it = prev;
              }
              if (it == rv.end())
                continue;
              if (std::fabs(it->first - yr) > EPS_)
                continue;
              float d = std::hypot(it->second.x - xr, it->second.y - yr);
              if (d > D_thresh_)
                continue;

              support[r].emplace_back(yr, d);
            }
          }

          // 支持集合转子段并打分
          for (auto &ent : support) {
            int r_idx = ent.first;
            auto vec = std::move(ent.second);
            if (vec.empty())
              continue;
            std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
              return a.first < b.first;
            });

            // 拆分子段
            std::vector<Interval> segs;
            int start = 0;
            for (int i = 1; i < (int)vec.size(); ++i) {
              float gap = vec[i].first - vec[i - 1].first;
              if (gap > 2.0f * precision_ + EPS_) {
                Interval iv{alignToPrecision(vec[start].first),
                            alignToPrecision(vec[i - 1].first), i - start};
                segs.push_back(iv);
                start = i;
              }
            }
            Interval ivlast{alignToPrecision(vec[start].first),
                            alignToPrecision(vec.back().first),
                            (int)vec.size() - start};
            segs.push_back(ivlast);

            int total_count = 0;
            for (const auto &iv : segs)
              total_count += iv.count;
            if (total_count < MIN_LEN_)
              continue;

            // 收集该子段内的 (y,d) 用于评分
            std::vector<std::pair<float, float>> allpd;
            allpd.reserve(vec.size());
            for (auto &pd : vec) {
              for (auto &iv : segs) {
                if (pd.first >= iv.y_start - EPS_ &&
                    pd.first <= iv.y_end + EPS_) {
                  allpd.push_back(pd);
                  break;
                }
              }
            }

            float std_dev = 0.0f, coverage = 0.0f;
            // 使用新的评分函数，考虑左右线长度
            float score = computeEnhancedScoreV2(allpd, ls.original_point_count,
                                                 right_point_counts[r_idx],
                                                 coverage, std_dev);
            if (score <= S_thresh_) {
              IntervalMatch c;
              c.l_idx = l;
              c.p_idx = p;
              c.r_idx = r_idx;
              c.intervals = std::move(segs);
              c.score = score;
              c.coverage = coverage;
              c.std_dev = std_dev;
              all_cands.push_back(std::move(c));
            }
          }
        }
      });

  // --------------- 步骤2: 高质量匹配优先处理 ---------------
  std::vector<IntervalMatch> final_matches;
  std::vector<bool> left_matched(L, false);
  std::vector<bool> right_matched(R, false);

  // 高质量匹配条件：点数>1000且得分较好
  const int HIGH_QUALITY_POINTS = 1000;
  const float HIGH_QUALITY_SCORE_THRESH = S_thresh_;
  // 收集高质量候选
  std::vector<IntervalMatch> high_quality_cands;
  for (const auto &cand : all_cands) {
    int left_points = left_samples[cand.l_idx].original_point_count;
    int right_points = right_point_counts[cand.r_idx];
    if (left_points > HIGH_QUALITY_POINTS &&
        right_points > HIGH_QUALITY_POINTS &&
        cand.score < HIGH_QUALITY_SCORE_THRESH) {
      high_quality_cands.push_back(cand);
    }
  }

  // 对高质量候选按分数排序，优先处理分数最好的
  std::sort(high_quality_cands.begin(), high_quality_cands.end(),
            [](const IntervalMatch &a, const IntervalMatch &b) {
              return a.score < b.score;
            });

  // 选择高质量匹配（避免冲突）
  for (const auto &cand : high_quality_cands) {
    if (!left_matched[cand.l_idx] && !right_matched[cand.r_idx]) {
      final_matches.push_back(cand);
      left_matched[cand.l_idx] = true;
      right_matched[cand.r_idx] = true;
    }
  }

  // --------------- 步骤3: 第一次匈牙利匹配（剩余线） ---------------
  const double INF_COST = 1e9;
  int n = std::max(L, R);
  std::vector<std::vector<double>> cost(n, std::vector<double>(n, INF_COST));
  std::vector<std::vector<IntervalMatch>> best_pair(
      L, std::vector<IntervalMatch>(R));
  std::vector<std::vector<bool>> has_best(L, std::vector<bool>(R, false));

  // 只考虑未匹配的线
  for (const auto &c : all_cands) {
    if (c.l_idx < 0 || c.l_idx >= L || c.r_idx < 0 || c.r_idx >= R)
      continue;
    if (left_matched[c.l_idx] || right_matched[c.r_idx])
      continue;

    if (!has_best[c.l_idx][c.r_idx] ||
        c.score < best_pair[c.l_idx][c.r_idx].score) {
      best_pair[c.l_idx][c.r_idx] = c;
      has_best[c.l_idx][c.r_idx] = true;
    }
  }

  // 填充cost矩阵
  for (int l = 0; l < L; ++l) {
    if (left_matched[l])
      continue;
    for (int r = 0; r < R; ++r) {
      if (right_matched[r])
        continue;
      if (has_best[l][r]) {
        cost[l][r] = static_cast<double>(best_pair[l][r].score);
      }
    }
  }

  // 匈牙利求解
  std::vector<int> matchL;
  hungarian1(cost, matchL);

  // 添加第一次匈牙利匹配结果
  for (int i = 0; i < n; ++i) {
    int j = matchL[i];
    if (i < L && j >= 0 && j < R && !left_matched[i] && !right_matched[j]) {
      if (has_best[i][j] && best_pair[i][j].coverage >= 0.05f) {
        IntervalMatch im = best_pair[i][j];
        merge_intervals_vec(im.intervals);
        final_matches.push_back(std::move(im));
        left_matched[i] = true;
        right_matched[j] = true;
      }
    }
  }

  // --------------- 步骤4: 更新可用区域并第二次匈牙利匹配 ---------------
  // 创建剩余区域 - 复制原始数据
  std::vector<LaserLine> remaining_left = laser_l;
  std::vector<LaserLine> remaining_right = laser_r;

  // 从剩余左线中移除已匹配的区间
  for (const auto &match : final_matches) {
    if (match.l_idx < remaining_left.size()) {
      removeMatchedIntervals(remaining_left[match.l_idx], match.intervals);
    }
  }

  // 过滤掉点数太少的线或整个标记为已匹配的线
  std::vector<int> valid_left_indices;
  std::vector<int> valid_right_indices;
  std::vector<LaserLine> valid_left_lines;
  std::vector<LaserLine> valid_right_lines;

  for (int i = 0; i < L; ++i) {
    // 只考虑未匹配且点数足够的左线
    if (!left_matched[i] && remaining_left[i].points.size() >= MIN_LEN_) {
      valid_left_indices.push_back(i);
      valid_left_lines.push_back(remaining_left[i]);
    }
  }

  for (int i = 0; i < R; ++i) {
    // 只考虑未匹配且点数足够的右线
    if (!right_matched[i] && remaining_right[i].points.size() >= MIN_LEN_) {
      valid_right_indices.push_back(i);
      valid_right_lines.push_back(remaining_right[i]);
    }
  }

  // 如果还有剩余可匹配的线，进行第二次匹配
  if (!valid_left_lines.empty() && !valid_right_lines.empty()) {
    // 为剩余区域生成候选
    auto remaining_candidates = generateCandidatesForRemaining(
        valid_left_lines, valid_right_lines, surfaces, calib);

    // 第二次匈牙利匹配
    int rem_L = static_cast<int>(valid_left_lines.size());
    int rem_R = static_cast<int>(valid_right_lines.size());
    int rem_n = std::max(rem_L, rem_R);

    if (!remaining_candidates.empty()) {
      std::vector<std::vector<double>> rem_cost(
          rem_n, std::vector<double>(rem_n, INF_COST));
      std::vector<std::vector<IntervalMatch>> rem_best_pair(
          rem_L, std::vector<IntervalMatch>(rem_R));
      std::vector<std::vector<bool>> rem_has_best(
          rem_L, std::vector<bool>(rem_R, false));

      // 填充最佳候选对
      for (const auto &cand : remaining_candidates) {
        if (cand.l_idx < 0 || cand.l_idx >= rem_L || cand.r_idx < 0 ||
            cand.r_idx >= rem_R)
          continue;
        if (!rem_has_best[cand.l_idx][cand.r_idx] ||
            cand.score < rem_best_pair[cand.l_idx][cand.r_idx].score) {
          rem_best_pair[cand.l_idx][cand.r_idx] = cand;
          rem_has_best[cand.l_idx][cand.r_idx] = true;
        }
      }

      // 构建cost矩阵
      for (int l = 0; l < rem_L; ++l) {
        for (int r = 0; r < rem_R; ++r) {
          if (rem_has_best[l][r]) {
            rem_cost[l][r] = static_cast<double>(rem_best_pair[l][r].score);
          }
        }
      }

      // 匈牙利算法求解
      std::vector<int> rem_matchL;
      hungarian1(rem_cost, rem_matchL);

      // 添加第二次匹配结果
      for (int i = 0; i < rem_n; ++i) {
        int j = rem_matchL[i];
        if (i < rem_L && j >= 0 && j < rem_R) {
          if (rem_has_best[i][j] && rem_best_pair[i][j].coverage >= 0.05f) {
            IntervalMatch im = rem_best_pair[i][j];
            // 恢复原始索引
            im.l_idx = valid_left_indices[i];
            im.r_idx = valid_right_indices[j];
            merge_intervals_vec(im.intervals);
            final_matches.push_back(std::move(im));
          }
        }
      }
    }
  }

#ifdef DEBUG_PLANE_MATCH_FINAL
  cv::Mat vis_all;
  cv::hconcat(rectify_l, rectify_r, vis_all);
  if (vis_all.channels() == 1)
    cv::cvtColor(vis_all, vis_all, cv::COLOR_GRAY2BGR);
  int off_all = rectify_l.cols;
  static const std::vector<cv::Scalar> palette30 = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},  {255, 255, 0}, {255, 0, 255},
      {0, 255, 255}, {128, 0, 0},   {0, 128, 0},  {0, 0, 128},   {128, 128, 0},
      {128, 0, 128}, {0, 128, 128}, {64, 0, 128}, {128, 64, 0},  {0, 128, 64},
      {64, 128, 0},  {0, 64, 128},  {128, 0, 64}, {192, 192, 0}, {192, 0, 192},
      {64, 255, 0},  {255, 64, 0},  {0, 64, 255}, {0, 255, 64},  {255, 0, 64},
      {64, 0, 255},  {192, 0, 64},  {64, 192, 0}, {0, 192, 64},  {64, 0, 192}};
  for (int idx = 0; idx < (int)final_matches.size(); ++idx) {
    auto &m = final_matches[idx];
    cv::Scalar col = palette30[idx % palette30.size()];
    // 左图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itL = laser_l[m.l_idx].points.lower_bound(y0);
        if (itL != laser_l[m.l_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itL);
          if (itL == laser_l[m.l_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itL->first - y0)) {
            itL = prev;
          }
        }
        if (fabs(itL->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itL->second.x, y0), 2, col, -1);
      }
    // 右图区间点
    for (auto &iv : m.intervals)
      for (float y0 = iv.y_start; y0 <= iv.y_end; y0 += precision_) {
        auto itR = laser_r[m.r_idx].points.lower_bound(y0);
        if (itR != laser_r[m.r_idx].points.begin()) {
          // 比较 it 和前一个元素哪个更接近 y0
          auto prev = std::prev(itR);
          if (itR == laser_r[m.r_idx].points.end() ||
              fabs(prev->first - y0) < fabs(itR->first - y0)) {
            itR = prev;
          }
        }
        if (fabs(itR->first - y0) > EPS_)
          continue;
        cv::circle(vis_all, cv::Point2f(itR->second.x + off_all, y0), 2, col,
                   -1);
      }

    // 计算整条线的中点位置（而不是每个区间的中点）
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    // 找到所有区间的最小和最大y值
    for (auto &iv : m.intervals) {
      if (iv.y_start < min_y)
        min_y = iv.y_start;
      if (iv.y_end > max_y)
        max_y = iv.y_end;
    }

    // 计算整条线的中点
    float mid_y = (min_y + max_y) / 2.0f;

    // 左图标签
    auto itL = laser_l[m.l_idx].points.lower_bound(mid_y);
    if (itL != laser_l[m.l_idx].points.begin()) {
      auto prev = std::prev(itL);
      if (itL == laser_l[m.l_idx].points.end() ||
          fabs(prev->first - mid_y) < fabs(itL->first - mid_y)) {
        itL = prev;
      }
    }
    if (fabs(itL->first - mid_y) <= EPS_) {
      std::string labelL = "L" + std::to_string(m.l_idx);
      cv::putText(vis_all, labelL, cv::Point(itL->second.x, mid_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2.5); // 绿色粗体
    }

    // 右图标签
    auto itR = laser_r[m.r_idx].points.lower_bound(mid_y);
    if (itR != laser_r[m.r_idx].points.begin()) {
      auto prev = std::prev(itR);
      if (itR == laser_r[m.r_idx].points.end() ||
          fabs(prev->first - mid_y) < fabs(itR->first - mid_y)) {
        itR = prev;
      }
    }
    if (fabs(itR->first - mid_y) <= EPS_) {
      std::string labelR = "R" + std::to_string(m.r_idx);
      cv::putText(vis_all, labelR,
                  cv::Point(itR->second.x + off_all, mid_y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 1.5, col, 2.5); // 绿色粗体
    }
  }
  cv::imshow("vis_img", vis_all);
  cv::waitKey(0);
  cv::destroyWindow("vis_img");

  static int dbg_idx = 0;
  cv::imwrite("debug_img/match_res_" + std::to_string(dbg_idx) + ".png",
              vis_all);
  dbg_idx++;
#endif

  return final_matches;
}

/************************************** 同名点匹配
 * **************************************/
std::vector<cv::Point3f> LaserProcessor::generateCloudPoints(
    const std::vector<std::tuple<int, int, int>> &laser_match,
    const std::vector<LaserLine> laser_l,
    const std::vector<LaserLine> laser_r) {

  // 记录点云生成结果
  std::vector<cv::Point3f> cloud_points;

  // 标定参数
  const auto calib_info = ConfigManager::getInstance().getCalibInfo();
  const float cx1 = calib_info.P[0].at<double>(0, 2);
  const float cy1 = calib_info.P[0].at<double>(1, 2);
  const float cx2 = calib_info.P[1].at<double>(0, 2);

  for (const auto &[l_idx, p_idx, r_idx] : laser_match) {
    // 划分左右图像的点
    std::vector<cv::Point2f> left_points, right_points;
    for (const auto &p : laser_l[l_idx].points) {
      const auto it = laser_r[r_idx].points.find(p.second.y);
      if (it == laser_r[r_idx].points.end())
        continue;

      // if (p.second.x >= 995 && p.second.x <= 1034 && p.second.y >= 560 &&
      // p.second.y <= 591)
      //     puts("");

      left_points.emplace_back(p.second.x, p.second.y);
      right_points.emplace_back(it->second.x, it->second.y);
    }

    // 三角测量
    cv::Mat points4D;
    cv::triangulatePoints(calib_info.P[0], calib_info.P[1], left_points,
                          right_points, points4D);

    // 转换为3D点
    for (int i = 0; i < points4D.cols; i++) {
      cv::Point3f point;
      float w = points4D.at<float>(3, i);
      if (std::abs(w) < 1e-6) {
        std::cerr << "警告: 点 " << i << " 重建可能不准确 (w≈0)" << std::endl;
        continue;
      }
      point.x = points4D.at<float>(0, i) / w;
      point.y = points4D.at<float>(1, i) / w;
      point.z = points4D.at<float>(2, i) / w;
      cloud_points.emplace_back(point);
    }
  }

  // std::ofstream ofs("cloudpoints.txt");
  // for (const auto& pt : cloud_points) {
  //     ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
  // }
  // std::cout << "点云已保存到 " << "cloudpoints.txt" << std::endl;

  return cloud_points;
}

std::vector<cv::Point3f>
LaserProcessor::generateCloudPoints2(const std::vector<IntervalMatch> &matches,
                                     const std::vector<LaserLine> &laser_l,
                                     const std::vector<LaserLine> &laser_r) {
  std::vector<cv::Point3f> cloud;
  const auto calib_info = ConfigManager::getInstance().getCalibInfo();

  // 遍历每条匹配区间
  for (const auto &m : matches) {
    std::vector<cv::Point2f> lp, rp;
    // 左右同一区间内点对
    for (const auto &iv : m.intervals) {
      for (float y = iv.y_start; y <= iv.y_end; y += precision_) {
        // 左侧
        auto itL = laser_l[m.l_idx].points.lower_bound(y);
        if (itL != laser_l[m.l_idx].points.begin()) {
          auto prevL = std::prev(itL);
          if (itL == laser_l[m.l_idx].points.end() ||
              fabs(prevL->first - y) < fabs(itL->first - y)) {
            itL = prevL;
          }
        }
        // 右侧
        auto itR = laser_r[m.r_idx].points.lower_bound(y);
        if (itR != laser_r[m.r_idx].points.begin()) {
          auto prevR = std::prev(itR);
          if (itR == laser_r[m.r_idx].points.end() ||
              fabs(prevR->first - y) < fabs(itR->first - y)) {
            itR = prevR;
          }
        }
        // 判断是否在精度范围内
        if (itL != laser_l[m.l_idx].points.end() &&
            fabs(itL->first - y) < EPS_ &&
            itR != laser_r[m.r_idx].points.end() &&
            fabs(itR->first - y) < EPS_) {
          lp.emplace_back(itL->second.x, y);
          rp.emplace_back(itR->second.x, y);
        }
      }
    }
    if (lp.empty())
      continue;
    cv::Mat pts4;
    cv::triangulatePoints(calib_info.P[0], calib_info.P[1], lp, rp, pts4);
    for (int i = 0; i < pts4.cols; ++i) {
      float w = pts4.at<float>(3, i);
      if (fabs(w) < 1e-6)
        continue;
      cloud.emplace_back(pts4.at<float>(0, i) / w, pts4.at<float>(1, i) / w,
                         pts4.at<float>(2, i) / w);
    }
  }
  return cloud;
}

std::vector<cv::Point3f>
LaserProcessor::generateCloudPoints3(const std::vector<IntervalMatch> &matches,
                                     const std::vector<LaserLine2> &laser_l,
                                     const std::vector<LaserLine2> &laser_r) {
  std::vector<cv::Point3f> cloud;
  const auto calib_info = ConfigManager::getInstance().getCalibInfo();

  // 遍历每条匹配区间
  for (const auto &m : matches) {
    std::vector<cv::Point2f> lp, rp;
    // 左右同一区间内点对
    for (const auto &iv : m.intervals) {
      for (float y = iv.y_start; y <= iv.y_end; y += precision_) {
        const LaserPoint *ptL = laser_l[m.l_idx].findPoint(y);
        const LaserPoint *ptR = laser_r[m.r_idx].findPoint(y);

        // 判断是否在精度范围内
        if (ptL && ptR) {
          lp.emplace_back(ptL->x, y);
          rp.emplace_back(ptR->x, y);
        }
      }
    }
    if (lp.empty())
      continue;
    cv::Mat pts4;
    cv::triangulatePoints(calib_info.P[0], calib_info.P[1], lp, rp, pts4);
    for (int i = 0; i < pts4.cols; ++i) {
      float w = pts4.at<float>(3, i);
      if (fabs(w) < 1e-6)
        continue;
      cloud.emplace_back(pts4.at<float>(0, i) / w, pts4.at<float>(1, i) / w,
                         pts4.at<float>(2, i) / w);
    }
  }
  return cloud;
}

/*************************************************************************************
 */
