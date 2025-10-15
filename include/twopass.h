#include "type.h"
#include <vector>
#include <random>
#include <opencv2/ximgproc.hpp>

// #define DEBUG_PassNew

cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}

void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	int num = 0;
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
					num++;
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

void Two_PassNew(const cv::Mat &img, cv::Mat &labImg)
{
    cv::Mat bwImg;
    cv::threshold(img, bwImg, 100, 255, cv::THRESH_BINARY);
	assert(bwImg.type() == CV_8UC1);
	labImg.create(bwImg.size(), CV_32SC1);   //bwImg.convertTo( labImg, CV_32SC1 );
	labImg = cv::Scalar(0);
	labImg.setTo(cv::Scalar(1), bwImg);
	assert(labImg.isContinuous());
	const int Rows = bwImg.rows - 1, Cols = bwImg.cols - 1;
	int label = 1;
	std::vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);
	//the first pass
	int *data_prev = (int*)labImg.data;   //0-th row : int* data_prev = labImg.ptr<int>(i-1);
	int *data_cur = (int*)(labImg.data + labImg.step); //1-st row : int* data_cur = labImg.ptr<int>(i);
	for (int i = 1; i < Rows; i++)
	{
		data_cur++;
		data_prev++;
		for (int j = 1; j<Cols; j++, data_cur++, data_prev++)
		{
			if (*data_cur != 1)
				continue;
			int left = *(data_cur - 1);
			int up = *data_prev;
			int neighborLabels[2];
			int cnt = 0;
			if (left>1)
				neighborLabels[cnt++] = left;
			if (up > 1)
				neighborLabels[cnt++] = up;
			if (!cnt)
			{
				labelSet.push_back(++label);
				labelSet[label] = label;
				*data_cur = label;
				continue;
			}
			int smallestLabel = neighborLabels[0];
			if (cnt == 2 && neighborLabels[1]<smallestLabel)
				smallestLabel = neighborLabels[1];
			*data_cur = smallestLabel;
			// 保存最小等价表
			for (int k = 0; k<cnt; k++)
			{
				int tempLabel = neighborLabels[k];
				int& oldSmallestLabel = labelSet[tempLabel];  //这里的&不是取地址符号,而是引用符号
				if (oldSmallestLabel > smallestLabel)
				{
					labelSet[oldSmallestLabel] = smallestLabel;
					oldSmallestLabel = smallestLabel;
				}
				else if (oldSmallestLabel<smallestLabel)
					labelSet[smallestLabel] = oldSmallestLabel;
			}
		}
		data_cur++;
		data_prev++;
	}
	//更新等价队列表,将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int prelabel = labelSet[curLabel];
		while (prelabel != curLabel)
		{
			curLabel = prelabel;
			prelabel = labelSet[prelabel];
		}
		labelSet[i] = curLabel;
	}
	//second pass
	data_cur = (int*)labImg.data;
	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < bwImg.cols - 1; j++, data_cur++)
			*data_cur = labelSet[*data_cur];
		data_cur++;
	}
}

// 中值滤波处理
std::map<int, float> median_prefilter(const std::map<int, int>& edge_map, int win = 5) {
    std::vector<int> ys;
    std::vector<float> xs;
    for (const auto& p : edge_map) {
        ys.push_back(p.first);
        xs.push_back(static_cast<float>(p.second));
    }
    
    std::map<int, float> output;
    if (ys.empty()) return output;
    
    int half = win / 2;
    for (size_t i = 0; i < ys.size(); i++) {
        int start = std::max(0, static_cast<int>(i) - half);
        int end = std::min(static_cast<int>(xs.size()), static_cast<int>(i) + half + 1);
        std::vector<float> buf(xs.begin() + start, xs.begin() + end);
        std::sort(buf.begin(), buf.end());
        float median = buf[buf.size() / 2];
        output[ys[i]] = median;
    }
    return output;
}

// 卡尔曼滤波实现
void kalman_filter_edge(
    const std::map<int, int>& edge_map, 
    std::vector<int>& ys_out, 
    std::vector<float>& xs_out,
    float Q_scale = 0.005f, 
    float R_val = 25.0f,
    float gate = 2.0f,
    int post_win = 5)
{
    ys_out.clear();
    xs_out.clear();
    if (edge_map.empty()) return;
    
    // 1. 中值预滤波
    std::map<int, float> pref = median_prefilter(edge_map);
    std::vector<int> ys;
    std::vector<float> xs;
    for (const auto& p : pref) {
        ys.push_back(p.first);
        xs.push_back(p.second);
    }
    
    // 2. 初始状态估计
    cv::Mat x0 = cv::Mat::zeros(3, 1, CV_32F);
    x0.at<float>(0) = xs[0]; // 初始位置
    
    // 初始化速度
    if (ys.size() > 1) {
        std::vector<float> velocities;
        for (size_t i = 1; i < std::min(static_cast<size_t>(3), ys.size()); i++) {
            float dy = static_cast<float>(ys[i] - ys[i-1]);
            dy = std::max(dy, 1e-3f);
            velocities.push_back((xs[i] - xs[i-1]) / dy);
        }
        if (!velocities.empty()) {
            x0.at<float>(1) = std::accumulate(velocities.begin(), velocities.end(), 0.0f) / velocities.size();
        }
    }
    
    // 3. 噪声参数设置
    cv::Mat Q_base = (cv::Mat_<float>(3,3) << 0.001f, 0, 0, 0, 0.01f, 0, 0, 0, 0.1f);
    cv::Mat Q = Q_base * Q_scale;
    cv::Mat R = (cv::Mat_<float>(1,1) << R_val);
    cv::Mat I3 = cv::Mat::eye(3, 3, CV_32F);
    
    // 4. 前向滤波
    std::vector<cv::Mat> Xf, Pf, Xp, Pp;
    cv::Mat x = x0.clone();
    cv::Mat P = I3.clone();
    int prev_y = ys[0];
    
    for (size_t i = 0; i < ys.size(); i++) {
        int y = ys[i];
        float dy = std::min(static_cast<float>(y - prev_y), 5.0f);
        
        // 状态转移矩阵
        cv::Mat F = (cv::Mat_<float>(3,3) << 
                     1, dy, 0.5f*dy*dy,
                     0, 1, dy,
                     0, 0, 1);
        
        // 预测步骤
        cv::Mat xp = F * x;
        cv::Mat Pp_pred = F * P * F.t() + Q;
        
        // 更新步骤
        cv::Mat H = (cv::Mat_<float>(1,3) << 1, 0, 0);
        cv::Mat z = (cv::Mat_<float>(1,1) << xs[i]);
        cv::Mat innov = z - H * xp;
        cv::Mat S = H * Pp_pred * H.t() + R;
        float sigma = std::sqrt(S.at<float>(0,0));
        
        if (std::abs(innov.at<float>(0,0)) > gate * sigma) {
            // 异常值处理
            x = xp.clone();
            P = Pp_pred + Q;
        } else {
            // 卡尔曼增益
            cv::Mat K = Pp_pred * H.t() * S.inv();
            
            // 状态更新
            x = xp + K * innov;
            cv::Mat I_KH = I3 - K * H;
            P = I_KH * Pp_pred * I_KH.t() + K * R * K.t();
        }
        
        // 保存状态
        Xf.push_back(x.clone());
        Pf.push_back(P.clone());
        Xp.push_back(xp.clone());
        Pp.push_back(Pp_pred.clone());
        
        prev_y = y;
    }
    
    // 5. 后向平滑
    std::vector<cv::Mat> Xs(ys.size()), Ps(ys.size());
    Xs.back() = Xf.back().clone();
    Ps.back() = Pf.back().clone();
    
    for (int i = static_cast<int>(ys.size())-2; i >= 0; i--) {
        float dy = std::min(static_cast<float>(ys[i+1] - ys[i]), 5.0f);
        cv::Mat F = (cv::Mat_<float>(3,3) << 
                     1, dy, 0.5f*dy*dy,
                     0, 1, dy,
                     0, 0, 1);
        
        cv::Mat inv_Pp;
        cv::invert(Pp[i+1], inv_Pp, cv::DECOMP_SVD);
        cv::Mat A = Pf[i] * F.t() * inv_Pp;
        
        Xs[i] = Xf[i] + A * (Xs[i+1] - Xp[i+1]);
        Ps[i] = Pf[i] + A * (Ps[i+1] - Pp[i+1]) * A.t();
    }
    
    // 6. 后处理（中值滤波）
    std::vector<float> smooth_temp;
    for (size_t i = 0; i < Xs.size(); i++) {
        smooth_temp.push_back(Xs[i].at<float>(0));
    }
    
    int half_win = post_win / 2;
    std::vector<float> smooth_final;
    for (size_t i = 0; i < smooth_temp.size(); i++) {
        int start = std::max(0, static_cast<int>(i) - half_win);
        int end = std::min(static_cast<int>(smooth_temp.size()), static_cast<int>(i) + half_win + 1);
        
        std::vector<float> win_buf(smooth_temp.begin() + start, smooth_temp.begin() + end);
        std::sort(win_buf.begin(), win_buf.end());
        smooth_final.push_back(win_buf[win_buf.size() / 2]);
    }
    
    // 输出结果
    ys_out = ys;
    xs_out = smooth_final;
}

void Two_PassNew3(
	cv::Mat &bin_img,
	std::vector<std::vector<std::pair<cv::Point, cv::Point>>>& final_contours,
	int img_idx)
{
	// 断开虚假连接
	cv::erode(bin_img, bin_img, cv::Mat());
	cv::dilate(bin_img, bin_img, cv::Mat());

    cv::Mat label, stats, centroids;
    int n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);
	// 面积与长宽比过滤
	for (int i = 1; i < n_label; ++i) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
        if (area > 200 && aspect > 1.0) {
            bin_img.setTo(255, label == i);
        } else {
            bin_img.setTo(0, label == i);
            label.setTo(0, label == i);
        }
    }

	// 边缘平滑度筛选
	auto computeSmoothness = [](const std::map<int, int>& edge_map) -> float {
		if (edge_map.size() < 2) return 0.0f; // 单点或无点视为完全平滑

		float total_diff = 0.0f;
		int valid_pairs = 0;
		int prev_y = edge_map.begin()->first;
		int prev_x = edge_map.begin()->second;

		// 使用迭代器遍历，确保有序处理
		for (auto it = std::next(edge_map.begin()); it != edge_map.end(); ++it) {
			int curr_y = it->first;
			int curr_x = it->second;

			// 只考虑连续y坐标的点（跳过y不连续的点）
			if (curr_y == prev_y + 1) {
				float dx = static_cast<float>(std::abs(curr_x - prev_x));
				total_diff += dx;
				valid_pairs++;
			}
			prev_y = curr_y;
			prev_x = curr_x;
		}

		if (valid_pairs == 0) return 0.0f; // 没有有效点对

		// 平滑度评分：平均x变化量
		return total_diff / valid_pairs;
	};

#ifdef DEBUG_PassNew
	cv::Mat contours_vis = bin_img.clone();
	cv::cvtColor(contours_vis, contours_vis, cv::COLOR_GRAY2BGR);
	std::mt19937 rng(12345); // 固定种子保证可复现
	std::uniform_int_distribution<int> color_dist(0, 255);
#endif

	// 边缘过滤
	for (int i = 1; i < n_label; ++i) {
		if (cv::countNonZero(label == i) == 0) continue;

		cv::Mat mask = (label == i); // 只保留当前连通区域
		cv::Mat region;
		bin_img.copyTo(region, mask); // 提取该区域的二值图

		// 提取左右边缘
		std::map<int, int> edge_maxX_map;
		std::map<int, int> edge_minX_map;
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(region, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for (const auto& pt : contours[0]) {
			// 找X最小值
			if (edge_minX_map.find(pt.y) == edge_minX_map.end())
				edge_minX_map[pt.y] = pt.x;
			else if (pt.x < edge_minX_map[pt.y])
				edge_minX_map[pt.y] = pt.x;

			// 找X最大值
			if (edge_maxX_map.find(pt.y) == edge_maxX_map.end())
				edge_maxX_map[pt.y] = pt.x;
			else if (pt.x > edge_maxX_map[pt.y])
				edge_maxX_map[pt.y] = pt.x;
		}

#ifdef DEBUG_PassNew
		// 显示对应边缘
		cv::Vec3b color(color_dist(rng), color_dist(rng), color_dist(rng));
		for (const auto& pt : contours[0]) {
			contours_vis.at<cv::Vec3b>(pt.y, pt.x) = color; // 绿色
		}
#endif

		// 边缘平滑度过滤
		float left_smooth = computeSmoothness(edge_minX_map);
		float right_smooth = computeSmoothness(edge_maxX_map);
		float std_thresh = 1.5f;
		if (left_smooth > std_thresh || right_smooth > std_thresh) continue;
		// printf("left_smooth: %.3f / right_smooth: %.3f\n", left_smooth, right_smooth);

		// 准备有序的 y、x_min、x_max 列表
		std::vector<int> Ys;
		std::vector<int> Xmins, Xmaxs;
		for (auto& [y, x_min] : edge_minX_map) {
			Ys.push_back(y);
			Xmins.push_back(x_min);
			Xmaxs.push_back(edge_maxX_map[y]);
		}

		const int jump_thresh = 5;
		std::vector<std::pair<int,int>> bad_ranges;  // 存放所有 [startY, endY]

		// 2. 扫描一次，检测多段跳变
		int n_edge_p = Ys.size();
		int j = 1;
		while (j < n_edge_p) {
			int prev_w = Xmaxs[j-1] - Xmins[j-1];
			int cur_w  = Xmaxs[j]   - Xmins[j];
			// 2.1 突变开始
			if (cur_w - prev_w >= jump_thresh) {
				int bad_y = Ys[j];
				int bad_x = Xmins[j];  // 左边缘突变点的 x
				
				// 2.2 在右边缘寻找 bad_x，确定结束位置
				int end_j = j+1;
				for (; end_j < n_edge_p; ++end_j) {
					if (Xmaxs[end_j] == bad_x) {
						break;
					}
				}

				int end_y = Ys.back();  // 默认到结尾
				if (end_j < n_edge_p) end_y = Ys[end_j];

				// 记录跳变区间 [bad_y, end_y]
        		bad_ranges.emplace_back(bad_y, end_y);

				// 2.3 跳变结束后，从 end_j + 2 继续检测
				if (end_j + 1 < n_edge_p) {
					prev_w = Xmaxs[end_j + 1] - Xmins[end_j + 1];  // 重置宽度基准
					j = end_j + 2;
					continue;
				} else {
					break;  // 到达末尾，退出循环
				}
			}
			// 无突变，继续扫描
			prev_w = cur_w;
			++j;
		}

		// 3. 最终遍历，滤除所有位于 bad_ranges 中的点
		std::vector<std::pair<cv::Point, cv::Point>> edge_pair;
		for (auto& [y, x_min] : edge_minX_map) {
			// 检查 y 是否在任何 bad_ranges 区间里
			bool in_bad = false;
			for (auto& [sy, ey] : bad_ranges) {
				if (y >= sy && y <= ey) {
					in_bad = true;
					break;
				}
			}
			if (in_bad) continue;

			int x_max = edge_maxX_map[y];
			float search_range = (x_max - x_min + 1);
			if (search_range < 3 || search_range > 25) continue;

			edge_pair.emplace_back(cv::Point(x_min, y), cv::Point(x_max, y));
		}

		final_contours.push_back(edge_pair);
	}

	// 只保留激光区域二值化图像
	bin_img.setTo(0); // 全部设为0
	for (const auto& edge_pair : final_contours) {
		for (const auto& p : edge_pair) {
			int y = p.first.y;
			int x_start = p.first.x - 1;
			int x_end = p.second.x + 1;
			// 边界检查
			x_start = std::max(0, x_start);
			x_end = std::min(bin_img.cols - 1, x_end);

			for (int x = x_start; x <= x_end; ++x) {
				bin_img.at<uchar>(y, x) = 255;
			}
		}
	}

#ifdef DEBUG_PassNew
	// 可视化
	static int vis_img_cnt = 0;
	cv::Mat color_label;
	LabelColor(label, color_label);
#endif

	// 再次按面积和长宽比过滤激光线
	n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);
	for (int i = 1; i < n_label; ++i) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
		if (area > 350 && aspect > 1.0) {
			bin_img.setTo(255, label == i);
		} else {
			bin_img.setTo(0, label == i);
			label.setTo(0, label == i);
		}
	}

	// 重新提取轮廓
	final_contours.clear();
	for (int i = 1; i < n_label; ++i) {
		if (cv::countNonZero(label == i) == 0) continue;

		cv::Mat mask = (label == i); // 只保留当前连通区域
		cv::Mat region;
		bin_img.copyTo(region, mask); // 提取该区域的二值图

		// 提取左右边缘
		std::map<int, int> edge_maxX_map;
		std::map<int, int> edge_minX_map;
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(region, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		for (const auto& pt : contours[0]) {
			// 找X最小值
			if (edge_minX_map.find(pt.y) == edge_minX_map.end())
				edge_minX_map[pt.y] = pt.x;
			else if (pt.x < edge_minX_map[pt.y])
				edge_minX_map[pt.y] = pt.x;

			// 找X最大值
			if (edge_maxX_map.find(pt.y) == edge_maxX_map.end())
				edge_maxX_map[pt.y] = pt.x;
			else if (pt.x > edge_maxX_map[pt.y])
				edge_maxX_map[pt.y] = pt.x;
		}

        // 应用卡尔曼滤波
        std::vector<int> left_ys, right_ys;
        std::vector<float> left_smooth, right_smooth;
        
        kalman_filter_edge(edge_minX_map, left_ys, left_smooth);
        kalman_filter_edge(edge_maxX_map, right_ys, right_smooth);
        
        // 构建点对
        std::vector<std::pair<cv::Point, cv::Point>> edge_pair;
        std::map<int, float> left_map, right_map;
        
        for (size_t j = 0; j < left_ys.size(); j++) {
            left_map[left_ys[j]] = left_smooth[j];
        }
        for (size_t j = 0; j < right_ys.size(); j++) {
            right_map[right_ys[j]] = right_smooth[j];
        }
        
        for (int y : left_ys) {
            if (right_map.find(y) != right_map.end()) {
                int x_min = static_cast<int>(std::round(left_map[y]));
                int x_max = static_cast<int>(std::round(right_map[y]));
                
                // 有效宽度检查
                if (x_max > x_min && ((x_max - x_min) < 3 || (x_max - x_min) > 25)) continue;
                edge_pair.emplace_back(cv::Point(x_min, y), cv::Point(x_max, y));
            }
        }
        
        if (!edge_pair.empty()) final_contours.push_back(edge_pair);
	}

	// 只保留激光区域二值化图像
	bin_img.setTo(0); // 全部设为0
	for (const auto& edge_pair : final_contours) {
		for (const auto& p : edge_pair) {
			int y = p.first.y;
			int x_start = p.first.x;
			int x_end = p.second.x;
			// 边界检查
			x_start = std::max(0, x_start);
			x_end = std::min(bin_img.cols - 1, x_end);

			for (int x = x_start; x <= x_end; ++x) {
				bin_img.at<uchar>(y, x) = 255;
			}
		}
	}

	// 再次按面积和长宽比过滤激光线
	n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);
	for (int i = 1; i < n_label; ++i) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
		if (area > 350 && aspect > 1.0) {
			bin_img.setTo(255, label == i);
		} else {
			bin_img.setTo(0, label == i);
			label.setTo(0, label == i);
		}
	}

#ifdef DEBUG_PassNew
	// 可视化
	cv::Mat direct_vis;
	cv::cvtColor(bin_img, direct_vis, cv::COLOR_GRAY2BGR);
	for (const auto& edge_pair : final_contours) {
		for (const auto& p : edge_pair) {
			direct_vis.at<cv::Vec3b>(cv::Point(p.first.x, p.first.y)) = cv::Vec3b(0, 0, 255); // 红色表示上边沿
			direct_vis.at<cv::Vec3b>(cv::Point(p.second.x, p.second.y)) = cv::Vec3b(255, 0, 0); // 蓝色表示下边沿
		}
	}
    if (vis_img_cnt % 2 == 0) {
		cv::imwrite(debug_img_dir / ("labelImg_l" + std::to_string(img_idx) + ".jpg"), color_label);
		cv::imwrite(debug_img_dir / ("direct_l_" + std::to_string(img_idx) + ".jpg"), direct_vis);
	}
    else {
		cv::imwrite(debug_img_dir / ("labelImg_r" + std::to_string(img_idx) + ".jpg"), color_label);
		cv::imwrite(debug_img_dir / ("direct_r_" + std::to_string(img_idx) + ".jpg"), direct_vis);
	}
    vis_img_cnt++;
#endif

}

void Two_PassNew4(
    cv::Mat &bin_img,
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>>& final_contours,
    int img_idx)
{
    // 1. 预处理：断开虚假连接
    static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin_img, bin_img, cv::MORPH_OPEN, kernel);

    // 2. 单次连通区域分析
    cv::Mat label, stats, centroids;
    int n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);

    // 预分配内存
    std::vector<bool> valid_regions(n_label, false);
    std::vector<std::vector<std::pair<cv::Point, cv::Point>>> temp_contours(n_label);

    // 3. 并行处理每个区域
    #pragma omp parallel for
    for (int i = 1; i < n_label; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
        
        // 面积与长宽比过滤
        if (area <= 200 || aspect <= 1.0f) {
            continue;
        }
        
        // 获取区域边界框
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        
        // 提取左右边缘
        std::vector<int> min_x_vec(height, std::numeric_limits<int>::max());
        std::vector<int> max_x_vec(height, std::numeric_limits<int>::min());
        
        // 直接扫描区域，避免使用findContours
        for (int row = y; row < y + height; ++row) {
            const int* label_ptr = label.ptr<int>(row);
            const uchar* bin_ptr = bin_img.ptr<uchar>(row);
            
            for (int col = x; col < x + width; ++col) {
                if (label_ptr[col] == i && bin_ptr[col] == 255) {
                    int rel_row = row - y;
                    if (col < min_x_vec[rel_row]) min_x_vec[rel_row] = col;
                    if (col > max_x_vec[rel_row]) max_x_vec[rel_row] = col;
                }
            }
        }
        
        // 构建边缘映射
        std::map<int, int> edge_minX_map, edge_maxX_map;
        for (int rel_row = 0; rel_row < height; ++rel_row) {
            if (min_x_vec[rel_row] != std::numeric_limits<int>::max()) {
                int abs_row = y + rel_row;
                edge_minX_map[abs_row] = min_x_vec[rel_row];
                edge_maxX_map[abs_row] = max_x_vec[rel_row];
            }
        }
        
        // 边缘平滑度计算
        auto computeSmoothness = [](const std::map<int, int>& edge_map) -> float {
            if (edge_map.size() < 2) return 0.0f;
            
            float total_diff = 0.0f;
            int valid_pairs = 0;
            auto it = edge_map.begin();
            int prev_y = it->first;
            int prev_x = it->second;
            ++it;
            
            for (; it != edge_map.end(); ++it) {
                int curr_y = it->first;
                int curr_x = it->second;
                
                if (curr_y == prev_y + 1) {
                    float dx = static_cast<float>(std::abs(curr_x - prev_x));
                    total_diff += dx;
                    valid_pairs++;
                }
                prev_y = curr_y;
                prev_x = curr_x;
            }
            
            return valid_pairs > 0 ? total_diff / valid_pairs : 0.0f;
        };
        
        // 边缘平滑度过滤
        float left_smooth = computeSmoothness(edge_minX_map);
        float right_smooth = computeSmoothness(edge_maxX_map);
        float std_thresh = 1.5f;
        
        if (left_smooth > std_thresh || right_smooth > std_thresh) {
            continue;
        }
        
        // 准备有序的 y、x_min、x_max 列表
        std::vector<int> Ys;
        std::vector<int> Xmins, Xmaxs;
        for (auto& [y_val, x_min] : edge_minX_map) {
            Ys.push_back(y_val);
            Xmins.push_back(x_min);
            Xmaxs.push_back(edge_maxX_map[y_val]);
        }
        
        // 检测跳变
        const int jump_thresh = 5;
        std::vector<std::pair<int, int>> bad_ranges;
        
        int n_edge_p = Ys.size();
        if (n_edge_p > 1) {
            int j = 1;
            int prev_w = Xmaxs[0] - Xmins[0];
            
            while (j < n_edge_p) {
                int cur_w = Xmaxs[j] - Xmins[j];
                
                if (cur_w - prev_w >= jump_thresh) {
                    int bad_y = Ys[j];
                    int bad_x = Xmins[j];
                    
                    int end_j = j + 1;
                    for (; end_j < n_edge_p; ++end_j) {
                        if (Xmaxs[end_j] == bad_x) {
                            break;
                        }
                    }
                    
                    int end_y = (end_j < n_edge_p) ? Ys[end_j] : Ys.back();
                    bad_ranges.emplace_back(bad_y, end_y);
                    
                    if (end_j + 1 < n_edge_p) {
                        prev_w = Xmaxs[end_j + 1] - Xmins[end_j + 1];
                        j = end_j + 2;
                    } else {
                        break;
                    }
                } else {
                    prev_w = cur_w;
                    ++j;
                }
            }
        }
        
        // 过滤跳变区间内的极
        std::vector<std::pair<cv::Point, cv::Point>> edge_pair;
        for (size_t idx = 0; idx < Ys.size(); ++idx) {
            int y = Ys[idx];
            int x_min = Xmins[idx];
            int x_max = Xmaxs[idx];
            
            // 检查是否在跳变区间
            bool in_bad = false;
            for (auto& [sy, ey] : bad_ranges) {
                if (y >= sy && y <= ey) {
                    in_bad = true;
                    break;
                }
            }
            
            if (in_bad) continue;
            
            float search_range = (x_max - x_min + 1);
            if (search_range < 3 || search_range > 25) continue;
            
            edge_pair.emplace_back(cv::Point(x_min, y), cv::Point(x_max, y));
        }
        
        if (!edge_pair.empty()) {
            #pragma omp critical
            {
                temp_contours[i] = edge_pair;
                valid_regions[i] = true;
            }
        }

        // 应用卡尔曼滤波
        // if (!edge_pair.empty()) {
        //     std::vector<int> left_ys, right_ys;
        //     std::vector<float> left_smooth, right_smooth;
            
        //     // 提取左右边缘极
        //     std::map<int, int> left_map, right_map;
        //     for (const auto& p : edge_pair) {
        //         left_map[p.first.y] = p.first.x;
        //         right_map[p.first.y] = p.second.x;
        //     }
            
        //     // 应用卡尔曼滤波
        //     kalman_filter_edge(left_map, left_ys, left_smooth);
        //     kalman_filter_edge(right_map, right_ys, right_smooth);
            
        //     // 构建滤波后的极对
        //     std::vector<std::pair<cv::Point, cv::Point>> filtered_edge_pair;
        //     std::map<int, float> left_filtered, right_filtered;
            
        //     for (size_t j = 0; j < left_ys.size(); j++) {
        //         left_filtered[left_ys[j]] = left_smooth[j];
        //     }
        //     for (size_t j = 0; j < right_ys.size(); j++) {
        //         right_filtered[right_ys[j]] = right_smooth[j];
        //     }
            
        //     for (int y : left_ys) {
        //         if (right_filtered.find(y) != right_filtered.end()) {
        //             int x_min = static_cast<int>(std::round(left_filtered[y]));
        //             int x_max = static_cast<int>(std::round(right_filtered[y]));
                    
        //             if (x_max > x_min && (x_max - x_min) >= 3 && (x_max - x_min) <= 25) {
        //                 filtered_edge_pair.emplace_back(cv::Point(x_min, y), cv::Point(x_max, y));
        //             }
        //         }
        //     }
            
        //     if (!filtered_edge_pair.empty()) {
        //         #pragma omp critical
        //         {
        //             temp_contours[i] = filtered_edge_pair;
        //             valid_regions[i] = true;
        //         }
        //     }
        // }
    }
    
    // 4. 收集所有有效区域
    final_contours.clear();
    for (int i = 1; i < n_label; ++i) {
        if (valid_regions[i] && !temp_contours[i].empty()) {
            final_contours.push_back(temp_contours[i]);
        }
    }

    // 5. 生成最终二值图像
    bin_img.setTo(0);
    for (const auto& edge_pair : final_contours) {
        for (const auto& p : edge_pair) {
            int y = p.first.y;
            int x_start = std::max(0, p.first.x - 1);
            int x_end = std::min(bin_img.cols - 1, p.second.x + 1);
            
            uchar* row_ptr = bin_img.ptr<uchar>(y);
            for (int x = x_start; x <= x_end; ++x) {
                row_ptr[x] = 255;
            }
        }
    }
    
    // 6. 最终过滤（基于面积和长宽比）
    n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);
    bin_img.setTo(0);
    for (int i = 1; i < n_label; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
        
        if (area > 350 && aspect > 1.0) {
            cv::Mat region_mask = (label == i);
            bin_img.setTo(255, region_mask);
        }
    }

    // 7. 根据最终的bin_img再次提取final_contours
    final_contours.clear();
    n_label = cv::connectedComponentsWithStats(bin_img, label, stats, centroids);
    for (int i = 1; i < n_label; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);
        
        // 最终过滤条件
        if (area > 350 && aspect > 1.0) {
            // 获取区域边界框
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            
            // 提取左右边缘
            std::vector<int> min_x_vec(height, std::numeric_limits<int>::max());
            std::vector<int> max_x_vec(height, std::numeric_limits<int>::min());
            
            for (int row = y; row < y + height; ++row) {
                const int* label_ptr = label.ptr<int>(row);
                const uchar* bin_ptr = bin_img.ptr<uchar>(row);
                
                for (int col = x; col < x + width; ++col) {
                    if (label_ptr[col] == i && bin_ptr[col] == 255) {
                        int rel_row = row - y;
                        if (col < min_x_vec[rel_row]) min_x_vec[rel_row] = col;
                        if (col > max_x_vec[rel_row]) max_x_vec[rel_row] = col;
                    }
                }
            }
            
            // 构建边缘映射
            std::map<int, int> edge_minX_map, edge_maxX_map;
            for (int rel_row = 0; rel_row < height; ++rel_row) {
                if (min_x_vec[rel_row] != std::numeric_limits<int>::max()) {
                    int abs_row = y + rel_row;
                    edge_minX_map[abs_row] = min_x_vec[rel_row];
                    edge_maxX_map[abs_row] = max_x_vec[rel_row];
                }
            }
            
            // 准备有序的 y、x_min、x_max 列表
            std::vector<int> Ys;
            std::vector<int> Xmins, Xmaxs;
            for (auto& [y_val, x_min] : edge_minX_map) {
                Ys.push_back(y_val);
                Xmins.push_back(x_min);
                Xmaxs.push_back(edge_maxX_map[y_val]);
            }
            
            // 构建边缘点对
            std::vector<std::pair<cv::Point, cv::Point>> edge_pair;
            for (size_t idx = 0; idx < Ys.size(); ++idx) {
                int y = Ys[idx];
                int x_min = Xmins[idx] - 5;
                int x_max = Xmaxs[idx];
                
                float search_range = (x_max - x_min + 1);
                if (search_range < 3 || search_range > 30) continue;
                
                edge_pair.emplace_back(cv::Point(x_min, y), cv::Point(x_max, y));
            }
            
            if (!edge_pair.empty()) {
                final_contours.push_back(edge_pair);
            }
        }
    }

#ifdef DEBUG_PassNew
	// 可视化
	static int vis_img_cnt = 0;
	cv::Mat color_label;
	LabelColor(label, color_label);
#endif

#ifdef DEBUG_PassNew
	// 可视化
	cv::Mat direct_vis;
	cv::cvtColor(bin_img, direct_vis, cv::COLOR_GRAY2BGR);
	for (const auto& edge_pair : final_contours) {
		for (const auto& p : edge_pair) {
			direct_vis.at<cv::Vec3b>(cv::Point(p.first.x, p.first.y)) = cv::Vec3b(0, 0, 255); // 红色表示上边沿
			direct_vis.at<cv::Vec3b>(cv::Point(p.second.x, p.second.y)) = cv::Vec3b(255, 0, 0); // 蓝色表示下边沿
		}
	}
    if (vis_img_cnt % 2 == 0) {
		cv::imwrite(debug_img_dir / ("labelImg_l" + std::to_string(img_idx) + ".jpg"), color_label);
		cv::imwrite(debug_img_dir / ("direct_l_" + std::to_string(img_idx) + ".jpg"), direct_vis);
	}
    else {
		cv::imwrite(debug_img_dir / ("labelImg_r" + std::to_string(img_idx) + ".jpg"), color_label);
		cv::imwrite(debug_img_dir / ("direct_r_" + std::to_string(img_idx) + ".jpg"), direct_vis);
	}
    vis_img_cnt++;
#endif
}

std::vector<cv::RotatedRect> DetectLaserRegions(cv::Mat& labImg) {
    std::vector<cv::RotatedRect> laserRects;
    if (labImg.empty() || labImg.type() != CV_32SC1)
        return laserRects;

    // 特征过滤阈值（根据实际场景调整）
    const int MIN_AREA = 50;          // 最小像素面积
    const float MIN_ASPECT = 4.0f;    // 最小长宽比
    // const float MAX_WIDTH = 45.0f;    // 最大宽度
	const float WIDTH_EXPAND = 4.5f;   // 宽度扩展像素数
	const float MIN_LENGTH = 50.0f;

    // 存储各标签区域坐标
    std::map<int, std::vector<cv::Point>> regionPoints;
    
    // 收集所有连通区域点集
    for (int y = 0; y < labImg.rows; ++y) {
        const int* row = labImg.ptr<int>(y);
        for (int x = 0; x < labImg.cols; ++x) {
            int label = row[x];
            if (label > 1) // 忽略背景(0)和未处理区域(1)
                regionPoints[label].emplace_back(x, y);
        }
    }

    // 处理各连通区域
    for (auto& [label, points] : regionPoints) {
        if (points.size() < MIN_AREA) continue;
        
        // 计算最小区域矩形
        cv::RotatedRect rr = cv::minAreaRect(points);
        cv::Size2f size = rr.size;
        float width = std::min(size.width, size.height);
        float height = std::max(size.width, size.height);
        float aspect = height / width;

        // 激光线判断条件
        if (aspect > MIN_ASPECT && 
            // width < MAX_WIDTH &&
			height > MIN_LENGTH) {

			// 扩展旋转矩形尺寸
            cv::Size2f newSize(
                (width < height) ? (size.width + WIDTH_EXPAND) : size.height,
                (width < height) ? size.height : (size.height + WIDTH_EXPAND)
            );
            rr.size = newSize;

            laserRects.push_back(rr);
        }
    }

    return laserRects;
}

