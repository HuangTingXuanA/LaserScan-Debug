#include "type.h"
#include <vector>
#include <opencv2/ximgproc.hpp>

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

void Two_PassNew2(const cv::Mat &img, cv::Mat &labImg)
{
    // 2. CLAHE 局部对比度增强
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(22, 22));
    cv::Mat img_clahe;
    clahe->apply(img, img_clahe);
	/** 可视化调参
	{
		// 2. CLAHE 参数配置
		struct CLAHEParams { double clipLimit; cv::Size tileSize; };
		std::vector<CLAHEParams> params = {
			{2.0, cv::Size(20, 20)},
			{2.5, cv::Size(18, 18)},
			{1.5, cv::Size(20, 20)},
			{2.0, cv::Size(22, 22)}
		};

		// 3. 生成 CLAHE 结果
		std::vector<cv::Mat> results;
		std::vector<std::string> titles;
		for (const auto& p : params) {
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(p.clipLimit, p.tileSize);
			cv::Mat out;
			clahe->apply(img, out);
			results.push_back(out);
			titles.push_back("clip=" + std::to_string(p.clipLimit) + ", tile=" +
							std::to_string(p.tileSize.width) + "x" +
							std::to_string(p.tileSize.height));
		}

		// 4. 布局：每行最多两张，并垂直分布
		int perRow = 2;
		int margin = 10;
		int imgH = img.rows;
		int imgW = img.cols;
		int n = static_cast<int>(results.size());
		int rowsCount = static_cast<int>(std::ceil(n / static_cast<double>(perRow)));

		int canvasW = perRow * imgW + (perRow - 1) * margin;
		int canvasH = rowsCount * imgH + (rowsCount - 1) * margin;
		cv::Mat canvas(canvasH, canvasW, CV_8UC1, cv::Scalar(0));

		for (int i = 0; i < n; ++i) {
			int r = i / perRow;
			int c = i % perRow;
			int x = c * (imgW + margin);
			int y = r * (imgH + margin);
			results[i].copyTo(canvas(cv::Rect(x, y, imgW, imgH)));
			cv::putText(canvas, titles[i], cv::Point(x + 5, y + 20), cv::FONT_HERSHEY_SIMPLEX,
						0.6, cv::Scalar(255), 1);
		}

		// 5. 显示窗口，并根据屏幕大小进行缩放
		cv::namedWindow("CLAHE Comparison", cv::WINDOW_NORMAL);
		// 将窗口大小设置为画布的一半（可根据实际屏幕分辨率调整）
		cv::resizeWindow("CLAHE Comparison", canvasW / 2, canvasH / 2);
		cv::imshow("CLAHE Comparison", canvas);
		cv::waitKey(0);
	}
	*/

    // === 多尺度二值化 ===
    cv::Mat bw_fine, bw_coarse;
    int fineBlock = 11, fineC = -14;
    int coarseBlock = 35, coarseC = -17;
    cv::adaptiveThreshold(img_clahe, bw_fine, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, fineBlock, fineC);
    cv::adaptiveThreshold(img_clahe, bw_coarse, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, coarseBlock, coarseC);

    // === 连通域过滤细线图 ===
    cv::Mat fine_labels, fine_stats, fine_centroids;
    int num = cv::connectedComponentsWithStats(bw_fine, fine_labels, fine_stats, fine_centroids);
    cv::Mat clean_fine = cv::Mat::zeros(bw_fine.size(), CV_8UC1);
    for (int i = 1; i < num; ++i) {
        int area = fine_stats.at<int>(i, cv::CC_STAT_AREA);
        int w = fine_stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = fine_stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);

        if (area > 50 && aspect > 1.1) {
            clean_fine.setTo(255, fine_labels == i);
        }
    }
	cv::dilate(clean_fine, clean_fine, cv::Mat(), cv::Point(-1, -1), 1); // 膨胀操作，增强细线

    // === 粗线图用细线掩码进行排除 ===
    // 直接用 clean_fine 作为掩码
    cv::Mat inverse_mask;
    cv::bitwise_not(clean_fine, inverse_mask);  // 反向掩码：非细线区域保留
    cv::Mat filtered_coarse;
    cv::bitwise_and(bw_coarse, inverse_mask, filtered_coarse);
	cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::erode(filtered_coarse, filtered_coarse, erode_kernel, cv::Point(-1, -1), 1); // 腐蚀操作，去除小噪点

    // === 对粗线图进一步进行连通域过滤 ===
    cv::Mat coarse_labels, coarse_stats, coarse_centroids;
    int n_coarse = cv::connectedComponentsWithStats(filtered_coarse, coarse_labels, coarse_stats, coarse_centroids);
    cv::Mat clean_coarse = cv::Mat::zeros(filtered_coarse.size(), CV_8UC1);
    for (int i = 1; i < n_coarse; ++i) {
        int area = coarse_stats.at<int>(i, cv::CC_STAT_AREA);
        int w = coarse_stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = coarse_stats.at<int>(i, cv::CC_STAT_HEIGHT);
        float aspect = static_cast<float>(std::max(w, h)) / std::min(w, h);

        // 可根据粗线特征单独设定更宽容的过滤条件
        if (area > 100 && aspect > 1.5) clean_coarse.setTo(255, coarse_labels == i);
		else clean_coarse.setTo(0, coarse_labels == i);
    }

	cv::Mat bwImg;
    cv::bitwise_or(clean_fine, clean_coarse, bwImg);

    // cv::threshold(img, bwImg, 150, 255, cv::THRESH_BINARY);
	// cv::adaptiveThreshold(img, bwImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 13, -10);

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

