#include "type.h"
#include <vector>

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
    cv::threshold(img, bwImg, 150, 255, cv::THRESH_BINARY);
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
    const float MIN_ASPECT = 8.0f;    // 最小长宽比
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

