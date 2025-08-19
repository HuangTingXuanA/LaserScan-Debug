import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def compute_smoothness(edge_map):
    """计算边缘平滑度（x变化量平均值）"""
    if len(edge_map) < 2:
        return 1000.0  # 单点或无点视为不平滑
    
    sorted_y = sorted(edge_map.keys())
    total_diff = 0.0
    valid_pairs = 0
    
    prev_y = sorted_y[0]
    prev_x = edge_map[prev_y]
    
    for i in range(1, len(sorted_y)):
        curr_y = sorted_y[i]
        curr_x = edge_map[curr_y]
        
        if curr_y == prev_y + 1:  # 连续y坐标
            dx = abs(curr_x - prev_x)
            total_diff += dx
            valid_pairs += 1
        
        prev_y = curr_y
        prev_x = curr_x
    
    return total_diff / valid_pairs if valid_pairs > 0 else 1000.0

def median_prefilter(edge_map, win=5):
    """中值预滤波器"""
    ys = sorted(edge_map.keys())
    xs = [edge_map[y] for y in ys]
    out = {}
    half = win // 2
    for i, y in enumerate(ys):
        buf = xs[max(0,i-half):min(len(xs),i+half+1)]
        m = np.median(buf)
        out[y] = int(round(m))
    return out

def smooth_edge(edge_map, jump_thresh=3.0, max_dy=3, min_seg_len=10):
    """卡尔曼滤波平滑边缘"""
    ys_all = sorted(edge_map.keys())
    # Segment by large gaps
    segments = []
    seg = [ys_all[0]]
    for y in ys_all[1:]:
        if y - seg[-1] > max_dy:
            segments.append(seg)
            seg = []
        seg.append(y)
    segments.append(seg)
    
    full_ys, full_meas, full_pred, full_smooth = [], [], [], []
    
    for seg in segments:
        if len(seg) < min_seg_len:
            pref = median_prefilter({y: edge_map[y] for y in seg}, win=5)
            for y in seg:
                full_ys.append(y)
                full_meas.append(edge_map[y])
                full_pred.append(pref[y])
                full_smooth.append(pref[y])
            continue
        
        # Prefilter each segment
        pref_map = median_prefilter({y: edge_map[y] for y in seg}, win=5)
        # Initialize KF
        x = np.array([pref_map[seg[0]], 0.0, 0.0], dtype=np.float32)
        P = np.eye(3, dtype=np.float32)
        Q = np.diag([1e-3, 1e-2, 1e-1]).astype(np.float32)
        R = np.array([[1e-2]], dtype=np.float32)
        prev_y = None
        
        for y in seg:
            z = np.array([pref_map[y]], dtype=np.float32)
            if prev_y is None:
                meas = z[0]
                pred = z[0]
                smooth = z[0]
                P = np.eye(3, dtype=np.float32)
            else:
                dy = min(y - prev_y, max_dy)
                # State transition
                F = np.array([[1, dy, 0.5*dy*dy],
                              [0, 1, dy],
                              [0, 0, 1]], dtype=np.float32)
                H = np.array([[1,0,0]], dtype=np.float32)
                # Predict
                x_pred = F.dot(x)
                P_pred = F.dot(P).dot(F.T) + Q
                meas = z[0]
                pred = x_pred[0]
                # Update
                innov = z - H.dot(x_pred)
                S = H.dot(P_pred).dot(H.T) + R
                K = P_pred.dot(H.T).dot(np.linalg.inv(S))
                if abs(innov[0]) > jump_thresh:
                    x = x_pred
                else:
                    x = x_pred + K.dot(innov)
                    P = (np.eye(3) - K.dot(H)).dot(P_pred)
                smooth = x[0]
            
            full_ys.append(y)
            full_meas.append(meas)
            full_pred.append(pred)
            full_smooth.append(smooth)
            prev_y = y
    
    return full_ys, full_meas, full_pred, full_smooth

def visualize_kalman(edge_min, edge_max, region_idx):
    """可视化卡尔曼滤波过程"""
    ys_min, meas_min, pred_min, smooth_min = smooth_edge(edge_min)
    ys_max, meas_max, pred_max, smooth_max = smooth_edge(edge_max)
    
    # KF跟踪结果
    plt.figure(figsize=(15, 6))
    
    # 左边缘
    plt.subplot(121)
    plt.plot(ys_min, meas_min, 'yo-', alpha=0.7, label='measurement')
    plt.plot(ys_min, pred_min, 'r--', alpha=0.7, label='prediction')
    plt.plot(ys_min, smooth_min, 'm-', linewidth=2, label='smoothed')
    plt.title(f'Region #{region_idx} - LEFT Edge KF Tracking')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 右边缘
    plt.subplot(122)
    plt.plot(ys_max, meas_max, 'co-', alpha=0.7, label='measurement')
    plt.plot(ys_max, pred_max, 'r--', alpha=0.7, label='prediction')
    plt.plot(ys_max, smooth_max, 'g-', linewidth=2, label='smoothed')
    plt.title(f'Region #{region_idx} - Right Edge KF Tracking')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 边缘位置可视化
    plt.figure(figsize=(8, 6))
    for y in edge_min.keys():
        plt.plot([edge_min[y], edge_max[y]], [y, y], 'c-', alpha=0.5)
    
    plt.plot([m for m in smooth_min], ys_min, 'mo-', markersize=4, label='smooth left')
    plt.plot([m for m in smooth_max], ys_max, 'go-', markersize=4, label='smooth right')
    plt.title(f'Region #{region_idx} - edge compare')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()  # y轴反转以匹配图像坐标
    plt.legend()
    plt.grid(True)
    plt.show()



# 1. 读取灰度图像
img_path = 'image.png'
bin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

plt.figure(figsize=(6,4))
plt.imshow(bin_img, cmap='gray')
plt.title('orign img')
plt.axis('off')
plt.show()

# 2. 形态学腐蚀膨胀（断开虚假连接）
kernel = np.ones((3, 3), np.uint8)
bin_img = cv2.erode(bin_img, kernel)
bin_img = cv2.dilate(bin_img, kernel)

# 3. 第一次连通区域分析过滤
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
filtered_img = np.zeros_like(bin_img)
edge_pairs = []  # 存储所有边缘对

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    # 面积>200且长宽比>1.0
    if area > 200 and aspect > 1.0:
        # 提取当前区域
        mask = (labels == i).astype(np.uint8)
        region = cv2.bitwise_and(bin_img, bin_img, mask=mask)
        
        # 4. 提取轮廓和左右边缘
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
            
        edge_min = defaultdict(lambda: 10000)  # y: min_x
        edge_max = defaultdict(lambda: -1)     # y: max_x
        
        for pt in contours[0].squeeze():
            x, y = pt[0], pt[1]
            edge_min[y] = min(edge_min[y], x)
            edge_max[y] = max(edge_max[y], x)
        
        # 5. 边缘平滑度过滤
        left_smooth = compute_smoothness(edge_min)
        right_smooth = compute_smoothness(edge_max)
        std_thresh = 1.5
        if (left_smooth > std_thresh or right_smooth > std_thresh):
            continue

        # 6. 异常跳变过滤
        valid_edges = []
        last_min = last_max = -1
        jump_thresh = 7
        
        bad_x_min = -1
        bad_x_max = -1
        sorted_y = sorted(edge_min.keys())
        for y in sorted_y:
            x_min = edge_min[y]
            x_max = edge_max[y]
            
            # 跳变检测
            if last_min >= 0:
                min_jump = abs(x_min - last_min) > jump_thresh or (bad_x_min == x_min)
                max_jump = abs(x_max - last_max) > jump_thresh or (bad_x_max == x_max)
                if min_jump:
                    last_min, last_max = x_min, x_max
                    bad_x_min = x_min
                    continue
                if max_jump:
                    last_min, last_max = x_min, x_max
                    bad_x_max = x_max
                    continue
                    
            last_min, last_max = x_min, x_max
            
            # 有效边缘对
            width = x_max - x_min + 1
            if 3 <= width <= 22:
                valid_edges.append((y, x_min, x_max))
        
        # 添加到最终结果
        if valid_edges:
            edge_pairs.append((i, valid_edges))

# 7. 重构二值图像
reconstructed_img = np.zeros_like(bin_img)
for region_idx, edges in edge_pairs:
    for y, x_min, x_max in edges:
        x_start = max(0, x_min - 1)
        x_end = min(reconstructed_img.shape[1] - 1, x_max + 1)
        reconstructed_img[y, x_start:x_end + 1] = 255

# 8. 第二次连通区域过滤
num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(reconstructed_img)
final_img = np.zeros_like(reconstructed_img)
final_regions = []

for i in range(1, num_labels2):
    area = stats2[i, cv2.CC_STAT_AREA]
    w = stats2[i, cv2.CC_STAT_WIDTH]
    h = stats2[i, cv2.CC_STAT_HEIGHT]
    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    # 面积>350且长宽比>1.0
    if area > 350 and aspect > 1.0:
        final_img[labels2 == i] = 255
        # 保存区域用于后续处理
        region_mask = (labels2 == i).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            final_regions.append((i, contours[0]))

# 可视化处理流程结果
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(bin_img, cmap='gray'), plt.title('orign img')
plt.subplot(132), plt.imshow(reconstructed_img, cmap='gray'), plt.title('edge filte img')
plt.subplot(133), plt.imshow(final_img, cmap='gray'), plt.title('before kalman img')
plt.tight_layout()
plt.show()

# 9. 卡尔曼滤波和可视化
for region_idx, contour in [final_regions[3], final_regions[4]]:
    # 提取边缘点
    edge_min = {}
    edge_max = {}
    for pt in contour.squeeze():
        x, y = pt[0], pt[1]
        # 更新最小x值
        if y not in edge_min or x < edge_min[y]:
            edge_min[y] = x
        # 更新最大x值
        if y not in edge_max or x > edge_max[y]:
            edge_max[y] = x
    
    # 可视化卡尔曼滤波过程
    visualize_kalman(edge_min, edge_max, region_idx)
    
    # 绘制区域轮廓
    mask = np.zeros_like(final_img)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'region #{region_idx} contours')
    plt.axis('off')
    plt.show()