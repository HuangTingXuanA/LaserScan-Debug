import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict

# 中值预滤波
def median_prefilter(edge_map, win=5):
    ys = sorted(edge_map.keys())
    xs = [edge_map[y] for y in ys]
    out = {}
    half = win // 2
    for i, y in enumerate(ys):
        buf = xs[max(0, i-half):min(len(xs), i+half+1)]
        out[y] = int(round(np.median(buf)))
    return out

# 双向卡尔曼滤波+后处理
def improved_kalman_edge(edge_map, Q_scale=0.005, R_val=25, gate=2.0, post_win=5):
    ys = sorted(edge_map.keys())
    if not ys:
        return [], []
    pref = median_prefilter(edge_map)
    x0 = np.array([pref[ys[0]], 0., 0.], np.float32)
    if len(ys) > 1:
        vel = []
        for i in range(1, min(4, len(ys))):
            dy = ys[i] - ys[i-1]
            vel.append((pref[ys[i]] - pref[ys[i-1]]) / max(dy, 1e-3))
        x0[1] = np.mean(vel)
    Q_base = np.diag([1e-3, 1e-2, 1e-1])
    Q = (Q_base * Q_scale).astype(np.float32)
    R = np.array([[R_val]], np.float32)
    I3 = np.eye(3, dtype=np.float32)
    Xf, Pf, Xp, Pp = [], [], [], []
    x, P = x0.copy(), I3.copy()
    prev = ys[0]
    for y in ys:
        z = np.array([pref[y]], np.float32)
        dy = min(y - prev, 5.)
        F = np.array([[1, dy, 0.5*dy**2], [0,1,dy], [0,0,1]], np.float32)
        H = np.array([[1,0,0]], np.float32)
        xp = F @ x
        Pp_pred = F @ P @ F.T + Q
        innov = z - H @ xp
        S = H @ Pp_pred @ H.T + R
        sigma = np.sqrt(S[0,0])
        if abs(innov[0]) > gate * sigma:
            x = xp.copy()
            P = Pp_pred.copy() + Q
            K = np.zeros((3,1), np.float32)
        else:
            K = Pp_pred @ H.T @ np.linalg.inv(S)
            x = xp + K @ innov
            P = (I3 - K @ H) @ Pp_pred @ (I3 - K @ H).T + K @ R @ K.T
        Xf.append(x.copy()); Pf.append(P.copy()); Xp.append(xp.copy()); Pp.append(Pp_pred.copy())
        prev = y
    N = len(ys)
    Xs = [None] * N; Ps = [None] * N
    Xs[-1], Ps[-1] = Xf[-1], Pf[-1]
    for i in range(N-2, -1, -1):
        dy = min(ys[i+1] - ys[i], 5.)
        F = np.array([[1, dy, 0.5*dy**2], [0,1,dy], [0,0,1]], np.float32)
        A = Pf[i] @ F.T @ np.linalg.inv(Pp[i+1])
        Xs[i] = Xf[i] + A @ (Xs[i+1] - Xp[i+1])
        Ps[i] = Pf[i] + A @ (Ps[i+1] - Pp[i+1]) @ A.T
    smooth = [Xs[i][0] for i in range(N)]
    out = []
    w = post_win; h = w // 2
    for i in range(N):
        buf = smooth[max(0, i-h):min(N, i+h+1)]
        out.append(np.median(buf))
    return ys, out

# 交互可视化
def interactive_kalman(edge_map_left, edge_map_right, region_idx):
    init_Qs, init_R, init_gate, init_pw = 0.0001, 25, 2.0, 5
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.3)
    def update(val):
        Qs = slider_Qs.val
        Rv = slider_R.val
        G = slider_gate.val
        W = int(slider_w.val)
        ys_l, sm_l = improved_kalman_edge(edge_map_left, Qs, Rv, G, W)
        ys_r, sm_r = improved_kalman_edge(edge_map_right, Qs, Rv, G, W)
        meas_l = [edge_map_left[y] for y in ys_l]
        meas_r = [edge_map_right[y] for y in ys_r]
        for ax in axes.flatten():
            ax.clear(); ax.grid(True, ls='--', alpha=0.4)
        axes[0,0].plot(ys_l, meas_l, 'o', ms=3, label=f'Meas L')
        axes[0,0].plot(ys_l, sm_l, 'r-', lw=2, label=f'Smooth L')
        axes[0,0].set_title(f'Left Edge (Q={Qs:.2f}, R={Rv:.2f}, σgate={G:.2f}, win={W})')
        axes[0,0].legend()
        axes[0,1].plot(ys_r, meas_r, 'o', ms=3, label='Meas R')
        axes[0,1].plot(ys_r, sm_r, 'g-', lw=2, label='Smooth R')
        axes[0,1].set_title('Right Edge')
        axes[0,1].legend()
        axes[1,0].plot(sm_l, ys_l, 'r-', label='SL')
        axes[1,0].plot(sm_r, ys_r, 'g-', label='SR')
        axes[1,0].invert_yaxis(); axes[1,0].legend(); axes[1,0].set_title('Smoothed Edges')
        cy = sorted(set(ys_l) & set(ys_r))
        bw = [sm_r[ys_r.index(y)] - sm_l[ys_l.index(y)] for y in cy]
        axes[1,1].plot(cy, bw, 'b-')
        axes[1,1].axhline(np.mean(bw), ls='--', c='r')
        axes[1,1].set_title('Bandwidth')
        fig.canvas.draw_idle()
    ax_Q = plt.axes([0.1, 0.22, 0.8, 0.02]);
    slider_Qs = Slider(ax_Q, 'Q scale', 0.0001, 0.01, valinit=init_Qs, valfmt='%1.4f')
    ax_R = plt.axes([0.1, 0.18, 0.8, 0.02]);
    slider_R = Slider(ax_R, 'R', 0.1, 70, valinit=init_R, valfmt='%1.2f')
    ax_G = plt.axes([0.1, 0.14, 0.8, 0.02]);
    slider_gate = Slider(ax_G, 'gate σ', 0.01, 10, valinit=init_gate, valfmt='%1.2f')
    ax_W = plt.axes([0.1, 0.10, 0.8, 0.02]);
    slider_w = Slider(ax_W, 'post_win', 1, 11, valinit=init_pw, valfmt='%d')
    for s in [slider_Qs, slider_R, slider_gate, slider_w]:
        s.on_changed(update)
    update(None)
    plt.suptitle(f'Region {region_idx} Interactive Tune')
    plt.show()

# 可视化卡尔曼滤波过程（增加带宽分析）
def visualize_kalman(edge_map_left, edge_map_right, region_idx):
    """可视化卡尔曼滤波处理过程（四视图）"""
    # 左边缘处理
    ys_left, smooth_left = improved_kalman_edge(edge_map_left)
    meas_left = [edge_map_left[y] for y in ys_left]
    
    # 右边缘处理
    ys_right, smooth_right = improved_kalman_edge(edge_map_right)
    meas_right = [edge_map_right[y] for y in ys_right]
    
    plt.figure(figsize=(15, 12))
    
    # 左边缘可视化
    plt.subplot(221)
    plt.plot(ys_left, meas_left, 'yo-', label='Measurements')
    plt.plot(ys_left, smooth_left, 'm-', linewidth=2, label='Kalman Smooth')
    plt.title(f'Region {region_idx} - Left Edge')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 右边缘可视化
    plt.subplot(222)
    plt.plot(ys_right, meas_right, 'co-', label='Measurements')
    plt.plot(ys_right, smooth_right, 'g-', linewidth=2, label='Kalman Smooth')
    plt.title(f'Region {region_idx} - Right Edge')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 双边缘位置对比
    plt.subplot(223)
    plt.plot(smooth_left, ys_left, 'm-', linewidth=2, label='Smooth Left')
    plt.plot(smooth_right, ys_right, 'g-', linewidth=2, label='Smooth Right')
    
    # 添加测量点
    plt.scatter(meas_left, ys_left, c='yellow', s=15, alpha=0.6, label='Meas Left')
    plt.scatter(meas_right, ys_right, c='cyan', s=15, alpha=0.6, label='Meas Right')
    
    plt.title(f'Region {region_idx} - Smoothed Edges')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().invert_yaxis()  # 反转Y轴以匹配图像坐标系
    
    # 带宽分析
    plt.subplot(224)
    # 确保y坐标对齐
    common_ys = sorted(set(ys_left) & set(ys_right))
    bandwidth = []
    for y in common_ys:
        idx_left = ys_left.index(y)
        idx_right = ys_right.index(y)
        bandwidth.append(smooth_right[idx_right] - smooth_left[idx_left])
    
    plt.plot(common_ys, bandwidth, 'b-', linewidth=2, label='Bandwidth')
    
    # 计算平均带宽
    avg_bandwidth = np.mean(bandwidth) if bandwidth else 0
    plt.axhline(y=avg_bandwidth, color='r', linestyle='--', label=f'Avg: {avg_bandwidth:.2f}')
    
    plt.title(f'Region {region_idx} - Bandwidth Analysis')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Bandwidth (pixels)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主处理流程（添加二值图像重构）
def main_processing(bin_img):
    """完整处理流程（返回卡尔曼滤波前后的二值图像）"""
    # 1. 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.erode(bin_img, kernel)
    bin_img = cv2.dilate(bin_img, kernel)
    
    # 保存卡尔曼滤波前的二值图像
    before_kalman_img = bin_img.copy()
    
    # 2. 连通区域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    
    # 创建卡尔曼滤波后的二值图像
    after_kalman_img = np.zeros_like(bin_img)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # 初始过滤
        if area > 200 and aspect > 1.0:
            # 提取区域
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # 提取左右边缘
                edge_min = {}
                edge_max = {}
                for pt in contours[0].squeeze():
                    x, y = pt[0], pt[1]
                    edge_min[y] = min(edge_min.get(y, x), x)
                    edge_max[y] = max(edge_max.get(y, x), x)
                
                # 可视化处理
                interactive_kalman(edge_min, edge_max, i)
                
                # 重构卡尔曼滤波后的区域
                left_ys, left_smooth = improved_kalman_edge(edge_min)
                right_ys, right_smooth = improved_kalman_edge(edge_max)
                
                # 确保y坐标对齐
                common_ys = sorted(set(left_ys) & set(right_ys))
                
                for y in common_ys:
                    idx_left = left_ys.index(y)
                    idx_right = right_ys.index(y)
                    
                    x_min = int(round(left_smooth[idx_left]))
                    x_max = int(round(right_smooth[idx_right]))
                    
                    # 确保有效宽度
                    if x_max > x_min and 3 <= (x_max - x_min) <= 30:
                        # 在结果图像上绘制
                        x_start = max(0, x_min)
                        x_end = min(after_kalman_img.shape[1]-1, x_max)
                        after_kalman_img[y, x_start:x_end+1] = 255
    
    return before_kalman_img, after_kalman_img

# 可视化二值图像对比
def visualize_binary_comparison(before_img, after_img):
    """可视化卡尔曼滤波前后的二值图像对比"""
    plt.figure(figsize=(15, 8))
    
    # 卡尔曼滤波前
    plt.subplot(121)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before Kalman Filtering')
    plt.axis('off')
    
    # 卡尔曼滤波后
    plt.subplot(122)
    plt.imshow(after_img, cmap='gray')
    plt.title('After Kalman Filtering')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 差异可视化
    diff_img = cv2.absdiff(before_img, after_img)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(diff_img, cmap='hot')
    plt.title('Difference (Red indicates changes)')
    plt.axis('off')
    plt.colorbar()
    plt.show()

# 加载并处理图像
if __name__ == "__main__":
    img_path = 'image.png'
    bin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if bin_img is None:
        raise ValueError("Failed to load image")
    
    # 二值化处理
    _, bin_img = cv2.threshold(bin_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 显示原始图像
    plt.figure(figsize=(8, 6))
    plt.imshow(bin_img, cmap='gray')
    plt.title('Original Binary Image')
    plt.axis('off')
    plt.show()
    
    # 执行处理流程并获取卡尔曼滤波前后的图像
    before_kalman_img, after_kalman_img = main_processing(bin_img)
    
    # 可视化卡尔曼滤波前后的二值图像对比
    visualize_binary_comparison(before_kalman_img, after_kalman_img)