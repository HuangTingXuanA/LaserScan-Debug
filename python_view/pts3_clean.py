import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import signal

# ==== Load file and parse lines ====
def load_lines_from_txt(path):
    lines = []
    current_line = []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("#Line"):
                if current_line:
                    lines.append(np.array(current_line, dtype=float))
                    current_line = []
                continue
            parts = ln.split()
            if len(parts) >= 3:
                current_line.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if current_line:
            lines.append(np.array(current_line, dtype=float))
    return lines

lines_data = load_lines_from_txt("test.txt")

# ==== Core Processing Functions ====
def compute_angles(pts3):
    """计算激光线各点的角度变化"""
    N = pts3.shape[0]
    angles = np.zeros(N)
    
    for i in range(1, N-1):
        a = pts3[i-1]
        b = pts3[i] 
        c = pts3[i+1]
        
        v1 = b - a
        v2 = c - b
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-9 or n2 < 1e-9:
            angles[i] = 0.0
        else:
            cosv = np.dot(v1, v2) / (n1 * n2)
            cosv = np.clip(cosv, -1.0, 1.0)
            angles[i] = math.degrees(math.acos(cosv))
    
    return angles

def detect_corner_points(pts3, angle_thresh_deg=10.0, min_gap=5):
    """检测激光线的转折点/拐角"""
    angles = compute_angles(pts3)
    N = len(pts3)
    
    corners = []
    last_corner = -min_gap
    
    for i in range(1, N-1):
        if angles[i] > angle_thresh_deg and (i - last_corner >= min_gap):
            corners.append(i)
            last_corner = i
    
    return corners, angles

def segment_laser_line(pts3, corners, min_seg_len=10):
    """根据转折点分割激光线"""
    N = len(pts3)
    segments = []
    
    start = 0
    for corner in corners:
        end = corner
        if end - start + 1 >= min_seg_len:
            segments.append((start, end))
        start = corner + 1
    
    # 添加最后一段
    if N - start >= min_seg_len:
        segments.append((start, N-1))
    
    return segments

def savgol_filter_segment(arr, win, deg):
    """对单个数组进行Savitzky-Golay滤波"""
    if len(arr) < 3:
        return arr.copy()
    
    # 确保窗口大小为奇数且不超过数组长度
    if win % 2 == 0:
        win += 1
    win = min(win, len(arr))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3
    
    # 确保多项式阶数不超过窗口大小-1
    deg = min(deg, win - 1)
    
    try:
        return signal.savgol_filter(arr, win, deg, mode='nearest')
    except:
        return arr.copy()

def aggressive_outlier_correction(original, filtered, correction_strength=0.8):
    """强化异常值修正：针对抖动问题的激进修正"""
    # 计算残差
    residuals = np.abs(original - filtered)
    
    if len(residuals) == 0:
        return filtered.copy()
    
    # 更激进的阈值策略 - 针对小幅抖动
    median_residual = np.median(residuals)
    
    # 对于小幅抖动（0.05mm~0.35mm），使用更低的阈值
    if median_residual < 0.5:  # 小幅抖动情况
        threshold = median_residual + 0.1  # 非常低的阈值
        base_strength = 0.9  # 高修正强度
    else:
        # 正常情况
        q75 = np.percentile(residuals, 75)
        threshold = q75 * 0.8  # 降低阈值
        base_strength = correction_strength
    
    # 计算修正权重 - 对小幅抖动更敏感
    weights = np.where(residuals > threshold, base_strength, 
                      base_strength * 0.5 * (residuals / threshold))
    
    # 确保权重在合理范围内
    weights = np.clip(weights, 0.0, base_strength)
    
    # 强化修正：更倾向于滤波结果
    corrected = original * (1 - weights) + filtered * weights
    
    return corrected

def multi_pass_smoothing(arr, passes=3):
    """多遍平滑：通过多次轻度平滑达到强平滑效果"""
    result = arr.copy()
    
    for i in range(passes):
        # 每次使用不同的窗口大小
        window_size = 5 + i * 2
        if window_size >= len(arr):
            window_size = len(arr) // 2 * 2 + 1
        if window_size < 3:
            window_size = 3
        
        # 使用移动平均
        result = fast_moving_average(result, window_size)
    
    return result

def enhanced_savgol_filter(arr, win, deg, iterations=3):
    """增强的SG滤波：多次迭代以获得更平滑的结果"""
    if len(arr) < 3:
        return arr.copy()
    
    # 确保窗口大小为奇数且不超过数组长度
    if win % 2 == 0:
        win += 1
    win = min(win, len(arr))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3
    
    # 确保多项式阶数不超过窗口大小-1
    deg = min(deg, win - 1)
    
    try:
        # 多次迭代SG滤波以获得更平滑的结果
        result = arr.copy()
        for i in range(iterations):
            # 每次迭代使用稍大的窗口
            current_win = min(win + i * 2, len(arr))
            if current_win % 2 == 0:
                current_win -= 1
            if current_win >= 3:
                result = signal.savgol_filter(result, current_win, deg, mode='nearest')
        return result
    except:
        return arr.copy()

def moving_average_filter(arr, window_size):
    """移动平均滤波器"""
    if window_size >= len(arr):
        return np.full_like(arr, np.mean(arr))
    
    filtered = np.zeros_like(arr)
    half_window = window_size // 2
    
    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        filtered[i] = np.mean(arr[start:end])
    
    return filtered

# ==== New Simplified Processing Pipeline ====
def process_laser_line(pts3, 
                      angle_thresh_deg=5.0,
                      min_gap=10,
                      min_seg_len=6,
                      sg_win=10,
                      sg_deg=1,
                      correction_strength=0.7):
    """
    改进的激光线处理流程：更强的平滑效果
    1. 识别转折点/拐角
    2. 分段处理
    3. 多级滤波（移动平均 + 增强SG滤波）
    4. 异常值修正
    """
    N = pts3.shape[0]
    
    # Step 1: 检测转折点
    corners, angles = detect_corner_points(pts3, angle_thresh_deg, min_gap)
    
    # Step 2: 分割激光线
    segments = segment_laser_line(pts3, corners, min_seg_len)
    
    # 如果没有检测到有效分段，将整条线作为一段处理
    if not segments:
        segments = [(0, N-1)]
    
    # Step 3: 多级滤波处理
    processed_pts = pts3.copy()
    sg_filtered_pts = pts3.copy()
    
    for start, end in segments:
        # 提取段数据
        seg_pts = pts3[start:end+1].copy()
        seg_len = end - start + 1
        
        # 根据段长度调整窗口大小
        adaptive_win = min(sg_win, seg_len // 2 * 2 + 1)
        if adaptive_win < 3:
            adaptive_win = 3
        
        # 快速几何分析
        geometry_type = analyze_segment_geometry(seg_pts)
        
        # 强化预滤波处理 - 针对抖动问题
        if geometry_type == "straight":
            # 直线段：强预滤波
            ma_window = max(5, adaptive_win // 2)
            seg_x_ma = fast_moving_average(seg_pts[:, 0], ma_window)
            seg_y_ma = fast_moving_average(seg_pts[:, 1], ma_window)
            seg_z_ma = fast_moving_average(seg_pts[:, 2], ma_window)
        else:
            # 曲线段：中等预滤波
            ma_window = max(3, adaptive_win // 3)
            seg_x_ma = fast_moving_average(seg_pts[:, 0], ma_window)
            seg_y_ma = fast_moving_average(seg_pts[:, 1], ma_window)
            seg_z_ma = fast_moving_average(seg_pts[:, 2], ma_window)
        
        # 强化平滑滤波 - 专门针对Z轴抖动
        seg_x_filtered = aggressive_smoothing_filter(seg_x_ma, adaptive_win, sg_deg, geometry_type)
        seg_y_filtered = aggressive_smoothing_filter(seg_y_ma, adaptive_win, sg_deg, geometry_type)
        
        # Z轴使用额外的局部平滑处理
        seg_z_smooth1 = aggressive_smoothing_filter(seg_z_ma, adaptive_win, sg_deg, geometry_type)
        seg_z_filtered = enhanced_local_smoothing(seg_z_smooth1, adaptive_win // 2)
        
        # 组合滤波结果
        seg_filtered = np.column_stack([seg_x_filtered, seg_y_filtered, seg_z_filtered])
        sg_filtered_pts[start:end+1] = seg_filtered
        
        # Step 4: 强化异常值修正 + 额外平滑处理
        seg_corrected_x = aggressive_outlier_correction(seg_pts[:, 0], seg_x_filtered, correction_strength)
        seg_corrected_y = aggressive_outlier_correction(seg_pts[:, 1], seg_y_filtered, correction_strength)
        seg_corrected_z = aggressive_outlier_correction(seg_pts[:, 2], seg_z_filtered, correction_strength)
        
        # Step 5: 针对Z轴抖动的额外处理
        if geometry_type == "straight":
            # 直线段：多遍平滑处理
            seg_corrected_z = multi_pass_smoothing(seg_corrected_z, passes=4)
        else:
            # 曲线段：轻度额外平滑
            seg_corrected_z = multi_pass_smoothing(seg_corrected_z, passes=2)
        
        seg_corrected = np.column_stack([seg_corrected_x, seg_corrected_y, seg_corrected_z])
        processed_pts[start:end+1] = seg_corrected
    
    return processed_pts, {
        'angles': angles,
        'corners': corners,
        'segments': segments,
        'sg_filtered': sg_filtered_pts
    }

# ==== Simplified Interactive Visualization ====
class LaserLineProcessor:
    def __init__(self, lines_data):
        self.lines_data = lines_data
        self.fig = None
        self.ax1 = None  # 3D scatter plot
        self.ax2 = None  # SG filter comparison
        self.ax3 = None  # Final result comparison
        self.sliders = {}
        self.current_processed_data = None  # 存储当前处理结果
        self.setup_interactive_plot()
    
    def setup_interactive_plot(self):
        # Create main figure window
        self.fig = plt.figure(figsize=(18, 8))
        
        # Create subplot layout - 3 columns, 2 rows (plots + sliders)
        gs = self.fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.3, wspace=0.3)
        
        # Create three main plots
        self.ax1 = self.fig.add_axes([0.02, 0.35, 0.28, 0.55], projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # SG filter comparison
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # Final result comparison
        
        # Add export button
        self.export_button_ax = plt.axes([0.85, 0.92, 0.12, 0.04])
        self.export_button = plt.Button(self.export_button_ax, 'Export Current')
        self.export_button.on_clicked(self.export_current_line)
        
        self.export_all_button_ax = plt.axes([0.72, 0.92, 0.12, 0.04])
        self.export_all_button = plt.Button(self.export_all_button_ax, 'Export All')
        self.export_all_button.on_clicked(self.export_all_lines)
        
        # Slider area
        slider_area = self.fig.add_subplot(gs[1, :])
        slider_area.axis('off')
        
        # Create sliders
        self.create_sliders()
        
        # Initial plot
        self.update_plot()
        
        plt.show()
    
    def create_sliders(self):
        # Extended slider parameters - 10 sliders for 5x2 layout
        # 范围为默认值±10
        slider_params = [
            ('line_idx', 0, len(self.lines_data)-1, 0, 1),
            ('angle_thresh', -5.0, 15.0, 5.0, 0.5),  # 默认5.0, 范围5±10
            ('min_gap', 0, 20, 10, 1),  # 默认10, 范围10±10
            ('min_seg_len', -4, 16, 6, 1),  # 默认6, 范围6±10
            ('sg_win', 0, 20, 10, 1),  # 默认10, 范围10±10
            ('sg_deg', -9, 11, 1, 1),  # 默认1, 范围1±10
            ('correction_strength', -9.3, 10.7, 0.7, 0.1),  # 默认0.7, 范围0.7±10
            ('zoom_factor', -9.4, 10.6, 0.6, 0.1),  # 默认0.6, 范围0.6±10
            ('point_size', -9.0, 11.0, 1.0, 0.1),  # 默认1.0, 范围1±10
            ('line_width', -9.4, 10.6, 0.6, 0.1)  # 默认0.6, 范围0.6±10
        ]
        
        # Create sliders in 5 columns, 2 rows layout with increased spacing
        slider_height = 0.03
        slider_width = 0.13
        col_spacing = 0.18  # Increased column spacing
        row_spacing = 0.05
        start_x = 0.05
        start_y = 0.12
        
        for i, (name, min_val, max_val, init_val, step) in enumerate(slider_params):
            col = i % 5  # 5 columns
            row = i // 5  # 2 rows
            
            x_pos = start_x + col * col_spacing
            y_pos = start_y - row * row_spacing
            
            ax_slider = plt.axes([x_pos, y_pos, slider_width, slider_height])
            slider = Slider(ax_slider, name, min_val, max_val, 
                          valinit=init_val, valstep=step, valfmt='%.1f')
            slider.on_changed(self.update_plot)
            self.sliders[name] = slider
    
    def get_current_params(self):
        return {
            'angle_thresh_deg': max(0.1, self.sliders['angle_thresh'].val),
            'min_gap': max(1, int(self.sliders['min_gap'].val)),
            'min_seg_len': max(3, int(self.sliders['min_seg_len'].val)),
            'sg_win': max(3, int(self.sliders['sg_win'].val)),
            'sg_deg': max(1, int(self.sliders['sg_deg'].val)),
            'correction_strength': max(0.1, min(1.0, self.sliders['correction_strength'].val))
        }
    
    def update_plot(self, val=None):
        # Get current parameters
        line_idx = int(self.sliders['line_idx'].val)
        params = self.get_current_params()
        
        # Process data with new pipeline
        pts = self.lines_data[line_idx]
        processed_pts, diag = process_laser_line(pts, **params)
        
        # 存储当前处理结果
        self.current_processed_data = {
            'line_idx': line_idx,
            'original_pts': pts,
            'processed_pts': processed_pts,
            'params': params,
            'diag': diag
        }
        
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        indices = np.arange(len(pts))
        
        # Plot 1: 3D scatter plot - only filtered points
        point_size = self.sliders['point_size'].val
        self.ax1.scatter(processed_pts[:,0], processed_pts[:,1], processed_pts[:,2], 
                        s=point_size, alpha=0.8, c='red', label='Processed')
        
        # Auto-zoom to data range with adjustable zoom factor
        x_range = np.ptp(processed_pts[:,0])
        y_range = np.ptp(processed_pts[:,1])
        z_range = np.ptp(processed_pts[:,2])
        
        x_center = np.mean(processed_pts[:,0])
        y_center = np.mean(processed_pts[:,1])
        z_center = np.mean(processed_pts[:,2])
        
        zoom_factor = self.sliders['zoom_factor'].val
        self.ax1.set_xlim(x_center - x_range*zoom_factor, x_center + x_range*zoom_factor)
        self.ax1.set_ylim(y_center - y_range*zoom_factor, y_center + y_range*zoom_factor)
        self.ax1.set_zlim(z_center - z_range*zoom_factor, z_center + z_range*zoom_factor)
        
        self.ax1.set_title(f"3D Laser Line {line_idx} - Processed")
        self.ax1.legend()
        
        # Plot 2: Z axis final comparison (Original vs Final Processed)
        line_width = self.sliders['line_width'].val
        self.ax2.plot(indices, pts[:,2], 'b-', alpha=0.7, linewidth=line_width, label='Original Z')
        self.ax2.plot(indices, processed_pts[:,2], 'r-', linewidth=line_width, label='Final Processed Z')
        
        self.ax2.set_title("Z Axis Final Comparison")
        self.ax2.set_xlabel("Point Index")
        self.ax2.set_ylabel("Z Coordinate")
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Angle changes and corner points
        self.ax3.plot(diag['angles'], 'g-', linewidth=line_width, label='Angle Changes')
        
        # Mark corner points
        for corner in diag['corners']:
            self.ax3.axvline(corner, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        
        self.ax3.set_title("Angle Changes & Corner Points")
        self.ax3.set_xlabel("Point Index")
        self.ax3.set_ylabel("Angle (degrees)")
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Refresh display
        self.fig.canvas.draw()
    
    def export_current_line(self, event):
        """导出当前激光线的处理结果"""
        if self.current_processed_data is None:
            print("没有可导出的数据")
            return
        
        line_idx = self.current_processed_data['line_idx']
        processed_pts = self.current_processed_data['processed_pts']
        
        filename = f"processed_line_{line_idx}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"#Line {line_idx}:\n")
            for point in processed_pts:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        print(f"当前激光线已导出到: {filename}")
    
    def export_all_lines(self, event):
        """导出所有激光线的处理结果"""
        if self.current_processed_data is None:
            print("请先处理数据")
            return
        
        # 使用当前参数处理所有激光线
        params = self.current_processed_data['params']
        
        filename = "all_processed_lines.txt"
        print(f"开始处理并导出所有 {len(self.lines_data)} 条激光线...")
        
        with open(filename, 'w') as f:
            for line_idx, pts in enumerate(self.lines_data):
                print(f"处理第 {line_idx + 1}/{len(self.lines_data)} 条激光线...")
                
                # 处理当前激光线
                processed_pts, _ = process_laser_line(pts, **params)
                
                # 写入文件
                f.write(f"#Line {line_idx}:\n")
                for point in processed_pts:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                f.write("\n")
        
        total_points = sum(len(line) for line in self.lines_data)
        print(f"导出完成！")
        print(f"- 文件: {filename}")
        print(f"- 激光线数: {len(self.lines_data)}")
        print(f"- 总点数: {total_points}")

def export_processed_lines_to_txt(lines_data, output_path="processed_laser_lines.txt", 
                                  angle_thresh_deg=5.0, min_gap=10, min_seg_len=6, 
                                  sg_win=10, sg_deg=1, correction_strength=0.7):
    """
    处理所有激光线并导出到txt文件
    """
    print(f"开始处理 {len(lines_data)} 条激光线...")
    
    with open(output_path, 'w') as f:
        for line_idx, pts in enumerate(lines_data):
            print(f"处理第 {line_idx + 1}/{len(lines_data)} 条激光线...")
            
            # 处理当前激光线
            processed_pts, diag = process_laser_line(
                pts, 
                angle_thresh_deg=angle_thresh_deg,
                min_gap=min_gap,
                min_seg_len=min_seg_len,
                sg_win=sg_win,
                sg_deg=sg_deg,
                correction_strength=correction_strength
            )
            
            # 写入文件头
            f.write(f"#Line {line_idx}:\n")
            
            # 写入处理后的点数据
            for point in processed_pts:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # 添加空行分隔
            f.write("\n")
    
    print(f"处理完成！结果已保存到: {output_path}")
    return output_path

def batch_process_and_export():
    """批量处理并导出所有激光线"""
    # 使用默认参数处理所有激光线
    output_file = export_processed_lines_to_txt(
        lines_data,
        output_path="processed_laser_lines.txt",
        angle_thresh_deg=5.0,
        min_gap=10,
        min_seg_len=6,
        sg_win=10,
        sg_deg=1,
        correction_strength=0.7
    )
    
    # 统计信息
    total_points = sum(len(line) for line in lines_data)
    print(f"\n处理统计:")
    print(f"- 总激光线数: {len(lines_data)}")
    print(f"- 总点数: {total_points}")
    print(f"- 输出文件: {output_file}")
    
    return output_file

# Create interactive processor
def create_laser_processor():
    return LaserLineProcessor(lines_data)

# Usage example
if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # 批量处理模式
        print("=== 批量处理模式 ===")
        batch_process_and_export()
    else:
        # 交互式可视化模式
        print("=== 交互式可视化模式 ===")
        print("提示: 运行 'python pts3_clean.py export' 可进行批量处理并导出")
        processor = create_laser_processor()
