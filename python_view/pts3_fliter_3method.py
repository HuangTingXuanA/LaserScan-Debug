import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
from scipy.ndimage import median_filter

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

# ==== Utility functions ====
def local_median_mad(zs, idx, win):
    half = win // 2
    lo = max(0, idx-half)
    hi = min(len(zs), idx+half+1)
    segment = zs[lo:hi]
    med = np.median(segment)
    mad = np.median(np.abs(segment - med))
    return med, mad

def median_filter_1d(arr, win):
    half = win // 2
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i-half)
        hi = min(len(arr), i+half+1)
        out[i] = np.median(arr[lo:hi])
    return out

def savgol_filter_windowed(arr, win, deg):
    """改进的Savitzky-Golay滤波器，处理边界效应"""
    if win % 2 == 0:
        win += 1  # 确保窗口大小为奇数
    if win >= len(arr):
        win = len(arr) - 1 if len(arr) > 1 else 1
        if win % 2 == 0:
            win -= 1
    if win < 3:
        return arr.copy()
    
    try:
        return signal.savgol_filter(arr, win, deg, mode='nearest')
    except:
        # 备用方案：简单的多项式拟合
        half = win // 2
        out = np.zeros_like(arr)
        x_idx = np.arange(win) - half
        for i in range(len(arr)):
            lo = max(0, i-half)
            hi = min(len(arr), i+half+1)
            seg = arr[lo:hi]
            if len(seg) < 3:
                out[i] = arr[i]
                continue
            x_local = np.arange(len(seg)) - len(seg)//2
            try:
                coeffs = np.polyfit(x_local, seg, min(deg, len(seg)-1))
                out[i] = np.polyval(coeffs, 0.0)
            except:
                out[i] = np.median(seg)
        return out

def bilateral_filter_1d(arr, spatial_sigma=2.0, intensity_sigma=0.5):
    """一维双边滤波器，保持边缘的同时平滑噪声"""
    filtered = np.zeros_like(arr)
    n = len(arr)
    
    for i in range(n):
        # 定义空间权重窗口
        window_size = int(3 * spatial_sigma)
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        
        # 计算空间权重
        spatial_weights = np.exp(-0.5 * ((np.arange(start, end) - i) / spatial_sigma) ** 2)
        
        # 计算强度权重
        intensity_diffs = arr[start:end] - arr[i]
        intensity_weights = np.exp(-0.5 * (intensity_diffs / intensity_sigma) ** 2)
        
        # 组合权重
        weights = spatial_weights * intensity_weights
        weights /= np.sum(weights)
        
        # 加权平均
        filtered[i] = np.sum(weights * arr[start:end])
    
    return filtered

def adaptive_median_filter(arr, base_win=5, max_win=15):
    """自适应中值滤波器"""
    filtered = np.zeros_like(arr)
    n = len(arr)
    
    for i in range(n):
        win_size = base_win
        while win_size <= max_win:
            half = win_size // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = arr[start:end]
            
            z_med = np.median(window)
            z_min = np.min(window)
            z_max = np.max(window)
            
            # Stage A
            A1 = z_med - z_min
            A2 = z_med - z_max
            
            if A1 > 0 and A2 < 0:
                # Stage B
                B1 = arr[i] - z_min
                B2 = arr[i] - z_max
                if B1 > 0 and B2 < 0:
                    filtered[i] = arr[i]
                else:
                    filtered[i] = z_med
                break
            else:
                win_size += 2
        
        if win_size > max_win:
            filtered[i] = np.median(arr[max(0, i-max_win//2):min(n, i+max_win//2+1)])
    
    return filtered

# ==== Processing pipeline ====
def per_line_pipeline_fixed(pts3,
                      angle_thresh_deg=5.0,
                      dz_thresh=2.0,
                      mad_window=11,
                      mad_k=3.0,
                      median_win=5,
                      sg_win=15,
                      sg_deg=2,
                      min_seg_len=6,
                      min_cut_gap=10,
                      use_bilateral=True,
                      bilateral_spatial=2.0,
                      bilateral_intensity=0.5,
                      use_adaptive_median=True):
    N = pts3.shape[0]
    xs = pts3[:,0]
    ys = pts3[:,1]
    zs = pts3[:,2]
    # compute angles
    angles = np.zeros(N)
    for i in range(1, N-1):
        a = pts3[i-1]; b = pts3[i]; c = pts3[i+1]
        v1 = b - a; v2 = c - b
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1<1e-9 or n2<1e-9:
            angles[i]=0.0
        else:
            cosv = np.dot(v1, v2)/(n1*n2)
            cosv = np.clip(cosv, -1.0, 1.0)
            angles[i] = math.degrees(math.acos(cosv))
    # detect cuts
    cuts = []
    last_cut = -min_cut_gap
    for i in range(1, N-1):
        angle_cut = angles[i] > angle_thresh_deg
        dz_cut = abs(zs[i+1]-zs[i]) > dz_thresh
        med, mad = local_median_mad(zs, i, mad_window)
        mad_cut = abs(zs[i]-med) > mad_k * mad
        if (mad_cut or (angle_cut and dz_cut)) and (i - last_cut >= min_cut_gap):
            cuts.append(i)
            last_cut = i
    # build segments
    segments = []
    s = 0
    for cut in cuts:
        e = cut
        if e - s + 1 >= min_seg_len:
            segments.append((s,e))
        s = cut + 1
    if N - s >= min_seg_len:
        segments.append((s, N-1))
    aligned_clean = np.full((N,3), np.nan, dtype=float)
    for (a,b) in segments:
        seg_x = xs[a:b+1].copy()
        seg_y = ys[a:b+1].copy()
        seg_z = zs[a:b+1].copy()
        
        # 多级滤波处理
        if use_adaptive_median:
            # 第一步：自适应中值滤波去除脉冲噪声
            medx = adaptive_median_filter(seg_x, median_win, median_win*2)
            medy = adaptive_median_filter(seg_y, median_win, median_win*2)
            medz = adaptive_median_filter(seg_z, median_win, median_win*2)
        else:
            # 传统中值滤波
            medx = median_filter_1d(seg_x, median_win)
            medy = median_filter_1d(seg_y, median_win)
            medz = median_filter_1d(seg_z, median_win)
        
        if use_bilateral:
            # 第二步：双边滤波保持边缘
            bilx = bilateral_filter_1d(medx, bilateral_spatial, bilateral_intensity)
            bily = bilateral_filter_1d(medy, bilateral_spatial, bilateral_intensity)
            bilz = bilateral_filter_1d(medz, bilateral_spatial, bilateral_intensity)
        else:
            bilx, bily, bilz = medx, medy, medz
        
        # 第三步：Savitzky-Golay滤波进一步平滑
        if sg_win > 1:
            smoothx = savgol_filter_windowed(bilx, sg_win, sg_deg)
            smoothy = savgol_filter_windowed(bily, sg_win, sg_deg)
            smoothz = savgol_filter_windowed(bilz, sg_win, sg_deg)
        else:
            smoothx, smoothy, smoothz = bilx, bily, bilz
        
        # 异常值检测和移除
        residuals = np.sqrt((seg_x - smoothx)**2 + (seg_y - smoothy)**2 + (seg_z - smoothz)**2)
        medr = np.median(residuals)
        madr = np.median(np.abs(residuals - medr))
        if madr < 1e-12: madr = 1e-12
        thresh = medr + mad_k * madr
        keep_mask = residuals <= thresh
        
        smooth_kept = np.vstack([smoothx, smoothy, smoothz]).T
        smooth_kept[~keep_mask] = np.nan
        aligned_clean[a:b+1] = smooth_kept
    
    return aligned_clean, {'angles':angles, 'cuts':cuts, 'segments':segments}

# ==== Interactive Plot with Two Interfaces ====
from matplotlib.widgets import Button

class InteractiveLaserLineProcessor:
    def __init__(self, lines_data):
        self.lines_data = lines_data
        self.current_line_idx = 0
        self.current_interface = 1  # 1 for main view, 2 for filter comparison
        self.fig = None
        self.axes = None
        self.sliders = {}
        self.button = None
        self.setup_interactive_plot()
    
    def setup_interactive_plot(self):
        # Create main figure window
        self.fig = plt.figure(figsize=(16, 10))
        
        # Create subplot layout
        gs = self.fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.3, wspace=0.3)
        
        # Three main plots - will be recreated based on interface
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.gs = gs
        
        # Button in top-left corner
        self.button = Button(plt.axes([0.02, 0.92, 0.12, 0.04]), 'Filter Details')
        self.button.on_clicked(self.switch_interface)
        
        # Slider area
        slider_area = self.fig.add_subplot(gs[1, :])
        slider_area.axis('off')
        
        # Create sliders
        self.create_sliders()
        
        # Initial plot
        self.update_plot()
        
        plt.show()
    
    def create_sliders(self):
        # All 10 slider parameters
        slider_params = [
            ('line_idx', 0, len(self.lines_data)-1, 0, 1),
            ('angle_thresh', 1.0, 20.0, 10.0, 0.5),
            ('dz_thresh', 0.5, 5.0, 2.0, 0.1),
            ('mad_k', 1.0, 5.0, 3.0, 0.1),
            ('min_cut_gap', 3, 20, 10, 1),
            ('median_win', 3, 15, 5, 2),
            ('sg_win', 5, 25, 15, 2),
            ('sg_deg', 1, 4, 2, 1),
            ('bilateral_spatial', 0.5, 5.0, 2.0, 0.1),
            ('bilateral_intensity', 0.1, 2.0, 0.5, 0.05)
        ]
        
        # Create sliders in 5 columns, 2 rows layout
        slider_height = 0.03
        slider_width = 0.13
        col_spacing = 0.2
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
                          valinit=init_val, valstep=step, valfmt='%.2f')
            slider.on_changed(self.update_plot)
            self.sliders[name] = slider
    
    def get_current_params(self):
        return {
            'angle_thresh_deg': self.sliders['angle_thresh'].val,
            'dz_thresh': self.sliders['dz_thresh'].val,
            'mad_k': self.sliders['mad_k'].val,
            'min_cut_gap': int(self.sliders['min_cut_gap'].val),
            'median_win': int(self.sliders['median_win'].val),
            'sg_win': int(self.sliders['sg_win'].val),
            'sg_deg': int(self.sliders['sg_deg'].val),
            'bilateral_spatial': self.sliders['bilateral_spatial'].val,
            'bilateral_intensity': self.sliders['bilateral_intensity'].val,
            'use_bilateral': True,
            'use_adaptive_median': True
        }
    
    def switch_interface(self, event):
        self.current_interface = 2 if self.current_interface == 1 else 1
        self.update_plot()
    
    def update_plot(self, val=None):
        # Get current parameters
        line_idx = int(self.sliders['line_idx'].val)
        params = self.get_current_params()
        
        # Process data
        pts = self.lines_data[line_idx]
        aligned_clean, diag = per_line_pipeline_fixed(pts, **params)
        mask_clean = ~np.isnan(aligned_clean[:,0])
        
        # Recreate subplots based on current interface
        if self.ax1: self.ax1.remove()
        if self.ax2: self.ax2.remove()
        if self.ax3: self.ax3.remove()
        
        indices = np.arange(len(pts))
        
        if self.current_interface == 1:
            # Interface 1: Main Analysis View - create larger 3D subplot moved left
            # [left, bottom, width, height] - move left and make larger
            self.ax1 = self.fig.add_axes([0.01, 0.35, 0.4, 0.45], projection='3d')
            self.ax2 = self.fig.add_subplot(self.gs[0, 1])
            self.ax3 = self.fig.add_subplot(self.gs[0, 2])
        else:
            # Interface 2: Filter Comparison View - all 2D subplots
            self.ax1 = self.fig.add_subplot(self.gs[0, 0])
            self.ax2 = self.fig.add_subplot(self.gs[0, 1])
            self.ax3 = self.fig.add_subplot(self.gs[0, 2])
        
        if self.current_interface == 1:
            # Interface 1: Main Analysis View
            self.button.label.set_text('Filter Details')
            
            # 3D scatter plot - only filtered points with enhanced detail view
            if np.any(mask_clean):
                filtered_pts = aligned_clean[mask_clean]
                self.ax1.scatter(filtered_pts[:,0], filtered_pts[:,1], filtered_pts[:,2], 
                                s=0.5, alpha=0.8, label='Filtered', c='red')
                
                # Calculate data range for zooming
                x_range = np.ptp(filtered_pts[:,0])
                y_range = np.ptp(filtered_pts[:,1]) 
                z_range = np.ptp(filtered_pts[:,2])
                
                x_center = np.mean(filtered_pts[:,0])
                y_center = np.mean(filtered_pts[:,1])
                z_center = np.mean(filtered_pts[:,2])
                
                # Set zoom factor to show more detail
                zoom_factor = 0.6
                self.ax1.set_xlim(x_center - x_range*zoom_factor, x_center + x_range*zoom_factor)
                self.ax1.set_ylim(y_center - y_range*zoom_factor, y_center + y_range*zoom_factor)
                self.ax1.set_zlim(z_center - z_range*zoom_factor, z_center + z_range*zoom_factor)
                
            self.ax1.set_title(f"3D Laser Line {line_idx} - Filtered")
            self.ax1.legend()
            
            # Z depth final comparison
            self.ax2.plot(indices, pts[:,2], 'b-', alpha=0.7, linewidth=0.8, label='Original Z')
            self.ax2.plot(indices, aligned_clean[:,2], 'r-', linewidth=0.8, label='Final Filtered Z')
            self.ax2.set_title("Z Depth Final Comparison")
            self.ax2.set_xlabel("Point Index")
            self.ax2.set_ylabel("Z Coordinate")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Angles and cut points
            self.ax3.plot(diag['angles'], 'g-', linewidth=0.8, label='Angle Changes')
            for c in diag['cuts']:
                self.ax3.axvline(c, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            self.ax3.set_title("Angle Changes & Cut Points")
            self.ax3.set_xlabel("Point Index")
            self.ax3.set_ylabel("Angle (degrees)")
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
            
        else:
            # Interface 2: Filter Comparison View (All 2D plots)
            self.button.label.set_text('Main Analysis')
            
            # Generate intermediate filtering results
            seg_z = pts[:,2].copy()
            med_z = median_filter_1d(seg_z, params['median_win'])
            bil_z = bilateral_filter_1d(med_z, params['bilateral_spatial'], params['bilateral_intensity'])
            
            # Median filter vs original (2D plot)
            self.ax1.plot(indices, pts[:,2], 'b-', alpha=0.7, linewidth=0.6, label='Original Z')
            self.ax1.plot(indices, med_z, 'g-', linewidth=0.6, label='Median Filtered Z')
            self.ax1.set_title("Median Filter vs Original")
            self.ax1.set_xlabel("Point Index")
            self.ax1.set_ylabel("Z Coordinate")
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # Bilateral filter vs original
            self.ax2.plot(indices, pts[:,2], 'b-', alpha=0.7, linewidth=0.6, label='Original Z')
            self.ax2.plot(indices, bil_z, 'orange', linewidth=0.6, label='Bilateral Filtered Z')
            self.ax2.set_title("Bilateral Filter vs Original")
            self.ax2.set_xlabel("Point Index")
            self.ax2.set_ylabel("Z Coordinate")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Final result vs original
            self.ax3.plot(indices, pts[:,2], 'b-', alpha=0.7, linewidth=0.6, label='Original Z')
            self.ax3.plot(indices, aligned_clean[:,2], 'r-', linewidth=0.6, label='Final Filtered Z')
            self.ax3.set_title("Final Result vs Original")
            self.ax3.set_xlabel("Point Index")
            self.ax3.set_ylabel("Z Coordinate")
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
        
        # Refresh display
        self.fig.canvas.draw()

# Create interactive processor
def create_interactive_processor():
    return InteractiveLaserLineProcessor(lines_data)

# Usage example
if __name__ == "__main__":
    processor = create_interactive_processor()
