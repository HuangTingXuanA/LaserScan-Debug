#!/usr/bin/env python3
# laser_truncation_ui_final_v2.py
# Interactive per-region tuning - final fixes:
#  - removed `require_both`
#  - consec & deriv range [0.1, 3.0]
#  - final diff is color overlay (removed/red, added/green) on grayscale
#  - each region is processed with its own sliders when pressing 'q' and stored; final merge uses stored masks

import os
import math
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# remove 'q' from default matplotlib keymaps to avoid quitting
for k in list(mpl.rcParams.keys()):
    if k.startswith('keymap.'):
        vals = list(mpl.rcParams[k])
        while 'q' in vals:
            vals.remove('q')
        mpl.rcParams[k] = vals

# -----------------------
# smoothing + detection
# -----------------------
def median_prefilter(edge_map, win=15):
    if not edge_map:
        return {}
    ys = sorted(edge_map.keys())
    xs = [edge_map[y] for y in ys]
    out = {}
    half = max(1, int(win // 2))
    n = len(xs)
    for i, y in enumerate(ys):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[y] = float(np.median(xs[s:e]))
    return out

def smooth_edge(edge_map, med_win=15, pos_win=5):
    ys = sorted(edge_map.keys())
    if not ys:
        return [], []
    pref = median_prefilter(edge_map, win=med_win)
    xs = [pref[y] for y in ys]
    n = len(xs)
    half = max(1, int(pos_win // 2))
    out = []
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out.append(float(np.median(xs[s:e])))
    return ys, out

def detect_and_remove_protrusions(
    left_ys, left_sm, right_ys, right_sm,
    deriv=1.6, bw_k=4.0, pad=3, min_segment_len=15,
    consec=5, end_guard=3
):
    """
    Robust detection:
      - deriv: multiplier for slope-threshold (float, 0.1~3)
      - consec: float minimal consecutive length; converted to integer via ceil
      - bw_k: multiplier for MAD-based BW threshold
      - pad: expansion (int)
      - returns (good_ys, dbg)
    """
    left_map = {y: x for y, x in zip(left_ys, left_sm)}
    right_map = {y: x for y, x in zip(right_ys, right_sm)}
    common_ys = sorted(set(left_ys) & set(right_ys))
    if not common_ys:
        return [], {}

    L = np.array([left_map[y] for y in common_ys], dtype=float)
    R = np.array([right_map[y] for y in common_ys], dtype=float)
    ys_arr = np.array(common_ys, dtype=int)
    n = len(common_ys)

    dy = np.maximum(1.0, np.diff(ys_arr))
    slope_L = np.zeros(n); slope_R = np.zeros(n)
    slope_L[1:] = np.abs(np.diff(L) / dy); slope_R[1:] = np.abs(np.diff(R) / dy)
    slope_comb = np.maximum(slope_L, slope_R)

    BW = R - L
    bw_jump = np.zeros(n); bw_jump[1:] = np.abs(np.diff(BW) / dy)

    def mad(a):
        med = np.median(a); return np.median(np.abs(a - med))

    med_slope = np.median(slope_comb); mad_slope = mad(slope_comb)
    scale_slope = max(mad_slope * 1.4826, 1e-6)
    slope_thresh = med_slope + deriv * scale_slope

    med_bw = np.median(BW); mad_bw = mad(BW)
    scale_bw = max(mad_bw * 1.4826, 1e-6)
    bw_thresh = scale_bw * bw_k

    slope_bad = slope_comb > slope_thresh
    bw_bad = np.abs(BW - med_bw) > bw_thresh
    bw_jump_bad = bw_jump > slope_thresh

    # fixed combine logic (removed require_both toggle)
    candidate_bad = slope_bad & (bw_bad | bw_jump_bad)

    # protect ends
    if end_guard > 0:
        candidate_bad[:min(end_guard, n)] = False
        candidate_bad[-min(end_guard, n):] = False

    final_bad = np.zeros_like(candidate_bad, dtype=bool)
    i = 0
    min_run = max(1, int(math.ceil(consec)))  # convert consec float to integer run length
    while i < n:
        if candidate_bad[i]:
            j = i + 1
            while j < n and candidate_bad[j]:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                final_bad[i:j] = True
            i = j
        else:
            i += 1

    # expand
    if pad > 0:
        bad_idx = np.where(final_bad)[0]
        for idx in bad_idx:
            s = max(0, idx - pad); e = min(n, idx + pad + 1)
            final_bad[s:e] = True

    # keep runs of good indices at least min_segment_len long
    good_inds = np.where(~final_bad)[0]
    kept = []
    if good_inds.size > 0:
        runs = np.split(good_inds, np.where(np.diff(good_inds) != 1)[0] + 1)
        for r in runs:
            if len(r) >= max(1, int(min_segment_len)):
                kept.extend(r.tolist())
    good_ys = [int(common_ys[i]) for i in kept]

    dbg = {
        'common_ys': common_ys,
        'BW': BW,
        'slope_comb': slope_comb,
        'candidate_bad': candidate_bad,
        'final_bad': final_bad,
        'slope_thresh': slope_thresh,
        'bw_thresh': bw_thresh,
        'med_bw': med_bw,
        'med_slope': med_slope,
        'min_run': min_run
    }
    return good_ys, dbg

# -----------------------
# UI and region flow
# -----------------------
def extract_regions_from_binary(bin_img, min_area=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    regions = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= min_area or min(w,h) == 0:
            continue
        mask = (labels == i).astype('uint8') * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=lambda x: x.shape[0])
        pts = c.reshape(-1,2)
        edge_min = {}; edge_max = {}
        for (x,y) in pts:
            y = int(y); x = int(x)
            if y not in edge_min: edge_min[y] = x
            else: edge_min[y] = min(edge_min[y], x)
            if y not in edge_max: edge_max[y] = x
            else: edge_max[y] = max(edge_max[y], x)
        regions.append({'edge_min': edge_min, 'edge_max': edge_max, 'region_id': i})
    return regions

def show_final_comparison(before_img, region_masks):
    """
    before_img: grayscale binary
    region_masks: list of masks (uint8 0/255) per region (len == num_regions), None for unprocessed
    """
    # merge masks
    merged = np.zeros_like(before_img)
    for m in region_masks:
        if m is not None:
            merged = np.clip(merged.astype(int) + (m>0).astype(int), 0, 1).astype(np.uint8) * 255

    # create colored diff overlay on original grayscale (normalize original for background)
    bg = before_img.astype(np.float32) / 255.0
    bg_rgb = np.stack([bg,bg,bg], axis=2)  # gray background [0..1]
    added = (merged==255) & (before_img==0)   # added pixels
    removed = (before_img==255) & (merged==0) # removed pixels

    overlay = bg_rgb.copy()
    # add green for added, red for removed
    overlay[added, 0] = np.minimum(overlay[added,0]*0.3 + 0.0, 1.0)  # reduce red
    overlay[added, 1] = 1.0   # green
    overlay[added, 2] = 0.0
    overlay[removed, 0] = 1.0 # red
    overlay[removed, 1] = 0.0
    overlay[removed, 2] = 0.0

    # plot
    plt.figure(figsize=(15,6))
    plt.subplot(1,3,1); plt.imshow(before_img, cmap='gray'); plt.title('Before (binary)'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(merged, cmap='gray'); plt.title('After (merged regions)'); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay); plt.title('Diff overlay (red removed, green added)'); plt.axis('off')
    plt.tight_layout(); plt.show()

def interactive_ui(regions, before_img, start_idx=0):
    if not regions:
        print("no regions")
        return

    # defaults and slider specs
    defaults = {
        'med_win': 15, 'pos_win': 5, 'deriv': 1.56, 'bw_k': 4.0,
        'pad': 3, 'consec': 5, 'min_seg': 15
    }
    slider_specs = [
        ('med_win', 1, 41, 2, defaults['med_win']),
        ('pos_win', 1, 21, 1, defaults['pos_win']),
        ('deriv', 0.01, 5.0, 0.01, defaults['deriv']),
        ('bw_k', 3, 10, 0.1, defaults['bw_k']),
        ('pad', 0, 12, 1, defaults['pad']),
        ('consec', 2, 20, 1, defaults['consec']),
        ('min_seg', 1, 50, 1, defaults['min_seg'])
    ]

    num_regions = len(regions)
    region_idx = start_idx % num_regions
    processed = [False]*num_regions
    processed_params = [None]*num_regions
    processed_region_masks = [None]*num_regions  # store mask per region (uint8 0/255)

    # UI layout: 3x5 grid (top charts + bottom two rows for sliders)
    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(3,5, height_ratios=[3,0.9,0.9], hspace=0.35, wspace=0.25)
    ax_left = fig.add_subplot(gs[0,:3]); ax_bw = fig.add_subplot(gs[0,3:])
    cell_axes = []
    for r in [1,2]:
        for c in range(5):
            ax = fig.add_subplot(gs[r,c]); ax.axis('off'); cell_axes.append(ax)

    # checkbox removed (require_both removed), keep a Reset button
    ax_reset = fig.add_axes([0.78, 0.02, 0.09, 0.06]); btn_reset = Button(ax_reset, 'Reset')

    # create sliders (no textual value displays)
    slider_objs = {}
    for i, spec in enumerate(slider_specs):
        name, vmin, vmax, step, init = spec
        if i >= len(cell_axes):
            break
        bbox = cell_axes[i].get_position()
        ax_slider = fig.add_axes([bbox.x0 + 0.02*bbox.width, bbox.y0 + 0.18*bbox.height,
                                  bbox.width*0.72, bbox.height*0.64])
        step_param = step if (step and step >= 1) else None
        s = Slider(ax_slider, name, vmin, vmax, valinit=init, valstep=step_param)
        slider_objs[name] = s

    removed_patches = []

    def draw_removed(final_bad, common_ys_local):
        nonlocal removed_patches
        for p in removed_patches:
            try: p.remove()
            except: pass
        removed_patches = []
        if not common_ys_local or final_bad is None:
            fig.canvas.draw_idle(); return
        removed_idxs = np.where(final_bad)[0]
        if removed_idxs.size == 0:
            fig.canvas.draw_idle(); return
        runs = np.split(removed_idxs, np.where(np.diff(removed_idxs) != 1)[0] + 1)
        for r in runs:
            y0 = common_ys_local[r[0]] - 0.5; y1 = common_ys_local[r[-1]] + 0.5
            patch = ax_left.axhspan(y0, y1, color='red', alpha=0.18)
            removed_patches.append(patch)
        fig.canvas.draw_idle()

    # render region using current slider values (preview)
    def render_region(idx):
        ax_left.clear(); ax_bw.clear()
        ax_left.invert_yaxis(); ax_left.grid(True, ls='--', alpha=0.4)
        region = regions[idx]
        edge_min = region['edge_min']; edge_max = region['edge_max']
        med_win = int(round(slider_objs['med_win'].val)); pos_win = int(round(slider_objs['pos_win'].val))
        left_ys, left_sm = smooth_edge(edge_min, med_win=med_win, pos_win=pos_win)
        right_ys, right_sm = smooth_edge(edge_max, med_win=med_win, pos_win=pos_win)
        ax_left.plot(left_sm, left_ys, '-', lw=2, color='orange', label='Left Smooth')
        ax_left.plot(right_sm, right_ys, '-', lw=2, color='cyan', label='Right Smooth')
        meas_left = [edge_min[y] for y in left_ys]; meas_right = [edge_max[y] for y in right_ys]
        ax_left.scatter(meas_left, left_ys, s=8, c='darkorange', zorder=5, label='Meas L')
        ax_left.scatter(meas_right, right_ys, s=8, c='deepskyblue', zorder=5, label='Meas R')
        proc_str = "Processed" if processed[idx] else "Not Processed"
        ax_left.set_title(f"Region {regions[idx].get('region_id', idx)} - {proc_str}")
        ax_left.legend()
        common_ys = sorted(set(left_ys) & set(right_ys))
        if common_ys:
            left_map = {y:x for y,x in zip(left_ys,left_sm)}
            right_map = {y:x for y,x in zip(right_ys,right_sm)}
            BW = np.array([right_map[y]-left_map[y] for y in common_ys], dtype=float)
            ax_bw.plot(common_ys, BW, '-', lw=2)
            ax_bw.axhline(np.median(BW), linestyle='--')
            ax_bw.set_title('Bandwidth'); ax_bw.grid(True, ls='--', alpha=0.4); ax_bw.invert_xaxis()
            # shading: if processed, show stored mask's final_bad via re-evaluation with stored params
            if processed[idx] and processed_params[idx] is not None:
                p = processed_params[idx]
                _, dbg_stored = detect_and_remove_protrusions(left_ys,left_sm,right_ys,right_sm,
                                                             deriv=p['deriv'], bw_k=p['bw_k'],
                                                             pad=p['pad'], min_segment_len=p['min_seg'],
                                                             consec=p['consec'], end_guard=3)
                draw_removed(dbg_stored.get('final_bad', np.array([])), dbg_stored.get('common_ys', []))
            else:
                # temporary shading for preview with current sliders
                pcur = {
                    'deriv': float(slider_objs['deriv'].val),
                    'bw_k': float(slider_objs['bw_k'].val),
                    'pad': int(round(slider_objs['pad'].val)),
                    'min_seg': int(round(slider_objs['min_seg'].val)),
                    'consec': float(slider_objs['consec'].val)
                }
                _, dbg_temp = detect_and_remove_protrusions(left_ys,left_sm,right_ys,right_sm,
                                                           deriv=pcur['deriv'], bw_k=pcur['bw_k'],
                                                           pad=pcur['pad'], min_segment_len=pcur['min_seg'],
                                                           consec=pcur['consec'], end_guard=3)
                draw_removed(dbg_temp.get('final_bad', np.array([])), dbg_temp.get('common_ys', []))
        fig.canvas.draw_idle()

    # process current region using current sliders: create and store region mask
    def process_current_region(idx):
        region = regions[idx]
        edge_min = region['edge_min']; edge_max = region['edge_max']
        params = {
            'med_win': int(round(slider_objs['med_win'].val)),
            'pos_win': int(round(slider_objs['pos_win'].val)),
            'deriv': float(slider_objs['deriv'].val),
            'bw_k': float(slider_objs['bw_k'].val),
            'pad': int(round(slider_objs['pad'].val)),
            'consec': float(slider_objs['consec'].val),
            'min_seg': int(round(slider_objs['min_seg'].val))
        }
        left_ys,left_sm = smooth_edge(edge_min, med_win=params['med_win'], pos_win=params['pos_win'])
        right_ys,right_sm = smooth_edge(edge_max, med_win=params['med_win'], pos_win=params['pos_win'])
        good_ys, dbg = detect_and_remove_protrusions(left_ys,left_sm,right_ys,right_sm,
                                                     deriv=params['deriv'], bw_k=params['bw_k'],
                                                     pad=params['pad'], min_segment_len=params['min_seg'],
                                                     consec=params['consec'], end_guard=3)
        # construct mask for this region only
        mask = np.zeros_like(before_img, dtype=np.uint8)
        left_map = {y:x for y,x in zip(left_ys,left_sm)}
        right_map = {y:x for y,x in zip(right_ys,right_sm)}
        for y in good_ys:
            x_min = int(round(left_map[y])); x_max = int(round(right_map[y]))
            if x_max > x_min and 3 <= (x_max - x_min) <= 30:
                x0 = max(0, x_min); x1 = min(mask.shape[1]-1, x_max)
                mask[y, x0:x1+1] = 255
        processed[idx] = True
        processed_params[idx] = params
        processed_region_masks[idx] = mask
        print(f"Processed region {regions[idx].get('region_id', idx)} (kept {len(good_ys)} rows).")

    # update callback (sliders) preview (no processing)
    def on_slider_change(_=None):
        render_region(region_idx)

    # reset sliders
    def reset(_=None):
        for name, vmin, vmax, step, init in slider_specs:
            if name in slider_objs:
                slider_objs[name].set_val(init)
        render_region(region_idx)

    # key handler: q -> process current region & jump to next not-yet-processed (or finish)
    def on_key(event):
        nonlocal region_idx
        if event.key == 'q':
            process_current_region(region_idx)
            # find next unprocessed
            next_idx = None
            for offset in range(1, num_regions+1):
                cand = (region_idx + offset) % num_regions
                if not processed[cand]:
                    next_idx = cand
                    break
            if next_idx is None:
                # all done - show merged comparison (use stored masks)
                print("All regions processed. Showing final comparison.")
                show_final_comparison(before_img, processed_region_masks)
            else:
                region_idx = next_idx
                render_region(region_idx)
        elif event.key == 'r':
            reset()

    # attach slider callbacks
    for s in slider_objs.values():
        s.on_changed(on_slider_change)
    btn_reset.on_clicked(reset)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # initial render
    render_region(region_idx)

    print("Interactive UI ready.")
    print(" - Adjust sliders (preview updates live).")
    print(" - Press 'q' to PROCESS current region (stores per-region mask) and advance to next unprocessed region.")
    print(" - Press 'r' to reset sliders to defaults.")
    plt.show(block=True)

# -----------------------
# main
# -----------------------
def main():
    candidates = ['imgs/image_1.png']
    img_path = None
    for p in candidates:
        if os.path.exists(p):
            img_path = p; break
    if img_path is None:
        raise FileNotFoundError("Place image.png next to script or at /mnt/data/image.png")

    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError("Failed to load image")
    _, bin_img = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # tiny morph open
    kernel = np.ones((3,3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    regions = extract_regions_from_binary(bin_img, min_area=200)
    if not regions:
        print("No regions found.")
        return
    print(f"Found {len(regions)} regions. Launching interactive UI.")
    interactive_ui(regions, before_img=bin_img, start_idx=0)

if __name__ == "__main__":
    main()
