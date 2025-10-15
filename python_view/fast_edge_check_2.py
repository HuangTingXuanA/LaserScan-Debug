#!/usr/bin/env python3
# laser_anomaly_ui_rewrite_v2.py
# Full UI + rewritten detection kernel (first-principles, robust local-baseline residual + quad/monotonic protection)
# Left: Region view; Right: Bandwidth view. Sliders below.
# Keys: 'q' -> process current region and advance; 'r' -> reset sliders.


import os, math
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks, peak_prominences

# remove default 'q' quit mapping in some backends
for k in list(mpl.rcParams.keys()):
    if k.startswith('keymap.'):
        vals = list(mpl.rcParams[k])
        while 'q' in vals:
            vals.remove('q')
        mpl.rcParams[k] = vals

eps = 1e-9

# -------------------------
# smoothing helpers (unchanged)
# -------------------------
def median_prefilter(edge_map, win=15):
    if not edge_map:
        return {}
    ys = sorted(edge_map.keys())
    xs = [float(edge_map[y]) for y in ys]
    w = max(1, int(win))
    if w % 2 == 0:
        w += 1
    half = w // 2
    n = len(xs)
    out = {}
    for i, y in enumerate(ys):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        buf = xs[start:end]
        out[y] = float(np.median(buf))
    return out

def smooth_edge(edge_map, med_win=15, pos_win=5):
    if not edge_map:
        return [], []
    pref = median_prefilter(edge_map, win=med_win)
    ys = sorted(pref.keys())
    xs = [float(pref[y]) for y in ys]
    w = max(1, int(pos_win))
    if w % 2 == 0:
        w += 1
    half = w // 2
    n = len(xs)
    smoothed = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        buf = xs[start:end]
        smoothed.append(float(np.median(buf)))
    return list(ys), smoothed

# -------------------------
# small helpers (unchanged)
# -------------------------
def running_median(arr, window):
    n = len(arr)
    if n == 0:
        return np.array([], dtype=float)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[i] = np.median(arr[s:e])
    return out

def quad_fit_rmse(y, bw):
    if len(y) < 3:
        return np.inf, 0.0, 0.0
    A = np.vstack([y**2, y, np.ones_like(y)]).T
    coef, *_ = np.linalg.lstsq(A, bw, rcond=None)
    pred = A.dot(coef)
    rmse = np.sqrt(np.mean((bw - pred)**2))
    amp = float(np.max(bw) - np.min(bw))
    return rmse, amp, float(coef[0])

# -------------------------
# detection core (保持之前思路)
# -------------------------
def detect_anomalous_runs(left_arr, right_arr, ys,
                          bw_k=4.0, abs_bw_px=3,
                          deriv_param=1.6,
                          w_amp=0.55, w_deriv=0.30, w_prom=0.15,
                          per_point_thresh=0.7, run_score_thresh=1.0,
                          consec=0.05, min_extrema=3,
                          use_multiscale=False, scales=[9,21]):
    ys = np.array(ys, dtype=int)
    L = np.array(left_arr, dtype=float)
    R = np.array(right_arr, dtype=float)
    n = len(ys)
    if n == 0:
        return [], {}

    BW = R - L
    med_bw = float(np.median(BW))
    mad = np.median(np.abs(BW - med_bw))
    scale_bw = max(mad * 1.4826, eps)

    # local baseline windows
    if use_multiscale and len(scales) >= 2:
        short_w, long_w = int(scales[0]), int(scales[1])
    else:
        short_w = max(5, min(9, n//40 or 5))
        long_w = max(15, min(31, n//10 or 15))

    base_short = running_median(BW, short_w)
    base_long = running_median(BW, long_w)
    BW_base = base_long
    residual = BW - BW_base

    local_scale = np.maximum(running_median(np.abs(BW - base_short), short_w) * 1.4826, eps)
    z_amp = np.abs(residual) / (bw_k * local_scale + eps)

    d1 = np.zeros(n, dtype=float)
    if n > 1:
        dy = np.maximum(1.0, np.diff(ys).astype(float))
        d1[1:] = np.abs(np.diff(residual) / dy)
    med_d1 = np.median(d1[1:]) if n > 1 else 0.0
    scale_d1 = max(np.median(np.abs(d1[1:] - med_d1)) * 1.4826, eps)
    z_deriv = d1 / (deriv_param * scale_d1 + eps)

    peaks = find_peaks(residual)[0]
    troughs = find_peaks(-residual)[0]
    prominences = np.zeros(n, dtype=float)
    if peaks.size > 0:
        prom_vals = peak_prominences(residual, peaks)[0]
        prominences[peaks] = prom_vals
    if troughs.size > 0:
        prom_vals_t = peak_prominences(-residual, troughs)[0]
        prominences[troughs] = np.maximum(prominences[troughs], prom_vals_t)
    max_prom = np.max(prominences) if prominences.size>0 else 1.0
    z_prom = prominences / (max_prom + eps)

    score_point = w_amp * z_amp + w_deriv * z_deriv + w_prom * z_prom
    candidate = score_point > per_point_thresh

    idxs = np.nonzero(candidate)[0]
    runs = []
    if idxs.size > 0:
        gap_tol = 1
        groups = []
        cur = [int(idxs[0])]
        for ii in idxs[1:]:
            if ii - cur[-1] <= gap_tol + 1:
                cur.append(int(ii))
            else:
                groups.append(cur)
                cur = [int(ii)]
        groups.append(cur)
        for g in groups:
            runs.append((int(g[0]), int(g[-1])))

    if consec <= 1.0:
        min_run = max(1, int(np.ceil(consec * max(1, n))))
    else:
        min_run = max(1, int(np.ceil(consec)))

    final_bad = np.zeros(n, dtype=bool)
    run_infos = []
    total_extrema = int(len(peaks) + len(troughs))

    full_dl = np.zeros(n); full_dr = np.zeros(n)
    if n > 1:
        full_dl[1:] = np.diff(L)
        full_dr[1:] = np.diff(R)

    for (s,e) in runs:
        run_idxs = np.arange(s, e+1)
        run_len = e - s + 1
        run_scores = score_point[run_idxs]
        run_mean_score = float(np.mean(run_scores))
        run_max_score = float(np.max(run_scores))
        run_mean_amp = float(np.mean(np.abs(residual[run_idxs])))
        run_max_der = float(np.max(d1[run_idxs]))
        run_prom = float(np.max(prominences[run_idxs])) if run_idxs.size>0 else 0.0

        y_sub = ys[run_idxs].astype(float)
        bw_sub = BW[run_idxs]
        rmse, amp_fit, curvature = quad_fit_rmse(y_sub, bw_sub)
        fit_confidence = 0.0
        if rmse > 0 and np.isfinite(rmse):
            fit_confidence = amp_fit / (rmse + eps)

        if run_len > 2:
            diffs = np.diff(BW_base[run_idxs])
            sign_frac = (np.sum(diffs > 0) / max(1, len(diffs)) , np.sum(diffs < 0) / max(1, len(diffs)))
            mono_strength = max(sign_frac)
        else:
            mono_strength = 0.0

        co_move_score = 0.0
        if run_len > 1:
            dl = full_dl[run_idxs]; dr = full_dr[run_idxs]
            co_move_mask = (np.sign(dl) == np.sign(dr)) & (np.abs(dl) > 0) & (np.abs(dr) > 0)
            co_move_score = float(np.mean(co_move_mask)) if co_move_mask.size>0 else 0.0

        score_run = (run_mean_score * 1.0) \
                    + (run_mean_amp / (bw_k * scale_bw + abs_bw_px + eps) * 0.6) \
                    + (run_max_der / (deriv_param * scale_d1 + eps) * 0.6) \
                    + math.log(1 + run_len) * 0.25 \
                    + (run_prom / (max_prom + eps)) * 0.25 \
                    - fit_confidence * 0.6

        exempt = False
        if co_move_score > 0.7 and run_mean_amp < max(abs_bw_px, 2.0):
            exempt = True
        if (run_len >= max(5, int(0.08 * n))) and (mono_strength > 0.85) and (run_mean_amp < (1.2 * scale_bw + eps)):
            exempt = True
        if (fit_confidence > 3.0) and (abs(curvature) < 1e-2):
            exempt = True

        adaptive_thresh = run_score_thresh * (0.75 if total_extrema <= min_extrema else 1.0)

        if (run_len >= min_run) and (score_run >= adaptive_thresh) and (not exempt):
            final_bad[run_idxs] = True

        run_infos.append({
            's': s, 'e': e, 'run_len': run_len,
            'run_mean_score': run_mean_score, 'run_max_score': run_max_score,
            'run_mean_amp': run_mean_amp, 'run_max_der': run_max_der,
            'run_prom': run_prom, 'score_run': score_run,
            'fit_conf': fit_confidence, 'curvature': curvature, 'co_move': co_move_score,
            'mono_strength': mono_strength, 'exempt': exempt
        })

    kept_idxs = np.where(~final_bad)[0]
    good_ys = [int(ys[i]) for i in kept_idxs]

    dbg = {
        'ys': ys, 'BW': BW, 'BW_base': BW_base, 'residual': residual,
        'd1': d1, 'score_point': score_point, 'candidate': candidate,
        'runs': run_infos, 'final_bad': final_bad,
        'mode_val': float(med_bw), 'med_bw': float(med_bw), 'scale_bw': float(scale_bw),
        'deriv_thresh': float(med_d1 + deriv_param * scale_d1 if n>1 else 0.0),
        'total_extrema': total_extrema, 'peaks': peaks, 'troughs': troughs,
        'z_amp': z_amp, 'z_deriv': z_deriv, 'z_prom': z_prom,
    }
    return good_ys, dbg

# -------------------------
# extract regions (unchanged)
# -------------------------
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

# -------------------------
# UI (consec range fixed; other logic mostly unchanged)
# -------------------------
def interactive_ui(regions, before_img, start_idx=0):
    if not regions:
        print("no regions"); return

    # 修改 consec 的范围为 0.001 ~ 2.0
    slider_specs = [
        ('med_win', 1, 41, 15),
        ('pos_win', 1, 21, 5),
        ('bw_k', 0.5, 8.0, 4.0),
        ('deriv_param', 0.1, 3.0, 1.6),
        ('per_point_thresh', 0.1, 1.0, 0.7),
        ('run_score_thresh', 0.1, 3.0, 1.0),
        ('consec', 0.001, 2.0, 0.05),   # <= 改动
        ('min_extrema', 0, 10, 3),
        ('use_multiscale', 0, 1, 0)
    ]

    num_regions = len(regions)
    region_idx = start_idx % num_regions
    processed_masks = [None] * num_regions
    processed_params = [None] * num_regions

    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(3,5, height_ratios=[3,0.9,0.9], hspace=0.35, wspace=0.25)
    ax_region = fig.add_subplot(gs[0, 0:3])
    ax_bw = fig.add_subplot(gs[0, 3:5])

    cell_axes = []
    for r in [1,2]:
        for c in range(5):
            ax = fig.add_subplot(gs[r,c]); ax.axis('off'); cell_axes.append(ax)

    ax_reset = fig.add_axes([0.78, 0.02, 0.09, 0.06]); btn_reset = Button(ax_reset, 'Reset')

    slider_objs = {}
    for i, spec in enumerate(slider_specs):
        name, vmin, vmax, init = spec
        if i >= len(cell_axes):
            break
        bbox = cell_axes[i].get_position()
        slider_ax = fig.add_axes([bbox.x0 + 0.02*bbox.width, bbox.y0 + 0.18*bbox.height,
                                  bbox.width*0.72, bbox.height*0.64])
        s = Slider(slider_ax, name, vmin, vmax, valinit=init)
        slider_objs[name] = s

    removed_patches = []
    def clear_removed():
        nonlocal removed_patches
        for p in removed_patches:
            try: p.remove()
            except: pass
        removed_patches = []

    def render_region(idx):
        clear_removed()
        ax_region.clear(); ax_bw.clear()
        ax_region.invert_yaxis(); ax_region.grid(True, ls='--', alpha=0.4)
        region = regions[idx]
        edge_min = region['edge_min']; edge_max = region['edge_max']
        med_win = int(round(slider_objs['med_win'].val)); pos_win = int(round(slider_objs['pos_win'].val))
        left_ys, left_sm = smooth_edge(edge_min, med_win=med_win, pos_win=pos_win)
        right_ys, right_sm = smooth_edge(edge_max, med_win=med_win, pos_win=pos_win)

        ax_region.plot(left_sm, left_ys, '-', lw=2, color='orange', label='Left Smooth')
        ax_region.plot(right_sm, right_ys, '-', lw=2, color='cyan', label='Right Smooth')
        ax_region.scatter([edge_min[y] for y in left_ys], left_ys, s=8, c='darkorange', label='Meas L')
        ax_region.scatter([edge_max[y] for y in right_ys], right_ys, s=8, c='deepskyblue', label='Meas R')

        ax_region.set_title(f"Region {regions[idx].get('region_id', idx)}")
        ax_region.legend(fontsize='small')

        common_ys = sorted(set(left_ys) & set(right_ys))
        if not common_ys:
            fig.canvas.draw_idle(); return None

        left_map = {y:x for y,x in zip(left_ys,left_sm)}
        right_map = {y:x for y,x in zip(right_ys,right_sm)}
        left_arr = [left_map[y] for y in common_ys]; right_arr = [right_map[y] for y in common_ys]

        params = {
            'bw_k': float(slider_objs['bw_k'].val),
            'abs_bw_px': int(round(slider_objs['per_point_thresh'].val*0+3)),
            'deriv_param': float(slider_objs['deriv_param'].val),
            'per_point_thresh': float(slider_objs['per_point_thresh'].val),
            'run_score_thresh': float(slider_objs['run_score_thresh'].val),
            'consec': float(slider_objs['consec'].val),
            'min_extrema': int(round(slider_objs['min_extrema'].val)),
            'use_multiscale': bool(round(slider_objs['use_multiscale'].val))
        }

        good_ys, dbg = detect_anomalous_runs(left_arr, right_arr, common_ys,
                                             bw_k=params['bw_k'],
                                             abs_bw_px=3,
                                             deriv_param=params['deriv_param'],
                                             per_point_thresh=params['per_point_thresh'],
                                             run_score_thresh=params['run_score_thresh'],
                                             consec=params['consec'],
                                             min_extrema=params['min_extrema'],
                                             use_multiscale=params['use_multiscale'],
                                             scales=[9,21])

        ys_arr = np.array(dbg.get('ys', common_ys), dtype=int)
        BW = dbg.get('BW', np.array([right_arr[i]-left_arr[i] for i in range(len(common_ys))]))
        mode_val = dbg.get('mode_val', float(np.median(BW)))
        bw_thresh = dbg.get('scale_bw', 0.0) * params['bw_k'] if dbg.get('scale_bw',0)>0 else params['bw_k']*1.0

        ax_bw.plot(ys_arr, BW, '-', lw=1.5, label='Bandwidth')
        ax_bw.axhline(mode_val, color='gray', linestyle='--', label='mode')
        ax_bw.axhline(mode_val + bw_thresh, color='r', linestyle=':', label='mode+thresh')
        ax_bw.axhline(mode_val - bw_thresh, color='r', linestyle=':', label='mode-thresh')
        ax_bw.fill_between(ys_arr, mode_val - bw_thresh, mode_val + bw_thresh, color='red', alpha=0.06)

        peaks = dbg.get('peaks', np.array([], dtype=int))
        troughs = dbg.get('troughs', np.array([], dtype=int))
        if peaks.size > 0:
            ax_bw.scatter(ys_arr[peaks], BW[peaks], c='red', marker='^', s=20, label='peaks')
        if troughs.size > 0:
            ax_bw.scatter(ys_arr[troughs], BW[troughs], c='green', marker='v', s=20, label='troughs')
        ax_bw.set_title('Bandwidth'); ax_bw.grid(True, ls='--', alpha=0.3)
        ax_bw.invert_xaxis()
        ax_bw.legend(fontsize='small')

        final_bad = dbg.get('final_bad', np.zeros(len(ys_arr), dtype=bool))
        bad_idxs = np.where(final_bad)[0]
        if bad_idxs.size > 0:
            runs = np.split(bad_idxs, np.where(np.diff(bad_idxs) != 1)[0] + 1)
            for r in runs:
                y0 = int(ys_arr[r[0]]) - 0.5; y1 = int(ys_arr[r[-1]]) + 0.5
                p = ax_region.axhspan(y0, y1, color='red', alpha=0.18)
                removed_patches.append(p)

        deriv_thresh = dbg.get('deriv_thresh', 0.0)
        min_run_display = (math.ceil(params['consec']*len(ys_arr)) if params['consec']<=1.0 else math.ceil(params['consec']))
        txt = (
            f"mode={mode_val:.2f}\n"
            f"med_bw={dbg.get('med_bw',0):.2f}\n"
            f"scale_bw={dbg.get('scale_bw',0):.2f}\n"
            f"deriv_thresh={deriv_thresh:.3g}\n"
            f"min_run={min_run_display}\n"
            f"extrema={dbg.get('total_extrema',0)}\n"
            f"multiscale={params['use_multiscale']}"
        )
        ax_bw.text(0.01, 0.98, txt, transform=ax_bw.transAxes, fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        fig.canvas.draw_idle()
        return dbg

    def process_region(idx):
        """
        关键改动：只做“滤除异常”——从原二值图 before_img 中删除检测出的异常行区间（不补点）。
        最终 processed_masks[idx] 存放的是“after”二值图（删除了异常处）。
        """
        region = regions[idx]
        edge_min = region['edge_min']; edge_max = region['edge_max']
        med_win = int(round(slider_objs['med_win'].val)); pos_win = int(round(slider_objs['pos_win'].val))
        left_ys, left_sm = smooth_edge(edge_min, med_win=med_win, pos_win=pos_win)
        right_ys, right_sm = smooth_edge(edge_max, med_win=med_win, pos_win=pos_win)
        common_ys = sorted(set(left_ys) & set(right_ys))

        # start from a copy of original binary image: we'll zero out (remove) anomalous spans
        mask_after = before_img.copy().astype(np.uint8)
        if not common_ys:
            processed_masks[idx] = mask_after; processed_params[idx] = None; return None

        left_map = {y:x for y,x in zip(left_ys,left_sm)}; right_map = {y:x for y,x in zip(right_ys,right_sm)}
        left_arr = [left_map[y] for y in common_ys]; right_arr = [right_map[y] for y in common_ys]

        params = {
            'bw_k': float(slider_objs['bw_k'].val),
            'deriv_param': float(slider_objs['deriv_param'].val),
            'per_point_thresh': float(slider_objs['per_point_thresh'].val),
            'run_score_thresh': float(slider_objs['run_score_thresh'].val),
            'consec': float(slider_objs['consec'].val),
            'min_extrema': int(round(slider_objs['min_extrema'].val)),
            'use_multiscale': bool(round(slider_objs['use_multiscale'].val))
        }
        good_ys, dbg = detect_anomalous_runs(left_arr, right_arr, common_ys,
                                             bw_k=params['bw_k'],
                                             abs_bw_px=3,
                                             deriv_param=params['deriv_param'],
                                             per_point_thresh=params['per_point_thresh'],
                                             run_score_thresh=params['run_score_thresh'],
                                             consec=params['consec'],
                                             min_extrema=params['min_extrema'],
                                             use_multiscale=params['use_multiscale'],
                                             scales=[9,21])

        # now compute bad ys and remove those spans from the after-mask
        all_ys_set = set(common_ys)
        good_set = set(good_ys)
        bad_ys = sorted(list(all_ys_set - good_set))
        for y in bad_ys:
            x_min = int(round(left_map[y])); x_max = int(round(right_map[y]))
            if x_max > x_min:
                xs = max(0, x_min); xe = min(mask_after.shape[1]-1, x_max)
                # zero out (remove) those pixels
                mask_after[y, xs:xe+1] = 0

        processed_masks[idx] = mask_after
        processed_params[idx] = params
        return dbg

    def on_change(_=None):
        render_region(region_idx)

    for s in slider_objs.values():
        s.on_changed(on_change)
    btn_reset.on_clicked(lambda ev: reset())

    def reset():
        for name, vmin, vmax, init in [(a[0], a[1], a[2], a[3]) for a in slider_specs]:
            if name in slider_objs:
                slider_objs[name].set_val(init)
        render_region(region_idx)

    def on_key(event):
        nonlocal region_idx
        if event.key == 'q':
            dbg = process_region(region_idx)
            print(f"Processed region {regions[region_idx].get('region_id', region_idx)}")
            # next unprocessed
            nxt = None
            for off in range(1, num_regions+1):
                cand = (region_idx + off) % num_regions
                if processed_masks[cand] is None:
                    nxt = cand; break
            if nxt is None:
                # merge all after-masks by logical AND with original (they are already subsets of original)
                merged_after = np.zeros_like(before_img, dtype=np.uint8)
                first = True
                for m in processed_masks:
                    if m is not None:
                        if first:
                            merged_after = (m>0).astype(np.uint8)*255
                            first = False
                        else:
                            # accumulate union of remaining pixels (we removed anomalies independently per region)
                            merged_after = np.clip(merged_after.astype(int) + (m>0).astype(int), 0, 1).astype(np.uint8)*255
                show_final_comparison(before_img, merged_after)
            else:
                region_idx = nxt
                render_region(region_idx)
        elif event.key == 'r':
            reset()

    fig.canvas.mpl_connect('key_press_event', on_key)

    render_region(region_idx)
    print("UI ready. Press 'q' to process current region & advance; 'r' to reset sliders.")
    plt.show(block=True)

# -------------------------
# final comparison
# -------------------------
def show_final_comparison(before_img, merged_after):
    before_bin = (before_img > 0)
    after_bin = (merged_after > 0)
    removed = before_bin & (~after_bin)
    added = after_bin & (~before_bin)
    bg = before_img.astype(np.float32)/255.0
    overlay = np.stack([bg,bg,bg], axis=2)
    overlay[removed,0] = 1.0; overlay[removed,1] = 0.0; overlay[removed,2] = 0.0
    overlay[added,0] = 0.0; overlay[added,1] = 1.0; overlay[added,2] = 0.0
    plt.figure(figsize=(15,6))
    plt.subplot(1,3,1); plt.imshow(before_img, cmap='gray'); plt.title('Before'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(merged_after, cmap='gray'); plt.title('After'); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(overlay); plt.title('Diff overlay (red removed, green added)'); plt.axis('off')
    plt.show()

# -------------------------
# main
# -------------------------
def main():
    candidates = ['imgs/image_2.png']
    img_path = None
    for p in candidates:
        if os.path.exists(p):
            img_path = p; break
    if img_path is None:
        raise FileNotFoundError("Place image.png next to script or at /mnt/data/image.png or update candidates list")
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError("Failed to load image")
    _, bin_img = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    regions = extract_regions_from_binary(bin_img, min_area=200)
    if not regions:
        print("No regions found."); return
    print(f"Found {len(regions)} regions.")
    interactive_ui(regions, before_img=bin_img, start_idx=0)

if __name__ == "__main__":
    main()
