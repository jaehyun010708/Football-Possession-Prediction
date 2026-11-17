"""
특징 347개 = 프레임 특징 48개 + 문맥 299개. 선수 정보만 씀 (공 안 씀).

입력
  X      (프레임, 슬롯, 2)  미터, 가운데 원점
  team   (슬롯,)            0 또는 1
  period (프레임,)          1 또는 2  (문맥이 하프 경계를 넘지 않게)
  spd, acc                  preprocess.speed_acc() 결과
출력 (5 Hz)
  F      (5Hz 프레임, 347)
  idx    원본 프레임 번호 (F의 각 행이 원본 몇 번째 프레임인지)
  names  특징 이름
"""
import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from ad_tree.config import STEP, HZ, PITCH_L, PITCH_W, JOG, RUN, SPRINT, LAG_SEC, WIN_SEC

warnings.filterwarnings("ignore", category=RuntimeWarning)
L, W = PITCH_L, PITCH_W

# 문맥을 붙일 핵심 13개
KEY = ["all_spd_med", "all_spd_mean", "all_n_run", "all_n_jog", "all_sx", "all_sy", "kick_sep",
       "n_out_side", "n_out_end", "n_corner3", "n_goal_area", "all_acc_med", "team_cdist"]


def _nanstat(f, a, mask):
    return f(np.where(mask, a, np.nan), axis=1)


def frame_features(x, y, sp, ac, team):
    """프레임 특징 48개. 입력은 이미 5 Hz로 줄인 배열."""
    m = ~np.isnan(x)
    names, cols = [], []

    def add(n, v):
        names.append(n); cols.append(np.asarray(v, np.float32))

    # (가) 움직임 12개 × 전체 / 팀 A / 팀 B
    for t, tn in [(None, "all"), (0, "h"), (1, "a")]:
        mk = m if t is None else m & (team == t)[None, :]
        mv = mk & ~np.isnan(sp)
        add(f"{tn}_n", mk.sum(1))
        add(f"{tn}_spd_med", _nanstat(np.nanmedian, sp, mv))
        add(f"{tn}_spd_mean", _nanstat(np.nanmean, sp, mv))
        add(f"{tn}_spd_p90", np.nanpercentile(np.where(mv, sp, np.nan), 90, axis=1))
        add(f"{tn}_n_jog", (np.where(mv, sp, 0) > JOG).sum(1))
        add(f"{tn}_n_run", (np.where(mv, sp, 0) > RUN).sum(1))
        add(f"{tn}_n_sprint", (np.where(mv, sp, 0) > SPRINT).sum(1))
        add(f"{tn}_acc_med", _nanstat(np.nanmedian, ac, mk & ~np.isnan(ac)))
        add(f"{tn}_cx_abs", np.abs(_nanstat(np.nanmean, x, mk)))
        add(f"{tn}_cy_abs", np.abs(_nanstat(np.nanmean, y, mk)))
        add(f"{tn}_sx", _nanstat(np.nanstd, x, mk))
        add(f"{tn}_sy", _nanstat(np.nanstd, y, mk))

    # (나) 배치·위치 12개
    def left_frac(t):
        return _nanstat(np.nanmean, (x < 0).astype(float), m & (team == t)[None, :])
    a, b = left_frac(0), left_frac(1)
    add("kick_sep", np.minimum(np.maximum(a, 1 - a), np.maximum(b, 1 - b)) * np.abs(a - b))
    r = np.hypot(x, y)
    add("cent_dist_min", np.nanmin(np.where(m, r, np.nan), axis=1))
    add("n_center3", (np.where(m, r, 99) < 3).sum(1))
    cd = np.hypot(np.abs(x) - L / 2, np.abs(y) - W / 2)
    add("corner_dist_min", np.nanmin(np.where(m, cd, np.nan), axis=1))
    add("n_corner3", (np.where(m, cd, 99) < 3).sum(1))
    add("n_out_side", (np.where(m, np.abs(y), 0) > W / 2 - 0.5).sum(1))
    add("n_out_end", (np.where(m, np.abs(x), 0) > L / 2 - 0.5).sum(1))
    add("max_abs_y", np.nanmax(np.where(m, np.abs(y), np.nan), axis=1))
    add("max_abs_x", np.nanmax(np.where(m, np.abs(x), np.nan), axis=1))
    add("n_goal_area", (m & (np.abs(x) > L / 2 - 5.5) & (np.abs(y) < 9.16)).sum(1))
    add("n_pen_area", (m & (np.abs(x) > L / 2 - 16.5) & (np.abs(y) < 20.16)).sum(1))
    mh, ma = m & (team == 0)[None, :], m & (team == 1)[None, :]
    add("team_cdist", np.hypot(_nanstat(np.nanmean, x, mh) - _nanstat(np.nanmean, x, ma),
                               _nanstat(np.nanmean, y, mh) - _nanstat(np.nanmean, y, ma)))
    return np.stack(cols, 1), names


def _fill(F):
    """빈 값(NaN)을 앞 값으로 채우고, 맨 앞은 열 중앙값, 그래도 없으면 0."""
    F = F.copy()
    for j in range(F.shape[1]):
        c = F[:, j]; ok = ~np.isnan(c)
        if ok.any():
            ii = np.where(ok, np.arange(len(c)), 0); np.maximum.accumulate(ii, out=ii)
            c = c[ii]; c[np.isnan(c)] = np.nanmedian(c)
        else:
            c = np.zeros_like(c)
        F[:, j] = c
    return F


def context_features(F0, names0):
    """문맥 299개. 이 함수는 하프 하나 안에서만 호출됨 (하프 경계를 넘지 않음)."""
    key = [names0.index(k) for k in KEY]
    Fk = F0[:, key]; n = len(F0)
    feats, names = [], []
    for off in LAG_SEC:                                   # 앞뒤 시점 값 13 × 14
        k = int(round(off * HZ))
        for sgn in (-1, 1):
            idx = np.clip(np.arange(n) + sgn * k, 0, n - 1)   # 하프 끝을 넘으면 하프 첫/끝 값
            feats.append(Fk[idx])
            names += [f"{KEY[j]}@{'+' if sgn > 0 else '-'}{off}" for j in range(len(KEY))]
    for w in WIN_SEC:                                     # 구간 평균 13 × 9
        win = int(w * HZ)
        cm = uniform_filter1d(Fk, win, axis=0, mode="nearest")
        h = win // 2
        past = cm[np.clip(np.arange(n) - h, 0, n - 1)]
        fut = cm[np.clip(np.arange(n) + h, 0, n - 1)]
        feats += [cm, past, fut]
        for tag in ("c", "p", "f"):
            names += [f"{KEY[j]}~{tag}{w}s" for j in range(len(KEY))]
    return np.hstack(feats), names


def build(X, team, period, spd, acc):
    idx = np.arange(0, len(X), STEP)
    x, y = X[idx, :, 0], X[idx, :, 1]
    F0, names0 = frame_features(x, y, spd[idx], acc[idx], team)
    per = period[idx]
    out = np.zeros((len(idx), 0), np.float32); names = None
    blocks = []
    for p in np.unique(per):
        ii = np.where(per == p)[0]
        f0 = _fill(F0[ii])
        fc, nc = context_features(f0, names0)
        blocks.append((ii, np.hstack([f0, fc]).astype(np.float32)))
        names = names0 + nc
    F = np.zeros((len(idx), len(names)), np.float32)
    for ii, b in blocks:
        F[ii] = b
    return F, idx, names


def standardize(F, ref_mask):
    """경기마다 표준화: ref_mask 프레임으로 구한 중앙값·사분위 범위 사용. 정답 안 씀."""
    R = F[ref_mask]
    med = np.median(R, 0)
    iqr = np.percentile(R, 75, 0) - np.percentile(R, 25, 0)
    iqr[iqr < 1e-6] = 1.0
    return ((F - med) / iqr).astype(np.float32)
