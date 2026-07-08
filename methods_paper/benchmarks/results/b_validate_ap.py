#!/usr/bin/env python3.13
"""Validate fluorostats.validate instance metrics against independent references.

Checks (methods-paper benchmark):
  1. Perfect match           -> F1=AP=precision=recall=1.0
  2. Known TP/FP/FN          -> exact tp/fp/fn and F1 = 2tp/(2tp+fp+fn)
  3. IoU threshold behavior  -> match below controlled IoU, no-match above
  4. average_precision       -> Kaggle DSB2018 AP = TP/(TP+FP+FN) averaged
  5. instance_f1 vs brute-force greedy-IoU matcher on random labeled images

References are implemented directly here (no external installs).
"""
import csv
import numpy as np
from fluorostats import validate

RESULTS = "/Users/rsiegelmann/Downloads/Projects/fluorostats/methods_paper/benchmarks/results"
TOL = 1e-9

rows = []  # (check, quantity, fluorostats, reference, pass)


def record(check, quantity, fs_val, ref_val, ok):
    rows.append((check, quantity, fs_val, ref_val, "PASS" if ok else "FAIL"))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {check} | {quantity}: fluorostats={fs_val}  ref={ref_val}")


# ---------------------------------------------------------------------------
# Independent reference: IoU between two boolean masks
# ---------------------------------------------------------------------------
def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Independent reference: greedy one-to-one IoU matcher on labeled images.
# Sort all (pred,gt) pairs with IoU >= threshold by descending IoU; greedily
# assign, forbidding reuse of any pred or gt. Returns tp, fp, fn.
# ---------------------------------------------------------------------------
def ref_greedy_match(pred, gt, thr):
    pred_ids = [i for i in np.unique(pred) if i != 0]
    gt_ids = [i for i in np.unique(gt) if i != 0]
    pred_masks = {i: (pred == i) for i in pred_ids}
    gt_masks = {j: (gt == j) for j in gt_ids}
    pairs = []
    for i in pred_ids:
        for j in gt_ids:
            iou = mask_iou(pred_masks[i], gt_masks[j])
            if iou >= thr:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g = set(), set()
    tp = 0
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn


def ref_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


# ===========================================================================
# CHECK 1: Perfect match
# ===========================================================================
def check1():
    gt = np.zeros((40, 40), int)
    gt[3:9, 3:9] = 1
    gt[15:22, 15:22] = 2
    gt[28:33, 5:12] = 3
    pred = gt.copy()
    f1 = validate.instance_f1(pred, gt, 0.5)
    ap = validate.average_precision(pred, gt)
    record("1-perfect", "f1", f1["f1"], 1.0, abs(f1["f1"] - 1.0) < TOL)
    record("1-perfect", "precision", f1["precision"], 1.0, abs(f1["precision"] - 1.0) < TOL)
    record("1-perfect", "recall", f1["recall"], 1.0, abs(f1["recall"] - 1.0) < TOL)
    record("1-perfect", "mAP", ap["mAP"], 1.0, abs(ap["mAP"] - 1.0) < TOL)


# ===========================================================================
# CHECK 2: Known TP / FP / FN construction
#   k=3 matched objects (identical in pred & gt), m=2 false positives
#   (pred-only), n=2 misses (gt-only). Objects are well separated so each
#   matched pair has IoU=1.0 and there is no cross-talk.
# ===========================================================================
def check2():
    k, m, n = 3, 2, 2
    shape = (60, 60)
    gt = np.zeros(shape, int)
    pred = np.zeros(shape, int)

    # k matched objects, placed on a grid, identical in both
    lbl = 1
    placements = [(2, 2), (2, 20), (2, 38)]  # k=3
    for (r, c) in placements[:k]:
        gt[r:r + 6, c:c + 6] = lbl
        pred[r:r + 6, c:c + 6] = lbl
        lbl += 1

    # n misses: gt-only objects
    miss_places = [(20, 2), (20, 20)]  # n=2
    for (r, c) in miss_places[:n]:
        gt[r:r + 6, c:c + 6] = lbl
        lbl += 1

    # m false positives: pred-only objects (fresh labels, no gt overlap)
    fp_places = [(40, 2), (40, 20)]  # m=2
    plbl = lbl
    for (r, c) in fp_places[:m]:
        pred[r:r + 6, c:c + 6] = plbl
        plbl += 1

    res = validate.match_instances(pred, gt, 0.5)
    exp_tp, exp_fp, exp_fn = k, m, n
    record("2-known", "tp", res["tp"], exp_tp, res["tp"] == exp_tp)
    record("2-known", "fp", res["fp"], exp_fp, res["fp"] == exp_fp)
    record("2-known", "fn", res["fn"], exp_fn, res["fn"] == exp_fn)

    f1 = validate.instance_f1(pred, gt, 0.5)["f1"]
    exp_f1 = ref_f1(exp_tp, exp_fp, exp_fn)  # 2*3/(6+2+2)=0.6
    record("2-known", "f1", f1, exp_f1, abs(f1 - exp_f1) < TOL)


# ===========================================================================
# CHECK 3: IoU threshold behavior
#   Build one pred / one gt object with a KNOWN IoU. Two axis-aligned
#   rectangles of area A each, overlapping in an intersection of area I.
#   IoU = I / (2A - I). Verify match at a threshold below IoU and no match
#   above it.
# ===========================================================================
def check3():
    shape = (30, 60)
    gt = np.zeros(shape, int)
    pred = np.zeros(shape, int)
    # gt rectangle: rows 5:15 (10), cols 5:15 (10) -> area 100
    gt[5:15, 5:15] = 1
    # pred rectangle: rows 5:15 (10), cols 10:20 (10) -> area 100
    pred[5:15, 10:20] = 1
    # intersection: cols 10:15 -> 10*5 = 50 ; union = 100+100-50 = 150
    known_iou = 50 / 150  # = 1/3
    # confirm our geometry via reference
    ref_iou = mask_iou(gt == 1, pred == 1)
    record("3-iou", "geometry_iou", ref_iou, known_iou, abs(ref_iou - known_iou) < TOL)

    thr_below = known_iou - 0.05
    thr_above = known_iou + 0.05

    below = validate.match_instances(pred, gt, thr_below)
    above = validate.match_instances(pred, gt, thr_above)
    record("3-iou", f"tp@thr={thr_below:.3f}(below)", below["tp"], 1, below["tp"] == 1)
    record("3-iou", f"tp@thr={thr_above:.3f}(above)", above["tp"], 0, above["tp"] == 0)
    # matched IoU value reported should equal the known IoU when matched
    if below["matched_ious"]:
        mi = below["matched_ious"][0]
        record("3-iou", "matched_iou_value", mi, known_iou, abs(mi - known_iou) < TOL)


# ===========================================================================
# CHECK 4: average_precision == mean over thresholds of TP/(TP+FP+FN)
#   Construct a scene where different thresholds give different TP counts,
#   so the average is non-trivial. Compute the reference AP independently
#   with the greedy matcher and compare per-threshold and mAP.
# ===========================================================================
def check4():
    shape = (40, 120)
    gt = np.zeros(shape, int)
    pred = np.zeros(shape, int)

    # Object A: perfect match (IoU 1.0)
    gt[5:15, 5:15] = 1
    pred[5:15, 5:15] = 1

    # Object B: partial overlap with a KNOWN IoU ~0.6 range
    # gt B: rows 20:30, cols 5:15 (area 100)
    gt[20:30, 5:15] = 2
    # pred B: rows 20:30, cols 7:17 -> intersection cols 7:15 = 10*8=80
    # union = 100+100-80 = 120 -> IoU = 80/120 = 0.6667
    pred[20:30, 7:17] = 2

    # Object C: gt-only miss
    gt[5:15, 40:50] = 3
    # Object D: pred-only false positive
    pred[20:30, 40:50] = 3

    thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    ap = validate.average_precision(pred, gt, thresholds)

    per_ref = {}
    ap_vals = []
    for thr in thresholds:
        tp, fp, fn = ref_greedy_match(pred, gt, thr)
        val = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        per_ref[thr] = val
        ap_vals.append(val)
    ref_map = float(np.mean(ap_vals))

    ok_all = True
    for thr in thresholds:
        fs = ap["per_threshold"][thr]
        rf = per_ref[thr]
        ok = abs(fs - rf) < TOL
        ok_all = ok_all and ok
        record("4-ap", f"per_thr@{thr}", round(fs, 6), round(rf, 6), ok)
    record("4-ap", "mAP", round(ap["mAP"], 6), round(ref_map, 6), abs(ap["mAP"] - ref_map) < TOL)


# ===========================================================================
# CHECK 5: instance_f1 vs independent brute-force greedy matcher, random imgs
# ===========================================================================
def make_random_labels(rng, shape=(80, 80), n_obj=8, size=7):
    """Place n_obj non-overlapping square blobs at random positions."""
    lab = np.zeros(shape, int)
    placed = 0
    tries = 0
    lbl = 1
    while placed < n_obj and tries < 500:
        tries += 1
        r = rng.integers(0, shape[0] - size)
        c = rng.integers(0, shape[1] - size)
        if lab[r:r + size, c:c + size].any():
            continue
        lab[r:r + size, c:c + size] = lbl
        lbl += 1
        placed += 1
    return lab


def perturb(rng, gt, drop_p=0.25, add_p=0.25, shift_max=3):
    """Create a pred from gt: drop some objects, shift some, add spurious."""
    shape = gt.shape
    pred = np.zeros(shape, int)
    out_lbl = 1
    ids = [i for i in np.unique(gt) if i != 0]
    for i in ids:
        if rng.random() < drop_p:
            continue  # dropped -> a miss (FN)
        mask = gt == i
        rr, cc = np.where(mask)
        dr = rng.integers(-shift_max, shift_max + 1)
        dc = rng.integers(-shift_max, shift_max + 1)
        nr = np.clip(rr + dr, 0, shape[0] - 1)
        nc = np.clip(cc + dc, 0, shape[1] - 1)
        pred[nr, nc] = out_lbl
        out_lbl += 1
    # add spurious objects (FPs)
    size = 7
    n_add = rng.integers(0, 4)
    for _ in range(n_add):
        if rng.random() < add_p:
            r = rng.integers(0, shape[0] - size)
            c = rng.integers(0, shape[1] - size)
            if pred[r:r + size, c:c + size].any():
                continue
            pred[r:r + size, c:c + size] = out_lbl
            out_lbl += 1
    return pred


def check5():
    rng = np.random.default_rng(20260708)
    n_trials = 40
    thr = 0.5
    all_ok = True
    mism = 0
    for t in range(n_trials):
        gt = make_random_labels(rng, n_obj=int(rng.integers(4, 10)))
        pred = perturb(rng, gt)
        fs = validate.instance_f1(pred, gt, thr)
        tp, fp, fn = ref_greedy_match(pred, gt, thr)
        ref_f1v = ref_f1(tp, fp, fn)
        ok = (fs["tp"] == tp and fs["fp"] == fp and fs["fn"] == fn
              and abs(fs["f1"] - ref_f1v) < 1e-9)
        if not ok:
            mism += 1
            all_ok = False
            print(f"    trial {t} MISMATCH fs=(tp{fs['tp']},fp{fs['fp']},fn{fs['fn']},f1{fs['f1']:.4f})"
                  f" ref=(tp{tp},fp{fp},fn{fn},f1{ref_f1v:.4f})")
    record("5-random", f"{n_trials}_trials_agree", f"{n_trials - mism}/{n_trials}",
           f"{n_trials}/{n_trials}", all_ok)


def main():
    check1()
    check2()
    check3()
    check4()
    check5()

    # write CSV
    csv_path = f"{RESULTS}/b_validate_ap.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "quantity", "fluorostats", "reference", "result"])
        w.writerows(rows)

    n_pass = sum(1 for r in rows if r[4] == "PASS")
    n_total = len(rows)

    # write markdown
    md_path = f"{RESULTS}/b_validate_ap.md"
    with open(md_path, "w") as fh:
        fh.write("# Benchmark B: Instance metric validation (instance_f1, "
                 "average_precision, match_instances)\n\n")
        fh.write(f"fluorostats 0.5.0. **{n_pass}/{n_total} checks PASS.**\n\n")
        fh.write("References implemented independently in this script: a mask-IoU "
                 "function, a brute-force greedy one-to-one IoU matcher, and the "
                 "Kaggle DSB2018 AP formula (TP/(TP+FP+FN) averaged over thresholds).\n\n")
        fh.write("| Check | Quantity | fluorostats | Reference | Result |\n")
        fh.write("|---|---|---|---|---|\n")
        for c, q, fs, rf, res in rows:
            fh.write(f"| {c} | {q} | {fs} | {rf} | {res} |\n")

    print(f"\n{n_pass}/{n_total} checks PASS")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
