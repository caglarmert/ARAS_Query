"""
ADEM — Synthetic Data Perception Study
=======================================
A four-stage human-perception questionnaire for evaluating synthetic remote-sensing
imagery, data-augmentation transforms, mask quality, and conditional generation.

Stages
------
1. Same / different identity under augmentation transforms (with different-image controls)
2. Image–mask correctness (with swapped-mask controls)
3. Best conditional generation among 3 generators for a given mask (single best pick)
4. Perceived-realism Likert (1 = completely synthetic … 5 = completely real) across all sources

Design notes
------------
* Ground truth (source folder, transform, correct pairing, control flag) is generated
  server-side, stored per participant, and re-attached on submit so participants never
  receive it in the payload. The quality metrics (FID/KID/IS/LPIPS/SSIM/MS-SSIM/PSNR)
  are intentionally NOT shown to participants — analyse them later against these logs.
* Every path and per-stage sample count is configurable via environment variables so the
  repo folder layout can change without touching code.
* Transforms are applied on the fly with Pillow + NumPy (deterministic per item) so the
  repo only needs the source images, not pre-rendered augmentation folders.
"""

import os
import io
import json
import uuid
import random
import hashlib
from datetime import datetime, timezone

import numpy as np
import cv2
import albumentations as A
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, abort

app = Flask(__name__)

# ----------------------------------------------------------------------------------
# CONFIGURATION  (override any of these with environment variables on Render)
# ----------------------------------------------------------------------------------
DATA_ROOT = os.environ.get("DATA_ROOT", ".")


def _p(env_key, default):
    return os.environ.get(env_key, default)


PATHS = {
    # Stage 1 — real images that get augmentation transforms applied
    "stage1_src": _p("S1_SRC", "samplednoref/ARAS400k_real"),

    # Stage 2 — same-named image/mask pairs for real and synthetic domains
    "stage2": {
        "real":  {"img": _p("S2_REAL_IMG",  "real/images"),
                  "mask": _p("S2_REAL_MASK", "real/masks")},
        "synth": {"img": _p("S2_SYNTH_IMG", "synth/images"),
                  "mask": _p("S2_SYNTH_MASK", "synth/masks")},
    },

    # Stage 3 — conditioning mask + real reference + 3 generators (same filename across folders)
    "stage3_mask": _p("S3_MASK", "sampled/conditioning_images"),
    "stage3_real": _p("S3_REAL", "sampled/ARAS400k-cnet"),
    "stage3_gen": {
        "ARAS-CSD-1E7":   _p("S3_GEN_A", "sampled/ARAS-CSD-1E7"),
        "ARAS-CSD-2":     _p("S3_GEN_B", "sampled/ARAS-CSD-2"),
        "ARAS-C-UNetGAN": _p("S3_GEN_C", "sampled/ARAS-C-UNetGAN"),
    },

    # Stage 4 — realism rating across every source. (folder, human label, is_real)
    "stage4": [
        ("sampled/ARAS400k-cnet",      "ARAS400k-cnet",        True),
        ("sampled/ARAS-CSD-1E7",       "ARAS-CSD-1E7",         False),
        ("sampled/ARAS-CSD-2",         "ARAS-CSD-2",           False),
        ("sampled/ARAS-C-UNetGAN",     "ARAS-C-UNetGAN",       False),
        ("samplednoref/ARAS400k_real", "ARAS400k_real",        True),
        ("samplednoref/ARAS400k_synth","ARAS400k_synth",       False),
        ("samplednoref/diverse",       "diverse (synth)",      False),
        ("samplednoref/BELDECN",       "BELDECN (real)",       True),
        ("samplednoref/BELDEK",        "BELDEK (real)",        True),
    ],
}

# Per-stage sample counts — dial these down freely. (env-overridable)
COUNTS = {
    "stage1_n":          int(_p("S1_N", "2")),        # base images; each shown under 7 transforms
    "stage1_controls":   int(_p("S1_CTRL", "6")),     # different-image control pairs
    "stage2_m":          int(_p("S2_M", "10")),       # image-mask pairs (~half made incorrect)
    "stage2_wrong_frac": float(_p("S2_WRONG", "0.5")),# fraction with swapped (wrong) masks
    "stage3_k":          int(_p("S3_K", "10")),       # mask + 3-generator comparisons
    "stage4_per_folder": int(_p("S4_PER", "1")),      # images sampled per source folder
    "attention_checks":  int(_p("ATTN", "1")),        # instructed-response checks per applicable stage
}

TRANSFORMS = ["baseline", "noise", "perspective", "rotation", "flip", "resizedcrop", "combined"]

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# 7-class segmentation legend (shared by frontend; kept here for reference / future server use)
LEGEND = [
    ("Tree",     (0, 100, 0)),
    ("Shrub",    (255, 182, 193)),
    ("Grass",    (154, 205, 50)),
    ("Crop",     (255, 215, 0)),
    ("Built-up", (139, 69, 19)),
    ("Barren",   (211, 211, 211)),
    ("Water",    (0, 0, 255)),
]

# In-memory plan store: participant_id -> {trial_id: ground_truth_dict}
# Ephemeral (resets on dyno restart); submit degrades gracefully if a plan is missing.
PLANS = {}


# ----------------------------------------------------------------------------------
# FILE HELPERS
# ----------------------------------------------------------------------------------
def _abs(rel):
    return os.path.normpath(os.path.join(DATA_ROOT, rel))


def list_images(rel_folder):
    folder = _abs(rel_folder)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def sample_files(rel_folder, k, rng):
    files = list_images(rel_folder)
    if not files:
        return []
    if k >= len(files):
        rng.shuffle(files)
        return files
    return rng.sample(files, k)


def img_url(rel_folder, filename):
    return f"/image?dir={rel_folder}&f={filename}"


def transform_url(rel_folder, filename, transform):
    return f"/transform?dir={rel_folder}&f={filename}&t={transform}"


# ----------------------------------------------------------------------------------
# AUGMENTATION TRANSFORMS  (deterministic per image+transform)
# ----------------------------------------------------------------------------------
def _seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2**32)


# These mirror EXACTLY the Albumentations pipelines used in training / evaluation.
# For the questionnaire we want each named condition to be visibly applied, so the
# per-op probabilities are forced to 1.0 by default (set FORCE_APPLY=0 to use the
# original training probabilities, in which case some trials show no change).
FORCE_APPLY = _p("FORCE_APPLY", "1") == "1"


def _pr(orig):
    """Probability for an op: forced to 1.0 for display, else the training value."""
    return 1.0 if FORCE_APPLY else orig


def _build_augs():
    return {
        "noise": A.Compose([
            A.GaussNoise(p=_pr(0.3)),
        ]),
        "perspective": A.Compose([
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=_pr(0.25)),
        ]),
        "rotation": A.Compose([
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-45, 45),
                     interpolation=cv2.INTER_LINEAR, p=_pr(0.25)),
        ]),
        "flip": A.Compose([
            A.HorizontalFlip(p=_pr(0.5)),
            A.VerticalFlip(p=_pr(0.5)),
        ]),
        "resizedcrop": A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33),
                                interpolation=cv2.INTER_LINEAR, p=_pr(0.25)),
            A.Resize(256, 256),
        ]),
        "combined": A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33),
                                interpolation=cv2.INTER_LINEAR, p=_pr(0.25)),
            A.Resize(256, 256),
            A.HorizontalFlip(p=_pr(0.5)),
            A.VerticalFlip(p=_pr(0.5)),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=_pr(0.25)),
            A.GaussNoise(var_limit=(10.0, 50.0), p=_pr(0.3)),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-45, 45),
                     interpolation=cv2.INTER_LINEAR, p=_pr(0.25)),
        ]),
    }


_AUGS = None


def _augs():
    global _AUGS
    if _AUGS is None:
        _AUGS = _build_augs()
    return _AUGS


def apply_transform(img, name, seed):
    """Apply a named Albumentations augmentation deterministically (per image+transform)."""
    img = img.convert("RGB")
    if name == "baseline":
        return img
    aug = _augs().get(name)
    if aug is None:
        return img
    # Seed both RNGs so a given trial always renders the same augmented image.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    arr = np.asarray(img)
    out = aug(image=arr)["image"]
    return Image.fromarray(out)


# ----------------------------------------------------------------------------------
# PLAN BUILDERS  — each trial: {trial_id, type, display..., gt:{...hidden...}}
# ----------------------------------------------------------------------------------
def _tid(prefix, i):
    return f"{prefix}_{i:03d}"


def build_stage1(rng):
    """Same / different under transforms. Real source images; transformed self = 'same',
    transformed *other* image = 'different' control."""
    src = PATHS["stage1_src"]
    base = sample_files(src, COUNTS["stage1_n"], rng)
    pool = list_images(src)
    trials = []
    i = 0
    for fn in base:
        for t in TRANSFORMS:
            trials.append({
                "trial_id": _tid("s1", i), "type": "s1_samediff",
                "left":  img_url(src, fn),
                "right": transform_url(src, fn, t),
                "gt": {"source": src, "filename": fn, "transform": t,
                       "expected": "same", "is_control": False},
            })
            i += 1
    # different-image controls: pair an image with a *transformed different* image
    others = [f for f in pool if f not in base] or pool
    for _ in range(COUNTS["stage1_controls"]):
        if len(pool) < 2:
            break
        a = rng.choice(base or pool)
        b = rng.choice([x for x in others if x != a] or pool)
        t = rng.choice([x for x in TRANSFORMS if x != "baseline"])
        trials.append({
            "trial_id": _tid("s1", i), "type": "s1_samediff",
            "left":  img_url(src, a),
            "right": transform_url(src, b, t),
            "gt": {"source": src, "filename": f"{a}|{b}", "transform": t,
                   "expected": "different", "is_control": True},
        })
        i += 1
    rng.shuffle(trials)
    return trials


def build_stage2(rng):
    """Image-mask correctness. Half real / half synth; a fraction get a swapped (wrong) mask."""
    cfg = PATHS["stage2"]
    m = COUNTS["stage2_m"]
    per_domain = {"real": m // 2, "synth": m - m // 2}
    trials, i = [], 0
    for domain, n in per_domain.items():
        img_dir, mask_dir = cfg[domain]["img"], cfg[domain]["mask"]
        # only filenames that exist in BOTH images and masks
        common = [f for f in list_images(img_dir) if f in set(list_images(mask_dir))]
        if not common:
            continue
        chosen = common if n >= len(common) else rng.sample(common, n)
        n_wrong = int(round(len(chosen) * COUNTS["stage2_wrong_frac"]))
        wrong_set = set(rng.sample(chosen, n_wrong)) if n_wrong else set()
        for fn in chosen:
            correct = fn not in wrong_set
            if correct:
                mask_fn = fn
            else:
                mask_fn = rng.choice([x for x in common if x != fn] or [fn])
            trials.append({
                "trial_id": _tid("s2", i), "type": "s2_maskcheck",
                "image": img_url(img_dir, fn),
                "mask":  img_url(mask_dir, mask_fn),
                "gt": {"domain": domain, "filename": fn, "mask_filename": mask_fn,
                       "expected": "yes" if correct else "no",
                       "is_control": (not correct), "control_kind": "swapped_mask"},
            })
            i += 1
    rng.shuffle(trials)
    _insert_attention(trials, "s2_maskcheck", rng)
    return trials


def build_stage3(rng):
    """One conditioning mask + the 3 generators' outputs (same filename). Single best pick."""
    mask_dir = PATHS["stage3_mask"]
    real_dir = PATHS["stage3_real"]
    gens = PATHS["stage3_gen"]
    gen_names = list(gens.keys())
    real_set = set(list_images(real_dir))
    # filenames present in the mask folder AND all generator folders
    sets = [set(list_images(d)) for d in gens.values()]
    common = [f for f in list_images(mask_dir) if all(f in s for s in sets)]
    chosen = common if COUNTS["stage3_k"] >= len(common) else rng.sample(common, COUNTS["stage3_k"])
    trials, i = [], 0
    for fn in chosen:
        order = gen_names[:]
        rng.shuffle(order)  # randomise the 3 positions to kill position bias
        options = [{"key": g, "image": img_url(gens[g], fn)} for g in order]
        trials.append({
            "trial_id": _tid("s3", i), "type": "s3_bestpick",
            "reference": img_url(real_dir, fn) if fn in real_set else None,
            "mask": img_url(mask_dir, fn),
            "options": options,
            "gt": {"filename": fn, "position_order": order, "is_control": False},
        })
        i += 1
    _insert_attention(trials, "s3_bestpick", rng)
    return trials


def build_stage4(rng):
    """Perceived realism Likert across every source folder."""
    trials, i = [], 0
    for folder, label, is_real in PATHS["stage4"]:
        for fn in sample_files(folder, COUNTS["stage4_per_folder"], rng):
            trials.append({
                "trial_id": _tid("s4", i), "type": "s4_realism",
                "image": img_url(folder, fn),
                "gt": {"source": folder, "label": label, "is_real": is_real,
                       "filename": fn, "is_control": False},
            })
            i += 1
    rng.shuffle(trials)
    _insert_attention(trials, "s4_realism", rng)
    return trials


def _insert_attention(trials, render_as, rng):
    """Sprinkle instructed-response attention checks into a stage."""
    for _ in range(COUNTS["attention_checks"]):
        if not trials:
            return
        target = rng.randint(1, 5)
        pos = rng.randint(0, len(trials))
        trials.insert(pos, {
            "trial_id": f"attn_{render_as}_{pos}", "type": "attention",
            "render_as": render_as, "target": target,
            "prompt": f"Attention check — please select option {target}.",
            "gt": {"expected": target, "is_control": True, "control_kind": "attention"},
        })


# ----------------------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/study")
def api_study():
    pid = request.args.get("pid") or uuid.uuid4().hex[:12]
    rng = random.Random(_seed(pid, datetime.now().timestamp()))
    stages = [
        {"id": "stage1", "trials": build_stage1(rng)},
        {"id": "stage2", "trials": build_stage2(rng)},
        {"id": "stage3", "trials": build_stage3(rng)},
        {"id": "stage4", "trials": build_stage4(rng)},
    ]
    # store ground truth server-side; strip it from the client payload
    gt_store = {}
    for st in stages:
        for tr in st["trials"]:
            gt_store[tr["trial_id"]] = {"stage": st["id"], **tr.pop("gt")}
    PLANS[pid] = gt_store
    return jsonify({"participant_id": pid, "legend": LEGEND, "stages": stages})


def _safe_dir(rel):
    """Resolve a requested relative dir and ensure it stays under DATA_ROOT."""
    root = os.path.abspath(DATA_ROOT)
    target = os.path.abspath(_abs(rel))
    if os.path.commonpath([root, target]) != root:
        abort(403)
    return target


@app.route("/image")
def serve_image():
    rel = request.args.get("dir", "")
    fn = request.args.get("f", "")
    path = os.path.join(_safe_dir(rel), fn)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/transform")
def serve_transform():
    rel = request.args.get("dir", "")
    fn = request.args.get("f", "")
    t = request.args.get("t", "baseline")
    path = os.path.join(_safe_dir(rel), fn)
    if not os.path.isfile(path):
        abort(404)
    img = Image.open(path)
    out = apply_transform(img, t, _seed(rel, fn, t))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# Google Sheets column order (one row per trial). Ground truth is flattened into
# explicit columns (not all are populated for every stage).
SHEET_HEADER = [
    "participant_id", "expertise", "stage", "trial_id", "trial_type",
    "response", "expected", "correct", "reaction_time_ms",
    "source", "label", "is_real", "filename", "mask_filename",
    "transform", "domain", "position_order", "is_control", "control_kind",
]


@app.route("/api/submit", methods=["POST"])
def submit():
    payload = request.json or {}
    pid = payload.get("participant_id", "")
    expertise = payload.get("expertise", "")
    responses = payload.get("responses", [])
    gt_store = PLANS.get(pid, {})

    def _cell(v):
        if isinstance(v, (list, tuple)):
            return "|".join(map(str, v))
        return v

    rows = []
    for r in responses:
        tid = r.get("trial_id", "")
        gt = gt_store.get(tid, {})
        expected = gt.get("expected", "")
        resp = r.get("response", "")
        correct = ""
        if expected != "":
            correct = str(str(resp) == str(expected))
        rows.append([
            pid, expertise, gt.get("stage", r.get("stage", "")), tid, r.get("type", ""),
            json.dumps(resp) if isinstance(resp, (dict, list)) else resp,
            expected, correct, r.get("reaction_time_ms", ""),
            gt.get("source", ""), gt.get("label", ""), gt.get("is_real", ""),
            gt.get("filename", ""), gt.get("mask_filename", ""),
            gt.get("transform", ""), gt.get("domain", ""),
            _cell(gt.get("position_order", "")), gt.get("is_control", ""),
            gt.get("control_kind", ""),
        ])

    try:
        import gspread
        cred_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
        gc = gspread.service_account(filename=cred_path)
        sh = gc.open_by_key(os.environ.get("SPREADSHEET_ID"))
        ws = sh.sheet1
        if not ws.get_all_values():            # write header once on an empty sheet
            ws.append_row(SHEET_HEADER)
        ws.append_rows(rows)
        PLANS.pop(pid, None)
        return jsonify({"status": "success", "rows": len(rows)})
    except Exception as e:
        print(f"[submit] Google Sheets error: {e}")
        # Fallback: never lose data — append to a local CSV.
        try:
            import csv
            new = not os.path.exists("responses_backup.csv")
            with open("responses_backup.csv", "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(SHEET_HEADER)
                w.writerows(rows)
        except Exception as e2:
            print(f"[submit] CSV fallback error: {e2}")
            return jsonify({"status": "error", "message": str(e)}), 500
        return jsonify({"status": "success", "rows": len(rows), "sink": "csv_fallback"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
