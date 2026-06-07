# ADEM — Synthetic Remote-Sensing Perception Study

A four-stage human-perception questionnaire (Flask) for evaluating synthetic
remote-sensing imagery, augmentation transforms, mask quality, and conditional
generation. Deployable on Render; logs to Google Sheets with a local-CSV fallback.

## Stages
1. **Transforms** — same/different identity under 7 augmentations (baseline, noise,
   perspective, rotation, flip, resizedcrop, combined) + different-image controls → d′.
2. **Mask check** — image–mask correctness, half real / half synth, with swapped-mask controls.
3. **Generators** — single best pick among 3 generators for one conditioning mask
   (positions randomised per trial).
4. **Realism** — 1 (completely synthetic) … 5 (completely real) Likert across every source
   → treat as graded detector for ROC/AUC per source.

Each stage carries instructed-response attention checks. Ground truth is held
server-side and never sent to the participant.

## Run locally
```bash
pip install -r requirements.txt
python app.py            # http://localhost:5000
```
Open with no dataset present and the frontend still runs in **preview mode** with
placeholder imagery (handy for testing the flow).

## Expected folder layout (all paths overridable via env vars)
```
samplednoref/ARAS400k_real/        # Stage 1 source (transforms applied on the fly)
real/images, real/masks            # Stage 2 (same filenames in img & mask)
synth/images, synth/masks          # Stage 2
sampled/conditioning_images/       # Stage 3 masks
sampled/ARAS-CSD-1E7|ARAS-CSD-2|ARAS-C-UNetGAN/   # Stage 3 generators (same filenames)
sampled/ARAS400k-cnet/             # Stage 4
samplednoref/ARAS400k_synth, diverse, BELDECN, BELDEK   # Stage 4
```

## Config (environment variables)
| Var | Default | Meaning |
|---|---|---|
| `S1_N` | 2 | Stage-1 base images (×7 transforms each) |
| `S1_CTRL` | 6 | Stage-1 different-image control pairs |
| `S2_M` | 10 | Stage-2 image–mask pairs |
| `S2_WRONG` | 0.5 | Fraction of Stage-2 pairs given a swapped (wrong) mask |
| `S3_K` | 10 | Stage-3 mask comparisons |
| `S4_PER` | 1 | Stage-4 images sampled per source folder |
| `ATTN` | 1 | Attention checks per applicable stage |
| `S1_SRC`, `S2_*`, `S3_*`, `S4_*` | see `app.py` | Folder paths |
| `DATA_ROOT` | `.` | Root all dataset paths resolve under |
| `SPREADSHEET_ID` | — | Google Sheet ID for results |
| `GOOGLE_CREDENTIALS_PATH` | `google_credentials.json` | Service-account JSON |

## Render deployment
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`
- Add a Secret File with your service-account JSON and set `GOOGLE_CREDENTIALS_PATH` to it,
  plus `SPREADSHEET_ID`. Share the sheet with the service-account email.
- Note: plan ground-truth is held in memory; a dyno restart mid-session still logs the
  participant's responses (without re-attached ground truth) and never loses data.

## Logged columns (one row per trial)
`participant_id, expertise, stage, trial_id, trial_type, response, expected, correct,
reaction_time_ms, source, label, is_real, filename, mask_filename, transform, domain,
position_order, is_control, control_kind`

Ground truth is flattened into explicit columns (only the ones relevant to a stage are
filled). `correct` is computed against `expected` where one exists.

