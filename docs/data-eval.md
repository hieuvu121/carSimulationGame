# Dataset & Evaluation Plan

Part of the design docs — start at [DESIGN.md](DESIGN.md) for the index.

---

## 7. Dataset plan

Owner: **Shobita** (Dataset collection and preparation). Shobita runs the recording sessions, cleans the data, writes the quality report and produces the split. Everyone records.

### 7.1 Labels and what to record

| Label | How to perform it | Variations to include |
|---|---|---|
| `fist` | Closed fist, palm facing the camera, thumb across or beside the fingers | Thumb inside/outside, slight tilt ±20°, palm angled |
| `v_sign` | Index + middle extended and spread, others folded | Fingers close/wide, thumb visible/hidden, tilt ±20° |
| `open_palm` | All five fingers extended, palm facing the camera | Fingers together/spread, tilt ±20° |
| `none` | **Anything else**: relaxed half-open hand, pointing (1 finger), thumbs-up, OK sign, three fingers, hand moving between gestures, back of hand, partially visible hand at the frame edge | As varied as possible. This is what prevents false triggers. |

**How a recording burst works (`collect.py`, [Section 4.5](interfaces.md#45-dataset--classifier--srcdata-shobita-srcmodel-siri)):**
1. Press the label key once to start. After a 1 s "GET READY", recording runs.
2. Press the same key again, or Space, to stop. Recording also stops by itself at 80 samples.

While recording is on, the recorder should **keep moving the hand slightly**: small rotations, moving closer or further within the condition's band, moving left or right across the frame. That way the 80 samples are not 80 copies of one pose.

### 7.2 Conditions

The `condition` value is `"<lighting>_<distance>"`.

| Lighting | Definition |
|---|---|
| `bright` | Room lights on + daylight/desk lamp in front of the user |
| `dim` | Main lights off; only a monitor/one lamp. Face still visible but grainy. Expect the camera to drop to about 15 FPS. |
| `backlit` | Bright window or lamp **behind** the user (stress test) |

| Distance | Definition (hand to camera) |
|---|---|
| `near` | ≈ 40 cm (hand fills ~1/3 of the frame height) |
| `far` | ≈ 100 cm (hand ~1/8 of the frame height) |

- **Core conditions [Must], for every user:** `bright_near`, `bright_far`, `dim_near`, `dim_far`.
- **Stress conditions [Nice], at least 2 users:** `backlit_near`, `backlit_far`. Used only for robustness evaluation ([Section 8.4](#84-robustness-lighting-distance-users)).

### 7.3 Amounts

| Item | Value |
|---|---|
| Users | **[Must]** All 6 team members (`u01`–`u06`). **[Nice]** ≥ 2 outside volunteers (`u07`, `u08`…) with consent. They go to the **test** split only, as extra unseen users. |
| Samples per label per condition per user | `RECORD_TARGET_PER_LABEL = 80` (8 s of recording at 10 Hz) |
| Per user (core) | 4 labels × 4 conditions × 80 = **1,280 rows** (~6 min of recording) |
| Team total (core, 6 users) | **7,680 rows** (~1.9 k per label); after the split, about 5,120 train (4 users) + 2,560 test (2 users) |
| Hands | Use the dominant hand. In `bright_near`, record 40 of the 80 samples per label with the **other** hand, so mirroring is trained and tested. |

### 7.4 Files, naming, privacy

- **Path:** `data/raw/<user_id>_<condition>_<YYYYmmdd-HHMMSS>.csv`, for example `data/raw/u03_dim_far_20261014-161502.csv`.
  - There is one file per run of `collect.py`, and **one file = one recording session**.
  - Files are never edited by hand. A bad session is excluded in `splits.json` ([Section 7.5](#75-cleaning-quality-report-and-trainvalidationtest-split)), or deleted and recorded again.
- **Schema:** `CSV_COLUMNS` ([Section 4.5](interfaces.md#45-dataset--classifier--srcdata-shobita-srcmodel-siri)): header row, UTF-8, comma-separated, floats written with 6 decimal places.
- **Privacy:**
  - Only 63 landmark numbers + handedness are stored. **No images or video** are ever written.
  - `user_id` is anonymous. The ID ↔ name list is kept by the Lead off-repo.
  - Volunteers give verbal consent and can ask for their files to be deleted.
- **Sample data:**
  - Contents: `data/sample/` holds 4 small CSVs from 4 pseudo-users (`u91`, `u92`, `u93`, `u94`), with 40 rows per label each, plus `data/sample/splits.json` (test user `u94`, train users `u91`–`u93`). Four users are needed so that `split.py`'s "≥ 3 train users" rule holds.
  - Origin: Shobita produces it in Iteration 1 from real MediaPipe output, with a minimal script and Triet's help, before `collect.py` exists.
  - Purpose: Siri and Evan develop against it before real data exists.

### 7.5 Cleaning, quality report and train/validation/test split

**Cleaning (invalid-sample removal, DS.3) [Must].** `load_dataset()` applies these rules every time data is loaded. Raw files are never modified, and every dropped row is counted by reason:

| Reason key | Rule |
|---|---|
| `bad_label` | `label` not in `LABELS` |
| `non_finite` | any `f0…f62` is NaN/inf, or a value is not a number |
| `bad_handedness` | `handedness` not in `{"Left", "Right"}` |
| `degenerate` | `preprocess()` raises (for example palm size ≈ 0) |
| `out_of_frame` | for `fist`/`v_sign`/`open_palm` only: more than `MAX_OUT_OF_FRAME_LANDMARKS` (5) landmarks have x or y outside `[-OUT_OF_FRAME_MARGIN, 1 + OUT_OF_FRAME_MARGIN]`. MediaPipe is guessing those points. Such rows are kept for `none`, where partly visible hands are intended. |
| `duplicate` | identical `f0…f62` to an earlier row in the same session |

**Whole sessions:** if a session is wrong (for example recorded under the wrong condition, or a teammate reports pressing the wrong label key), Shobita adds it to `excluded_files` in `splits.json` with a reason. Excluded files are in no split.

**Quality report (DS.4) [Must].** `python -m src.data.dataset --report` writes `results/dataset_report.txt`:
- rows read/kept and dropped rows per reason;
- counts per user × condition × label, flagging cells below `MIN_ROWS_PER_CELL` (60) for re-recording;
- label balance overall and per split;
- excluded files with reasons.

**[Nice]:** list rows whose preprocessed features are more than 3 standard deviations from their label's per-user mean, for manual review of possible mislabels.

**Train/validation/test split (DS.5) [Must] — user-based, never random:**

- **Test set:** `N_TEST_USERS = 2` users are held out **entirely**: all their sessions, all conditions.
  - They are chosen once, at random with `SEED` (or named explicitly with `--test`), **before Siri starts tuning**.
  - Outside volunteers, if any, are added to the test set.
  - The test set measures generalisation to **new people** (R10, "users").
- **Train set:** all remaining users (4 of the 6 team members).
- **Validation:** with only 4 train users, a fixed single validation user would give very noisy results. So validation is **leave-one-user-out (LOUO) over the train users**: each train user is the validation user once, with the model trained on the other 3. LOUO is used to choose the SVM hyperparameters and the confidence threshold.
- **Leakage rules:**
  - No user appears in two splits. Since every session belongs to exactly one user, no session can appear in two splits either.
  - `make_splits`/`load_splits` enforce both and raise `ValueError` otherwise.
  - `evaluate.py` also refuses a model whose `trained_on_users` overlaps the test users.
- **Frozen:** once committed, `data/splits.json` is not changed. The only exception is adding new sessions of users already in a split, or new volunteers to `test`. Any other change needs Lead approval, plus re-running tuning and evaluation.

`data/splits.json` format (written by `split.py`, read by Siri via `load_splits`/`load_split`):

```json
{
  "version": 1,
  "created_utc": "2026-10-16T09:00:00Z",
  "seed": 218,
  "data_dir": "data/raw",
  "test_users": ["u02", "u05"],
  "train_users": ["u01", "u03", "u04", "u06"],
  "validation": "leave_one_user_out",
  "files": {
    "train": ["u01_bright_near_20261014-161502.csv", "u01_bright_far_20261014-162210.csv"],
    "test":  ["u02_bright_near_20261014-170210.csv", "u02_dim_far_20261014-183005.csv"]
  },
  "excluded_files": {"u04_dim_far_20261014-180001.csv": "recorded under bright light by mistake"}
}
```

---

## 8. Evaluation plan

**Scope for 8 days.** Every item is tagged:
- **[Must]** items are required for the presentation: playable by gestures, test-set accuracy + confusion matrix, latency median/p95, lighting and distance comparison.
- **[Nice]** items are done only after every [Must] item is finished: backlit condition, outside volunteers, 240 FPS video check, per-hand accuracy, latency stage breakdown, and the other extras marked below.

All outputs go to `results/` in the formats listed in [Section 8.5](#85-who-measures-what-and-output-formats), so they can go straight into the slides.

### 8.1 Gesture accuracy (offline, Siri)

**Validation (on train users only) — hyperparameters and threshold:**
- **[Must]** Grid `SVM_GRID_C × SVM_GRID_GAMMA` (9 combinations), each scored by mean macro-F1 over leave-one-user-out folds on the **train users** (`sklearn.model_selection.LeaveOneGroupOut`, `groups = user_id`). The best pair is written to `config.py`, and the per-fold scores to `results/validation_scores.csv`.
- **[Must]** Choose `CONFIDENCE_THRESHOLD` from the out-of-fold validation predictions ([Section 6.4](game-logic.md#64-confidence-threshold-siri-applied-inside-gestureclassifierpredict)).
- **[Must]** Train the final model on **all train users** with the chosen parameters and save it. **This is the model the game ships with.** It is never retrained on test users, so the reported test numbers describe the real model.

**Test (on the held-out test users) — run once:**
- **[Must]** Overall accuracy and macro-F1 (raw, `threshold = 0`), per-class precision/recall, and accuracy per test user. That is two numbers, which come free with the same predictions and cover "different users" (R10).
- **[Must]** Confusion matrix: 4×4, row-normalised. Both raw counts and percentages are saved.
- **[Must]** Commonly confused gestures (CL.3): the top 3 off-diagonal cells, each with a short explanation. For example, fist vs V sign is expected when the two fingers are not clearly spread or are seen edge-on.
- **[Nice]** With the threshold applied: rejection rate and accuracy on accepted samples.
- **[Nice]** Example landmark plots (no images) of misclassified samples.
- **Rule:** the test set is evaluated **after** the model is final. If anything is changed after seeing test results, the slides say so explicitly.

### 8.2 Live action accuracy (Evan) [Nice]

Offline accuracy is per frame. Players experience **actions**. Each tester does a scripted sequence in the running game (`--log-latency` on). An observer reads the next command from a printed sheet and ticks whether the correct action happened within 1 s.

- **Script:** 30 commands per run: 10 × left, 10 × right, 5 × pause, 5 × resume-by-steer, in a fixed shuffled order.
- **Also counted:** false triggers during a 30 s "free hand movement, no gestures" segment.
- **Metric:** action success rate = correct / 30; false triggers per minute.

The [Must] "playable by gestures" goal is checked separately, by the Iteration 4 exit criteria ([Section 11](workflow.md#11-development-plan-5-iterations)). This scripted measurement is the [Nice] quantitative version.

### 8.3 Response time (Evan)

All timestamps come from `time.perf_counter()`, one process-wide clock shared by the camera, worker and main threads.

| Metric | Definition | What it captures | Tag |
|---|---|---|---|
| **Response time** (headline) | `t_action − t_onset`. `t_onset` = `Frame.t_capture` of the **first frame of the uninterrupted run of raw classifier outputs equal to the triggering label** (the streak that ended in the debounced event). `t_action` = `perf_counter()` on the main thread immediately after the `GameController` method returned `True` (or after `pause()` for pause events). | Everything from the moment the system first *saw* the gesture to the car moving: MediaPipe + preprocess + SVM + smoothing window + debounce + queue wait to the next render frame + game call | **[Must]** median and p95 |
| Processing latency | `t_action − t_trigger`. `t_trigger` = `Frame.t_capture` of the frame whose processing emitted the action. | Compute cost of one pass through the pipeline, plus up to one render frame (≈ 17 ms at 60 FPS) of queue wait | [Nice] |
| Stage breakdown | `t_landmarks − t_trigger`, `t_predict − t_landmarks`, `t_action − t_predict` | Where the time goes | [Nice] |

- **Excluded, and stated in the slides:** sensor exposure and USB/driver buffering before `cv2.read()` returns (typically 1–2 frames) and the display refresh after `t_action`.
- **[Must]** At least 50 actions per lighting condition (`bright`, `dim`), reported as median and p95 per condition.
- **[Nice]** 240 FPS cross-check: film the screen and hand with a phone slow-motion video for 10 trials, and compare the hand-movement-to-car-movement time with the logged response time.
- **[Nice]** Camera FPS and worker FPS: rolling mean over 1 s, shown on the HUD and in the summary.
- **Targets:**
  - response time median ≤ 250 ms, p95 ≤ 400 ms;
  - processing latency median ≤ 40 ms;
  - worker ≥ 15 FPS in `dim`.

### 8.4 Robustness (lighting, distance, users)

| Factor | Offline (Siri, test-set predictions) | Live (Evan, [Section 8.2](#82-live-action-accuracy-evan-nice) script) |
|---|---|---|
| Lighting | **[Must]** test accuracy `bright` vs `dim`. **[Nice]** `backlit` rows (never trained on: a pure stress test). | [Nice] script in `bright`, `dim`, `backlit`. **[Must]** latency per lighting ([Section 8.3](#83-response-time-evan)). |
| Distance | **[Must]** test accuracy `near` vs `far` | [Nice] script at ≈ 40 cm and ≈ 100 cm |
| Users | **[Must]** accuracy per test user, plus the spread of LOUO validation scores over the train users | [Nice] outside volunteers (`u07+`, test-only users) |
| Hands | [Nice] accuracy on left-hand vs right-hand test rows | – |
| Detection | [Nice] MediaPipe detection rate per lighting (frames with a hand / all frames while the hand is in view), measured by Triet | [Nice] unintended auto-pauses per run |

Live test conditions (lamp position, distance marks on the desk) are set up by Shobita, using the same definitions as [Section 7.2](#72-conditions).

### 8.5 Who measures what, and output formats

| Output file | Content | Owner | Tag | Slide use |
|---|---|---|---|---|
| `results/dataset_report.txt` | rows kept/dropped per reason, counts per user × condition × label, split sizes | Shobita | [Must] | Table |
| `data/splits.json` | user-based split ([Section 7.5](#75-cleaning-quality-report-and-trainvalidationtest-split)) | Shobita | [Must] | One line: "train 4 users / test 2 users" |
| `results/validation_scores.csv` | per fold × parameter macro-F1 (LOUO over train users) | Siri | [Must] | – |
| `results/classification_report.txt` | sklearn classification report on the **test users** (raw) | Siri | [Must] | Table |
| `results/confusion_matrix.png` + `.csv` | 4×4 row-normalised on the test users, counts in CSV | Siri | [Must] | Figure |
| `results/per_condition_accuracy.csv` + `.png` | `condition, lighting, distance, n, accuracy` on the test users | Siri | [Must] | Grouped bars |
| `results/per_user_accuracy.csv` | `user_id, split (test/validation), n, accuracy, macro_f1` | Siri | [Must] | Table |
| `results/threshold_curve.png` | accepted-accuracy & rejection vs threshold (validation predictions) | Siri | [Nice] | Figure (justifies 0.70) |
| `results/per_hand_accuracy.csv` | `handedness, n, accuracy` on the test users | Siri | [Nice] | – |
| `results/detection_rate.csv` | `lighting, frames, frames_with_hand, rate` | Triet | [Nice] | – |
| `results/latency_<ts>.csv` | One row per `ActionTiming` + derived `response_ms, processing_ms` | Evan | [Must] | – |
| `results/latency_summary.csv` + `latency_hist.png` | median / p95 (+ mean / max) of response time per lighting condition | Evan | [Must] (stage columns [Nice]) | Table + histogram |
| `results/live_trials.csv` | `tester, lighting, distance, command, expected, observed, success` | Evan (Shobita sets conditions) | [Nice] | Table |
| `docs/notes/{triet,uy,shobita,siri,evan}.md` | Technical notes | each person | [Must] | Text |
| Slides | Assembled from the above | Lead | [Must] | – |

All charts: matplotlib, 150 dpi PNG, readable axis labels, titles that state the protocol. For example: "Confusion matrix — 2 held-out test users, 2,560 samples; model trained on 4 other users".
