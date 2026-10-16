# Gesture Car Game: Presentation Fact Sheet

Extracted from the repo at `main` @ `bad9e61` on 2026-10-15. Every number cites its source.
"computed" = output of `presentation_tools/compute_facts.py` (saved in `presentation_tools/output/facts.json`).
Regenerate: `python -m presentation_tools.compute_facts && python -m presentation_tools.make_figures`.

---

## A. Overview & code structure

### Tech stack
| Component | Version | Source |
|---|---|---|
| Python | 3.12 (64-bit only); local venv 3.12.14 | `requirements.txt` line 1; `.venv` |
| OpenCV | opencv-contrib-python 4.11.0.86 | `requirements.txt` |
| MediaPipe | 0.10.21 | `requirements.txt` |
| scikit-learn | 1.6.1 | `requirements.txt` |
| Game library | pygame 2.6.1 | `requirements.txt` |
| Others | numpy 1.26.4, scipy 1.17.1, joblib 1.4.2, matplotlib 3.11.2, protobuf 4.25.9, pytest 8.3.5 | `requirements.txt` |
| Lock file verified on | macOS arm64 only. Windows marked "PENDING" | `requirements.lock` header |

### Folder tree (2 levels)
| Path | Role | Owner |
|---|---|---|
| `src/config.py` | All tunable constants in one place | Hieu (Lead) |
| `src/capture/camera.py` | Threaded OpenCV capture, mirroring, FPS measurement | Triet |
| `src/capture/landmarks.py` | `HandTracker`: MediaPipe Hands wrapper → `HandLandmarks` | Triet |
| `src/capture/preprocess.py` | `preprocess()`: shared 63-feature function (train + live) | Triet |
| `src/capture/check/*.py` | Webcam/FPS/preview check scripts | Triet |
| `src/data/collect.py` | Recording tool (key-labelled CSV recorder) | Shobita |
| `src/data/dataset.py` | `load_dataset()`: cleaning rules + quality report | Shobita |
| `src/data/split.py` | User-based train/test split → `data/splits.json` | Shobita |
| `src/model/train.py` | Scaler + SVM, LOUO grid search, saves model | Siri |
| `src/model/evaluate.py` | Test-set accuracy, confusion matrix, report | Siri |
| `src/model/predict.py` | `GestureClassifier`: loads model, `predict()` | Siri |
| `src/game/entities.py` | `Car`, `Obstacle`, AABB `collides()` | Uy |
| `src/game/gameplay.py` | Lanes, spawning, speed ramp, score | Uy |
| `src/game/states.py` | `GameState` enum + `StateMachine` | Uy |
| `src/game/controller.py` | `GameController`: the only API integration uses | Uy |
| `src/game/render.py`, `ui.py` | Road/car drawing; HUD + overlays | Uy |
| `src/game/keyboard_demo.py` | Game playable with arrow keys only | Uy |
| `src/app/main.py` | Live entry point (Pygame loop) | Evan |
| `src/app/worker.py` | `GestureWorker` thread: frame → landmarks → predict → smooth | Evan |
| `src/app/smoothing.py` | `GestureSmoother` + `Debouncer` | Evan |
| `src/app/actions.py` | `gesture_to_action()` mapping + `apply_action()` | Evan |
| `src/app/model_adapter.py` | `LiveGestureClassifier`: adds confidence + threshold | Evan |
| `src/app/latency.py` | `LatencyLogger` → `results/latency_*.csv` | Evan |
| `src/app/mock_classifier.py` | Mock camera/tracker/classifier driven by keys | Evan + Lead |
| `tests/` (19 files) | Unit + integration tests, headless | all; review Hieu |
| `data/raw/` (28 CSVs), `data/splits.json` | Recorded landmarks + frozen split | Shobita |
| `models/classifier.pkl` | Trained Pipeline (417,105 bytes) | Siri |
| `results/` | Dataset report, FPS logs, Triet test XML | various |
| `docs/` | DESIGN, interfaces, game-logic, data-eval, workflow, notes | all |

### Commands (run from repo root; source: `README.md` "How to run")
| Task | Command |
|---|---|
| Record data | `python -m src.data.collect --user u03 --condition bright_near` |
| Cleaning report | `python -m src.data.dataset --report` |
| Split | `python -m src.data.split --test u02 u05` (example in README) |
| Train | `python -m src.model.train` |
| Evaluate | `python -m src.model.evaluate` |
| Keyboard-only game | `python -m src.game.keyboard_demo` (add `--controller` for states/HUD) |
| Full gesture game | `python -m src.app.main` (real webcam + SVM) |
| No-webcam dev mode | `python -m src.app.main --classifier mock --camera mock` |
| Latency logging | `python -m src.app.main --log-latency` |
| Tests | `pytest` |

### Interfaces (for the architecture diagram)
| Caller | Callee | Input → Output | Source |
|---|---|---|---|
| `GestureWorker` (worker thread) | `Camera.read()` | → `Frame(image, index, t_capture)` | `src/capture/camera.py` |
| `GestureWorker` | `HandTracker.detect(image_bgr)` | BGR image → `HandLandmarks(points 21×3, handedness, score)` or `None` | `src/capture/landmarks.py` |
| `GestureWorker`, `load_dataset()` | `preprocess(landmarks)` | `HandLandmarks` → `float32[63]` | `src/capture/preprocess.py` |
| `GestureWorker` | `LiveGestureClassifier.predict(features)` | `[63]` → `(label, confidence)` | `src/app/model_adapter.py` |
| `LiveGestureClassifier` | `GestureClassifier` / sklearn `predict_proba` | `[1,63]` → 4 probabilities | `src/model/predict.py` |
| `GestureWorker` | `GestureSmoother.update(token, t)` → `Debouncer.update(stable, t)` | token → stable token → event or `None` | `src/app/smoothing.py` |
| `main.py` (main thread) | `worker.poll_events()` | → `list[GestureEvent]` (thread-safe queue) | `src/app/worker.py` |
| `main.py` | `gesture_to_action(token, state)` → `apply_action(controller, action)` | → `bool` (changed?) | `src/app/actions.py` |
| `apply_action` | `GameController.move_left/move_right/pause/resume/restart` | → `bool` / `None` | `src/game/controller.py` |
| `main.py` | `controller.update(dt)`, `render(surface)`, `set_gesture_info(label, conf, hand)` | per frame | `src/game/controller.py` |
| `train.py` / `evaluate.py` | `load_dataset()`, `load_splits()` | CSVs → `Dataset(X, y, groups, …)` | `src/data/dataset.py`, `split.py` |

Threads: camera capture thread, gesture worker thread, Pygame main thread (`docs/DESIGN.md` §2.1, "three threads").

### Third-party code, tutorials, pretrained models
| What | How we use it / what we changed | Source |
|---|---|---|
| MediaPipe Hands (pretrained palm detector + 21-landmark model) | Used as-is through `mp.solutions.hands.Hands`. We added a wrapper, validation, and a single-hand `HandLandmarks` output | `src/capture/landmarks.py` `HandTracker.__init__/detect` |
| scikit-learn `SVC`, `StandardScaler`, `GridSearchCV`, `LeaveOneGroupOut` | Standard library usage. Our code is the pipeline, grid and LOUO grouping by user | `src/model/train.py` |
| thinkall/hand-gesture-recognition-mediapipe | Idea only: number keys record keypoints to CSV. We added start/stop bursts, 1 s delay, 10 Hz cap, auto-stop at 80, undo | `docs/notes/shobita.md` References |
| anantSinghCross/realtime-hand-gesture-recognition | Compared against: it saves images; we save landmarks | `docs/notes/shobita.md` |
| haneure/Hand-Tracking-Capture; gabang2 gist | Listed as references for landmark CSV capture / gesture rules | `docs/notes/shobita.md` |
| Any tutorial used by Triet, Uy, Siri, Evan | NOT FOUND – ask Triet, Uy, Siri, Evan | (no references in their notes or code) |

---

## B. Hand tracking & features (Triet)

### Settings
| Setting | Value | Source |
|---|---|---|
| `static_image_mode` | `False` (video/tracking mode) | `src/capture/landmarks.py` `HandTracker.__init__` |
| `max_num_hands` | 1 | `src/config.py` `MP_MAX_NUM_HANDS` |
| `model_complexity` | 0 (fastest) | `src/config.py` `MP_MODEL_COMPLEXITY` |
| `min_detection_confidence` | 0.6 | `src/config.py` |
| `min_tracking_confidence` | 0.5 | `src/config.py` |
| Resolution | 640 × 480 px | `src/config.py` `FRAME_WIDTH/HEIGHT` |
| Requested FPS | 30 | `src/config.py` `CAMERA_FPS` |
| Measured FPS | bright 30.07 (301 frames / 10.01 s); dim 29.94 (300 / 10.02 s) | `results/triet-fps-bright.txt`, `triet-fps-dim.txt` |
| Mirroring | On (`cv2.flip(image, 1)`), selfie view | `src/config.py` `MIRROR_FRAME`; `camera.py` `_capture_frame` |
| Capture | Background thread, keeps only newest frame | `src/config.py` `CAMERA_THREADED`; `docs/notes/triet.md` |

Note: `config.py` says FPS is "≈15 in dim light", but Triet measured 29.94 FPS in dim light.

### Landmarks → features (`src/capture/preprocess.py` `preprocess()`)
1. **Validate:** shape (21, 3), finite values, handedness ∈ {Left, Right}.
2. **Aspect correction:** x and z × (640/480), so x and y share one scale.
3. **Wrist-relative:** subtract landmark 0 (wrist), so the wrist becomes (0, 0, 0).
4. **Left → right mirroring:** if handedness is "Left", negate x.
5. **Scale normalisation:** divide by the x-y distance wrist → middle-finger MCP (landmark 9). The function rejects the frame if that distance is < 1e-6.
6. **Flatten:** 21 × 3 → **63 features** (x, y, z all kept; z is scaled too). Rotation is deliberately kept.

**Why:** subtracting the wrist removes hand **position**, dividing by palm length removes hand **size/distance**, and mirroring left hands removes the hand **side**. One function serves both training and live play, so the features always match.

*Shared feature function: core of `preprocess()` (`src/capture/preprocess.py`)*
```python
points = np.array(landmarks.points, dtype=np.float64, copy=True)
points[:, [0, 2]] *= FRAME_WIDTH / FRAME_HEIGHT     # aspect-correct x and z
points -= points[WRIST_IDX].copy()                  # wrist-relative
if landmarks.handedness == "Left":
    points[:, 0] *= -1                              # mirror left hand
palm = float(np.linalg.norm(points[MIDDLE_MCP_IDX, :2]))
if palm < MIN_PALM_SIZE:
    raise ValueError("Degenerate hand: wrist and middle-finger MCP are too close.")
points /= palm                                      # scale-invariant
features = points.reshape(NUM_FEATURES).astype(np.float32)   # 63 values
```

---

## C. Gestures

| Class | Hand pose | Game command | Source |
|---|---|---|---|
| `fist` | Closed fist, palm to camera | PLAYING/COUNTDOWN: move left. PAUSED: resume + move left | `docs/data-eval.md` §7.1; `src/app/actions.py` `_MAPPING` |
| `v_sign` | Index + middle extended and spread, others folded | Move right. PAUSED: resume + move right | same |
| `open_palm` | All five fingers extended, palm to camera | PLAYING/COUNTDOWN: pause. GAME_OVER: restart | same |
| `none` (neutral class) | Anything else: relaxed hand, pointing, thumbs-up, OK, three fingers, back of hand, partly visible | No action, in any state | same |
| `no_hand` (pipeline token, not a class) | MediaPipe finds no hand | PLAYING/COUNTDOWN: pause | `src/config.py` `NO_HAND`; `actions.py` |

A prediction with confidence below 0.70 becomes `none` (`src/app/model_adapter.py` `LiveGestureClassifier.predict`).

---

## D. Dataset (Shobita)

### Recording tool (`src/data/collect.py`)
- **Controls:** `1` fist, `2` V sign, `3` open palm, `0` none. The same key or Space stops, Backspace undoes the last burst, Q quits (module docstring, `KEY_LABELS`).
- **Timing:** a 1.0 s "GET READY" delay, then 10 rows/s, auto-stopping at 80 rows per label (`config.py` `RECORD_START_DELAY_S`, `RECORD_SAMPLE_HZ`, `RECORD_TARGET_PER_LABEL`).
- **Saved per sample:** `user_id, condition, label, f0…f62` (**raw** x, y, z of 21 landmarks), `handedness`. No images are saved (`landmarks_to_row`; `config.py` `CSV_COLUMNS`).
- **File format:** CSV, one file per user × condition session, named `u03_dim_far_YYYYMMDD-HHMMSS.csv` (`dataset.py` `parse_file_name`).
- **Metadata:** participant ID, condition (lighting_distance), label, handedness. The session is the file name.

### Participants & conditions
| Item | Value | Source |
|---|---|---|
| Participants | 7 (u01–u07). u07 is an outside volunteer, used for test only | computed; `docs/notes/shobita.md` |
| Sessions (files) | 28 = 7 users × 4 conditions | computed |
| Conditions | `bright_near`, `bright_far`, `dim_near`, `dim_far`. No `backlit` recorded | computed; `shobita.md` |
| Bright | Room lights + daylight or desk lamp in front | `shobita.md` "Recording conditions" |
| Dim | Main lights off, only monitor or one lamp | same |
| Near / far | ≈ 40 cm / ≈ 100 cm hand-to-camera | same |
| Hand used | Right 6,751 rows, Left 2,209 rows (MediaPipe label). u02 mostly left (1,235); u03 about half (646 L) | computed `handedness_per_user` |
| Background | NOT FOUND – ask Shobita | – |
| Camera | All recorded with one laptop webcam | `shobita.md` §6 |

### Cleaning (`src/data/dataset.py` `check_row`)
| Rule | Removes | Rows removed |
|---|---|---|
| `bad_label` | Label not one of the 4 classes | 0 |
| `non_finite` | Non-numeric / NaN / inf / wrong column count | 0 |
| `bad_handedness` | Handedness not Left/Right | 0 |
| `degenerate` | `preprocess()` fails (e.g. palm length ≈ 0) | 0 |
| `out_of_frame` | > 5 landmarks outside [-0.05, 1.05] (gestures only, not `none`) | 0 |
| `duplicate` | Identical row within the same session | 0 |
| **Total** | 8,960 read → **8,960 kept** | `results/dataset_report.txt`; computed |

Before cleaning, Shobita deleted one incomplete false-start file manually (u06 dim_far, 32 fist rows) (`shobita.md` §4). This happened outside the cleaning code.

### Split: by **person** (`data/splits.json`, `src/data/split.py`)
| Split | Users | fist | v_sign | open_palm | none | Total |
|---|---|---|---|---|---|---|
| Train (LOUO validation inside) | u01, u02, u03, u04 | 1,280 | 1,280 | 1,280 | 1,280 | 5,120 |
| Validation | One train user held out per fold (leave-one-user-out) | 320 per fold | 320 | 320 | 320 | 1,280 / fold |
| Test | u05, u06, u07 | 960 | 960 | 960 | 960 | 3,840 |

Sources: `results/dataset_report.txt`; computed. Validation per fold = 80 rows × 4 conditions per user.
Note: `config.py` `N_TEST_USERS = 2`, but the frozen split uses **3** test users.

---

## E. Classifier (Siri)

### Pipeline & search (`src/model/train.py`)
| Item | Value |
|---|---|
| Pipeline | `StandardScaler` → `SVC(probability=True, random_state=42)` (`build_pipeline`) |
| Search space | `C ∈ {0.1, 1, 10}` × `kernel ∈ {linear, rbf}`, gamma = sklearn default `'scale'` (`train_model` `param_grid`) |
| Search method | `GridSearchCV`, 6 combinations |
| CV scheme | `LeaveOneGroupOut` grouped by user, so 4 folds over train users |
| Best | **C = 10, kernel = rbf, gamma = 'scale'**. Saved model confirms this (computed `model`) |
| Best CV accuracy | **0.867** (computed re-run; matches `docs/notes/siri.md` "86.7%") |
| Support vectors | 743 (computed) |

Note: `config.py` defines `SVM_GRID_GAMMA = ("scale", 0.01, 0.1)`, but `train.py` never uses it. **Gamma was not tuned**, so a C × gamma heatmap is not possible. We plot C × kernel instead.

### Validation score per combination (computed by re-running the exact grid from `train.py`)
| Kernel | C | Mean LOUO acc | Std | u01 out | u02 out | u03 out | u04 out |
|---|---|---|---|---|---|---|---|
| linear | 0.1 | 0.862 | 0.069 | 0.791 | 0.812 | 0.966 | 0.880 |
| rbf | 0.1 | 0.828 | 0.051 | 0.791 | 0.802 | 0.915 | 0.802 |
| linear | 1 | 0.865 | 0.064 | 0.788 | 0.821 | 0.948 | 0.904 |
| rbf | 1 | 0.862 | 0.065 | 0.784 | 0.826 | 0.958 | 0.881 |
| linear | 10 | 0.859 | 0.056 | 0.784 | 0.834 | 0.936 | 0.882 |
| **rbf** | **10** | **0.867** | 0.068 | 0.782 | 0.829 | 0.963 | 0.894 |

Takeaway: every combination scores within 0.828–0.867, and the spread across held-out users (0.78–0.97) is much larger than the spread across hyperparameters.

### Final TEST set (users u05, u06, u07; 3,840 rows) (computed; matches `img/confusion_matrix.png`)
| Metric | Value |
|---|---|
| Accuracy | **85.44 %** |
| Macro precision / recall / F1 | 0.865 / 0.854 / 0.843 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| fist | 0.879 | 0.973 | 0.923 | 960 |
| v_sign | 0.965 | 0.969 | 0.967 | 960 |
| open_palm | 0.738 | 0.963 | 0.835 | 960 |
| none | 0.879 | 0.514 | 0.648 | 960 |

Confusion matrix (rows = true, cols = predicted):
| true \ pred | fist | v_sign | open_palm | none |
|---|---|---|---|---|
| fist | 934 | 1 | 1 | 24 |
| v_sign | 12 | 930 | 8 | 10 |
| open_palm | 1 | 1 | 924 | 34 |
| none | 116 | 32 | 319 | 493 |

### Breakdowns (computed)
| Group | Accuracy |
|---|---|
| u05 / u06 / u07 | 83.4 % / 86.2 % / 86.8 % |
| bright_near / bright_far / dim_near / dim_far | 84.9 % / 86.0 % / 87.8 % / 83.0 % |
| bright vs dim | 85.5 % vs 85.4 % |
| near vs far | 86.4 % vs 84.5 % |
| Right hand (n = 3,664) vs **Left hand (n = 176)** | 86.6 % vs **61.9 %** |

### Most confused pairs (computed `top_confusions`)
| True → predicted | Count | Likely reason |
|---|---|---|
| none → open_palm | 319 | `none` includes relaxed, half-open hands with extended fingers, which look like a palm (`siri.md` §5; `data-eval.md` §7.1) |
| none → fist | 116 | `none` includes pointing/thumbs-up/curled poses whose folded fingers look like a fist (inferred from the `none` pose list) |
| open_palm → none | 34 | Same boundary in reverse: a palm with fingers together or tilted looks like a relaxed hand (inferred) |

Fist ↔ V sign was the risk the team expected (`docs/risks.md` #2), but only 13 errors occurred in that pair.

### Other models compared
NOT FOUND – ask Siri. The repo contains only the SVM (linear and RBF kernels inside the grid). `siri.md` mentions NNs and Random Forests as "alternatives considered", with no scores.

### Confidence, timing, model file
| Item | Value | Source |
|---|---|---|
| Confidence | Platt-scaled SVM probabilities (`probability=True`). Confidence = max class probability. Below **0.70** → `none` | `train.py`; `model_adapter.py` `predict` |
| Threshold justification (≤5 % false triggers on validation `none`) | NOT FOUND – ask Siri | required by `docs/workflow.md` Iteration 3 |
| Effect of threshold on test | Accuracy 85.4 % → **88.2 %**; 9.0 % of frames rejected; `none` recall 0.51 → 0.67; none→palm still 255 | computed `live_*` |
| Mean confidence, correct vs wrong | 0.959 vs 0.784 | computed |
| Inference time per prediction | `predict_proba` median **0.160 ms**, p95 0.205 ms (n = 1,000, this Mac) | computed `timing` |
| Saved model | `models/classifier.pkl`, joblib-dumped sklearn Pipeline, **417,105 bytes (≈ 407 KB)** | computed |

---

## F. Car game (Uy)

| Item | Value | Source |
|---|---|---|
| Window | 480 × 720 px, 48 px HUD, rendered at 60 FPS | `config.py` |
| Lanes | 3 lanes × 160 px; centres at x = 80 / 240 / 400 | `config.py`; `entities.py` `lane_center_x` |
| Lane change | Instant jump of one lane. Blocked at edges (returns `False`) | `gameplay.py` `move_left/move_right` |
| Car / obstacle size | 70 × 110 px / 90 × 70 px | `config.py` |
| Spawning | **Distance-based**: a new row every 300 px of travel, with 1 obstacle (75 %) or 2 (25 %) in random distinct lanes, never all 3 | `gameplay.py` `_maybe_spawn_row/_spawn_row`; `DOUBLE_OBSTACLE_PROB = 0.25` |
| Speed | 220 px/s + 6 px/s per second of play, capped at 650 px/s, so the cap is reached after ≈ 72 s | `gameplay.py` `update`; `config.py` |
| Collision | Axis-aligned bounding boxes, strict overlap (touching edges do not count); `dt` clamped to 0.1 s so nothing tunnels | `entities.py` `collides`; `config.py` `MAX_DT` |
| Score | 10 points per second of active play (`int` shown) | `config.py` `SCORE_PER_SECOND`; `gameplay.py` |
| Restart | New obstacle pattern (the RNG is not reseeded) | `gameplay.py` `reset` |

### States (`src/game/states.py`)
| From | Event | To |
|---|---|---|
| (start) | — | PAUSED |
| PAUSED | `resume()` | COUNTDOWN (2.0 s; obstacles frozen, steering allowed) |
| COUNTDOWN | timer ends | PLAYING with **0.75 s grace** (no collisions) |
| PLAYING / COUNTDOWN | `pause()` | PAUSED |
| PLAYING | collision (after grace) | GAME_OVER |
| any | `restart()` | COUNTDOWN (round reset) |

### Keyboard controls
| Program | Keys | Source |
|---|---|---|
| `keyboard_demo` | ← / → steer, Esc quit; with `--controller`: P pause/resume, R restart; 1/2/3/0/H feed fake HUD gesture | `keyboard_demo.py` |
| `app.main` | ← / → steer, P pause/resume, R restart, Esc quit; mock mode: hold 1/2/3/H | `main.py` `_keyboard`, `_status` |

### Control API for gesture input (`src/game/controller.py` `GameController`)
`move_left() -> bool`, `move_right() -> bool`, `pause()`, `resume()`, `restart()`, `is_paused()`, `state`, `score`, `update(dt)`, `render(surface, fps=None)`, `set_gesture_info(label, confidence, hand_present)`. Steering works only in PLAYING/COUNTDOWN. The return value tells the latency logger whether the action had a visible effect.

---

## G. Live integration (Evan)

### Loop structure
- **3 threads:** camera capture, `GestureWorker`, Pygame main thread (`docs/DESIGN.md` §2.1).
- **Worker, per frame** (`worker.py` `_run`): `camera.read()` → `tracker.detect()` → if a hand is found, `preprocess()` + `classifier.predict()`, else token `no_hand` → `smoother.update()` → `debouncer.update()` → publish an observation and queue any event.
- **Main thread, per frame** (`main.py` `run`): tick at 60 FPS → handle keys → `controller.set_gesture_info()` from `worker.latest()` → for each `poll_events()`: `gesture_to_action()` → `apply_action()` → log latency → `controller.update(dt)` → `render()` → status panel → `flip()`.

### Parameters (`src/config.py`)
| Parameter | Value | Meaning |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.70 | Below this, the prediction becomes `none` |
| `SMOOTHING_WINDOW_S` | 0.25 s | Majority-vote window (capture-time based, so independent of FPS) |
| `SMOOTHING_MIN_FRACTION` | 0.6 | A token needs ≥ 60 % of the window to become stable; otherwise the previous token stays (hysteresis) |
| `SMOOTHING_MIN_FRAMES` | 3 | The window never shrinks below 3 frames (low-FPS guard) |
| `ACTION_COOLDOWN_S` | 0.30 s | Minimum gap between steering events |

### Held-gesture protection (`smoothing.py` `Debouncer`)
- **Edge trigger:** an event fires only when the *stable* token changes, so holding a fist for 5 s gives one move.
- **Return to neutral:** to repeat a move, relax until `none` becomes stable, then show the gesture again.
- **Cooldown:** fist/V events at least 0.30 s apart. A rejected edge is dropped, not queued. Palm and no-hand bypass the cooldown.

### No hand & pause/resume
- No hand → raw `no_hand`. Once it wins the 0.25 s vote, PLAYING/COUNTDOWN → **pause** (`actions.py`). Target "pauses within ≈ 0.3 s" (`docs/workflow.md` Iteration 2 acceptance); measured value NOT FOUND – ask Evan.
- Open palm → pause. In GAME_OVER → restart.
- Paused + fist / V sign → `resume()` then move (the "resume-by-steer" rule). `none` never resumes (`actions.py` `apply_action`).

*Smoothing + debounce core (`src/app/smoothing.py`)*
```python
# GestureSmoother.update: time-window majority vote with hysteresis
self._history.append((t, token))
while len(self._history) > self.min_frames and self._history[0][0] < t - self.window_s:
    self._history.popleft()
if len(self._history) >= self.min_frames:
    winner, count = Counter(tok for _, tok in self._history).most_common(1)[0]
    if count / len(self._history) >= self.min_fraction:
        self.stable = winner
# Debouncer.update: emit only on change; steering cooldown
if stable == self._last_stable: return None
self._last_stable = stable
if stable == LABEL_NONE: return None
if stable in (LABEL_FIST, LABEL_V_SIGN):
    if now - self._last_emit < self.cooldown_s: return None
    self._last_emit = now
return stable
```

---

## H. Testing & evaluation (Hieu)

### Automated tests: run 2026-10-15, **274 passed, 1 skipped, 0 failed** (computed `pytest`)
| File | Checks | Pass / skip |
|---|---|---|
| `test_imports.py` | Every module imports; config invariants | 41 / 0 |
| `test_camera.py` | Threaded capture, newest-frame, mirroring, failures, timeouts, FPS | 25 / 0 |
| `test_landmarks.py` | BGR→RGB, first hand only, real MediaPipe on blank image | 8 / 0 |
| `test_preprocess.py` | Invariance (translation/scale), left/right mirroring, validation | 22 / 0 |
| `test_collect.py` | Recorder start/stop, delay, auto-stop, undo | 13 / 0 |
| `test_dataset.py` | Every cleaning rule + quality report | 14 / **1 skipped** (needs `data/sample/`, empty) |
| `test_split.py` | User-based split, leakage guards | 12 / 0 |
| `test_model.py` | **Empty: 0 tests** (Siri's model tests missing) | 0 / 0 |
| `test_model_adapter.py` | Real SVM loads; probabilities, threshold, bad input | 11 / 0 |
| `test_entities.py` | Lane geometry, AABB collision | 12 / 0 |
| `test_gameplay.py` | Spawning (never 3 lanes), speed ramp, score | 14 / 0 |
| `test_states.py` | All state transitions, countdown, grace | 21 / 0 |
| `test_controller.py` | Controller API per state | 35 / 0 |
| `test_actions.py` | All 20 gesture×state cells, resume-before-move | 22 / 0 |
| `test_smoothing.py` | Noisy streams, different FPS, one command per hold, cooldown | 10 / 0 |
| `test_worker.py` | Real worker thread with scripted predictions | 5 / 0 |
| `test_latency.py` | Latency metrics + CSV output | 2 / 0 |
| `test_main.py` | App smoke run, mock selection, errors | 3 / 0 |
| `test_mock_classifier.py` | Key-state snapshot mapping | 4 / 0 |

### Response time
| Item | Value | Source |
|---|---|---|
| Method | `time.perf_counter()` on one clock: onset of raw streak → trigger frame → landmarks done → predict done → controller call returns. Logged per action that changed the game | `src/app/latency.py`; `docs/notes/evan.md` |
| Capture FPS | 30.07 bright / 29.94 dim | `results/triet-fps-*.txt` |
| Render FPS | target 60 | `config.py` `TARGET_FPS` |
| MediaPipe landmark time | NOT FOUND – ask Evan (no `results/latency_*.csv`) | – |
| Feature extraction | median **0.013 ms** (offline, this Mac, n = 1,000) | computed `timing.preprocess` |
| SVM classification | median **0.160 ms**, p95 0.205 ms (offline) | computed `timing.svm_predict_proba` |
| Smoothing + debounce | median **0.002 ms** (offline) | computed |
| Game update / render time | NOT FOUND – ask Uy/Evan | – |
| End-to-end gesture→action (mean / median / p95) | NOT FOUND – ask Evan (needs ≥ 50 actions per lighting condition) | `docs/workflow.md` Iteration 5 |

Design floor (derived from config, not measured): a gesture must win a 0.25 s window at ≥ 60 %, so the expected smoothing delay is ≈ 0.15–0.25 s plus MediaPipe time.

### Live tests across users/conditions
- Live success rate per user / lighting / distance: NOT FOUND – ask Evan (`results/live_trials.csv` missing).
- Gesture-only play checklist (bright + dim): NOT FOUND – ask Evan.
- ≥ 3 full games per member + issues filed: NOT FOUND – ask Triet, Uy, Shobita.
- **Offline replay (stand-in, computed):** test user u05 `bright_near` (320 recorded rows at 10 Hz) fed through the real model + threshold + smoother + debouncer fired **10 commands**:
  - 4 correct: fist, V sign, open palm, plus one **duplicate V sign** after a 0.3 s `none` dip.
  - **6 false**: 4× palm, 2× fist, all during the `none` segment.
  - See figure `07_raw_vs_smoothed_replay.png`.

### Bugs found in review (Hieu)
| Bug | Status | Evidence |
|---|---|---|
| `ui._draw_centered()` lost its body in commit `fa9c863`, so the PAUSED / GAME OVER / countdown text never rendered. No test caught it | Fixed locally on 2026-10-15, **not yet committed** | `src/game/ui.py` (git diff) |
| `main.py` labels the real controller "TEMPORARY CONTROLLER: no obstacles/collisions" because it detects `lane`/`calls`, which `d820671` added | Open | `src/app/main.py` `_status`, `run` |
| 9-line debug panel (≈ 220 px) covers the game HUD and the top of the road | Open | `src/app/main.py` `_status` |
| `tests/test_model.py` has no tests | Open | file |
| `evaluate.py` has no `--timing`; no `results/` files written; model name `classifier.pkl` ≠ spec `gesture_svm.joblib` | Open | `src/model/evaluate.py`, `docs/workflow.md` |
| Keyboard key-state regression in mocks (SDL scancodes) | Fixed by Evan | `docs/notes/evan.md` |
| Font cache crash after `pygame.quit()` | Fixed by Uy (`fa9c863`) | git log |

### When it works best / worst (from the results above)
- **Best:** right hand, steering gestures (V sign F1 0.97, fist 0.92), near distance (86.4 %), dim_near (87.8 %). Lighting barely matters (85.5 % bright vs 85.4 % dim).
- **Worst:** **`none` poses** (recall 0.51; 33 % read as open palm), which become unwanted pauses or moves in play. **Left hand** in test (61.9 %, n = 176). `dim_far` (83.0 %). Some users are much harder than others (validation fold for u01: ≈ 0.78).

---

## I. Limitations & future work

### Known limitations (from code/notes)
- `none` class is hard: recall 0.51 (computed; `siri.md` §6).
- Single static frame, no motion/temporal model (`siri.md` §6).
- Single hand; with two hands visible, the controlling hand may switch (`triet.md` §6).
- 7 people, one webcam, no backlit data (`shobita.md` §6).
- Left hand under-represented: 346 of 2,240 `bright_near` rows are left-hand (`shobita.md` §6).
- No preprocessing-version metadata in the model file (`model_adapter.py` comment; `evan.md`).
- Lock file not verified on Windows (`requirements.lock` header).
- Rotation not normalised (`preprocess.py` comment, by design).
- TODO / FIXME comments: **none** in `src/`, `tests/`, `docs/` (grep).
- Doc drift noted in code: `GameController.lane`/`calls` and `render(fps)` need edits in `docs/interfaces.md` §4.4 (`controller.py` docstrings).

### Concrete improvements
1. **Record more varied `none` data** (relaxed half-open hands especially) from train users and retrain. This targets the 319 none→palm errors.
2. **Tune gamma** (the grid in `config.py` already exists but is unused) and add class weights or a per-class threshold for `open_palm`.
3. **Require return-to-`none` hold time** or raise `SMOOTHING_MIN_FRACTION` to stop the duplicate-move seen in the replay.
4. Add rotation normalisation or more left-hand training data to close the 62 % vs 87 % left/right gap.
5. Measure the real latency and live success (the missing items below), and add `test_model.py`.
6. Shrink or toggle the debug panel in `main.py` so the HUD stays visible.

---

## J. Figures

### Existing figures/screenshots in the repo
| Path | Content |
|---|---|
| `img/confusion_matrix.png` | Terminal output of `evaluate.py`: 85.44 %, confusion matrix, classification report |
| `img/a.png` | `results/triet-tests.xml` open in an editor (55 tests, 0 failures) |
| `img/Screenshot 2026-10-11 170016.png` | Two Triet tests PASSED (first hand only; left/right mirroring) |
| `img/Screenshot 2026-10-11 170138.png` | `triet-fps-bright.txt` (30.07 FPS) |
| `img/Screenshot 2026-10-11 170212.png` | `triet-fps-dim.txt` (29.94 FPS) |
| `img/Screenshot 2026-10-11 170344.png` | `triet-preview-log.txt` frames/timestamps (~30 FPS) |
| Gameplay screenshots / GIF / video | NOT FOUND – ask Uy (Iteration 5 task) |
| Note | `siri.md` links `../img/confustion_matrix.png` (typo, broken link) |

### Generated by `presentation_tools/make_figures.py` → `presentation_figures/` (300 dpi)
| File | Status |
|---|---|
| `01_class_distribution_per_split.png` | ✅ |
| `02_confusion_matrix_normalised.png` | ✅ |
| `03_per_class_f1.png` | ✅ |
| `04_validation_heatmap_C_x_kernel.png` | ✅ C × **kernel** (gamma was not searched) |
| `05_latency_breakdown.png` | ⛔ skipped: no `results/latency_*.csv` from webcam runs |
| `06_offline_test_accuracy_per_user_condition.png` | ✅ offline test accuracy (not live success) |
| `06b_live_success_rate.png` | ⛔ skipped: no `results/live_trials.csv` |
| `07_raw_vs_smoothed_replay.png` | ✅ offline replay of real u05 data, labelled as such |

The script re-checks for the missing files on every run, so figures 05 and 06b appear automatically once the data is committed.

---

## MISSING INFO checklist

**Triet**
- [ ] Any tutorial/reference used for camera/landmarks/preprocess code.
- [ ] MediaPipe landmark time per frame (or provide via Evan's latency CSV).
- [ ] Played ≥ 3 full gesture games + issues filed.

**Uy**
- [ ] Gameplay screenshots of every state + GIF/video for slides.
- [ ] Game update/render time per frame.
- [ ] Any tutorial/reference used; technical notes `docs/notes/uy.md` (file does not exist).
- [ ] Difficulty check (teammate survives ≥ 30 s); ≥ 3 gesture games played.

**Shobita**
- [ ] Background description of recording sessions.
- [ ] ≥ 3 gesture games played + issues filed.

**Siri**
- [ ] Other models compared (KNN/RF/MLP) with scores, if any were tried.
- [ ] Evidence for the 0.70 threshold (false-trigger rate on validation `none` ≤ 5 %).
- [ ] `tests/test_model.py` (currently empty); `evaluate.py --timing`; `results/` evaluation files.
- [ ] Gamma tuning (configured but unused), if it should be claimed.
- [ ] Any tutorial/reference used.

**Evan**
- [ ] `results/latency_*.csv` from real webcam, ≥ 50 actions each in bright and dim → end-to-end mean/median/p95 + per-stage MediaPipe time.
- [ ] Measured time from hand removal to auto-pause.
- [ ] Live success-rate trials per user / lighting / distance (`results/live_trials.csv`).
- [ ] Gesture-only play checklist (7 steps) passed in bright and dim.
- [ ] Any tutorial/reference used.

**Hieu (Lead)**
- [ ] Commit the `ui.py` overlay fix; decide on the `main.py` label/panel issues.
- [ ] Windows verification of `requirements.lock`; tag `v0.9-integrated`.
