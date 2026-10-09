# Repository Structure & Interface Specifications

Part of the design docs — start at [DESIGN.md](DESIGN.md) for the index.

---

## 3. Repository structure

Owners are named as in [Section 1.4](DESIGN.md#14-team-and-ownership): **Lead**, **Triet** (Hand tracking), **Uy** (Car game), **Shobita** (Dataset), **Siri** (Classifier), **Evan** (Live integration).

```text
carSimulationGame/
├── .github/
│   └── workflows/tests.yml         # Lead    – OPTIONAL CI: pytest on every push/PR (§12)
├── src/
│   ├── __init__.py                 # Lead    – marks src as a package (empty)
│   ├── config.py                   # Lead    – ALL tunable constants (§6.5); no project imports
│   ├── capture/                    # ── Triet (Hand tracking) ──
│   │   ├── __init__.py             # Triet   – re-exports Camera, Frame, HandTracker, HandLandmarks, preprocess
│   │   ├── camera.py               # Triet   – Camera + Frame: threaded OpenCV capture, mirroring, timestamps, FPS
│   │   ├── landmarks.py            # Triet   – HandLandmarks + HandTracker (MediaPipe Hands wrapper, landmark drawing)
│   │   └── preprocess.py           # Triet   – THE single preprocess() (+ row_to_landmarks) for training AND live
│   ├── data/                       # ── Shobita (Dataset) ──
│   │   ├── __init__.py             # Shobita – re-exports Dataset, load_dataset, Splits, load_splits, load_split
│   │   ├── collect.py              # Shobita – recording tool: label key starts, same key/Space stops → data/raw/*.csv
│   │   ├── dataset.py              # Shobita – load_dataset(): schema checks, invalid-sample removal, --report
│   │   └── split.py                # Shobita – user-based train/test split → data/splits.json; load_split()
│   ├── game/                       # ── Uy (Car game) ──
│   │   ├── __init__.py             # Uy      – re-exports GameController, GameState
│   │   ├── entities.py             # Uy      – Car, Obstacle dataclasses + collides() (AABB)
│   │   ├── gameplay.py             # Uy      – Gameplay: lanes, spawning, movement, collisions, score, speed
│   │   ├── render.py               # Uy      – draw_world(): road, lane lines, car, obstacles (Pygame)
│   │   ├── keyboard_demo.py        # Uy      – standalone keyboard-only loop driving Gameplay (dev tool)
│   │   ├── states.py               # Uy      – GameState enum + StateMachine (pure, no Pygame)
│   │   ├── ui.py                   # Uy      – HUD + pause/countdown/game-over overlays (Lead/Triet backup, §10)
│   │   └── controller.py           # Uy      – GameController: the ONLY API integration calls
│   ├── model/                      # ── Siri (Classifier) ──
│   │   ├── __init__.py             # Siri    – re-exports GestureClassifier, Classifier
│   │   ├── train.py                # Siri    – build_pipeline(), tune (LOUO over train users), train_model(), save
│   │   ├── evaluate.py             # Siri    – one-time test-set evaluation, confusion matrix, per-condition tables
│   │   └── predict.py              # Siri    – Classifier Protocol + GestureClassifier (load model, predict())
│   └── app/                        # ── Evan (Live integration) ──
│       ├── __init__.py             # Evan    – empty
│       ├── main.py                 # Evan    – entry point: Pygame loop (60 FPS), CLI flags, keyboard fallback, preview
│       ├── worker.py               # Evan    – GestureWorker thread: frame → landmarks → predict → smoothing → events
│       ├── actions.py              # Evan    – Action enum, gesture_to_action() (THE mapping), apply_action()
│       ├── smoothing.py            # Evan    – GestureSmoother (time-window majority vote) + Debouncer
│       ├── latency.py              # Evan    – LatencyLogger: timestamps → results/latency_*.csv + summary
│       └── mock_classifier.py      # Evan    – KeyboardState, MockClassifier, MockCamera, MockHandTracker
├── data/                           # ── Shobita owns everything under data/ ──
│   ├── raw/                        # Shobita – one CSV per recording session (§7.4); never edited by hand
│   ├── sample/                     # Shobita – 4 pseudo-users' CSVs + splits.json for development (§7.4)
│   └── splits.json                 # Shobita – user-based split, frozen before tuning (§7.5)
├── models/
│   └── gesture_svm.joblib          # Siri    – trained model bundle (committed; a few MB)
├── results/                        # Shobita/Siri/Evan – evaluation outputs for the slides (§8.5)
├── tests/
│   ├── conftest.py                 # Lead    – sets SDL_VIDEODRIVER=dummy, shared fixtures (fake HandLandmarks)
│   ├── test_imports.py             # Lead    – every module imports; config sanity checks
│   ├── test_camera.py              # Triet
│   ├── test_landmarks.py           # Triet
│   ├── test_preprocess.py          # Triet
│   ├── test_entities.py            # Uy
│   ├── test_gameplay.py            # Uy
│   ├── test_states.py              # Uy
│   ├── test_controller.py          # Uy
│   ├── test_collect.py             # Shobita
│   ├── test_dataset.py             # Shobita
│   ├── test_split.py               # Shobita
│   ├── test_model.py               # Siri
│   ├── test_actions.py             # Evan
│   ├── test_smoothing.py           # Evan
│   ├── test_mock_classifier.py     # Evan
│   └── test_worker.py              # Evan
├── docs/
│   ├── DESIGN.md                   # Lead    – index, overview (§1), architecture (§2), traceability (§16)
│   ├── interfaces.md               # Lead    – repo structure (§3), interface specifications (§4)
│   ├── game-logic.md               # Lead    – state machine (§5), gesture pipeline & config.py (§6)
│   ├── data-eval.md                # Lead    – dataset plan (§7), evaluation plan (§8)
│   ├── workflow.md                 # Lead    – testing, tasks, iterations, Git, coding standards (§9–§13)
│   ├── risks.md                    # Lead    – risks & mitigations (§15)
│   └── notes/
│       ├── TEMPLATE.md             # Lead    – technical notes template (§14)
│       ├── {triet,uy,shobita,siri,evan}.md  # each person – technical notes (§14)
│       └── img/                    # Uy      – game-state screenshots for notes and slides (Iteration 5)
├── .gitignore                      # Lead    – venv/, __pycache__/, .pytest_cache/, *.pyc, .vscode/ (except settings.json)
├── requirements.txt                # Lead    – direct dependencies, exact pins (§3.1)
├── requirements.lock               # Lead    – full `pip freeze` incl. transitive deps; what everyone installs (§3.1)
└── README.md                       # Lead    – setup, how to run game / collector / training / tests
```

How to run things (always from the repository root, so `src` is importable):

| Command | What it does | Owner |
|---|---|---|
| `python -m src.app.main` | Full game, real webcam, real SVM | Evan |
| `python -m src.app.main --classifier mock` | Real webcam + MediaPipe, gestures typed on keys 1/2/3 | Evan |
| `python -m src.app.main --classifier mock --camera mock` | No webcam at all (pure keyboard dev mode) | Evan |
| `python -m src.app.main --log-latency` | Same as default, plus writes `results/latency_<timestamp>.csv` | Evan |
| `python -m src.game.keyboard_demo` | Game only, arrow keys | Uy |
| `python -m src.data.collect --user u03 --condition bright_near` | Recording tool | Shobita |
| `python -m src.data.dataset --report` | Cleaning + quality report → `results/dataset_report.txt` | Shobita |
| `python -m src.data.split --test u02 u05` | Write the user-based split `data/splits.json` | Shobita |
| `python -m src.model.train` | Tune on train users (LOUO), train, save `models/gesture_svm.joblib` | Siri |
| `python -m src.model.evaluate` | One-time evaluation on the test users, writes `results/` | Siri |
| `pytest` | All unit tests | everyone |

### 3.1 `requirements.txt` and `requirements.lock`

`requirements.txt` lists our **direct** dependencies with exact pins. We installed this set into a clean 64-bit Python 3.12.14 venv on macOS and checked it: `mediapipe.solutions.hands.Hands` constructs and runs.

```text
# Python 3.12 (64-bit) ONLY. Verify: python -c "import struct,sys; print(struct.calcsize('P')*8, sys.version)"
mediapipe==0.10.21
opencv-contrib-python==4.11.0.86
numpy==1.26.4
scikit-learn==1.6.1
scipy==1.17.1
joblib==1.4.2
pygame==2.6.1
matplotlib==3.11.2
protobuf==4.25.9
pytest==8.3.5
```

Why these exact versions:

- **`mediapipe==0.10.21`** is the newest release that ships CPython 3.12 wheels for Windows (`win_amd64`), macOS (`universal2`, `x86_64`) and Linux (`manylinux_2_28_x86_64`) **and** still provides the legacy `mp.solutions.hands` API that we use. Releases from 0.10.30 onwards switched to a different packaging (pure `py3` wheels), and we have not verified `mp.solutions` there, so we do not use them.
- **`numpy==1.26.4`**: MediaPipe 0.10.21 requires `numpy<2`. 1.26.4 is the last 1.x release and supports 3.12.
- **`opencv-contrib-python`, not `opencv-python`**: MediaPipe already depends on the contrib build. Installing both puts two conflicting `cv2` packages in one environment. **Never `pip install opencv-python`.**
- **`protobuf==4.25.9`**: MediaPipe requires `protobuf>=4.25.3,<5`. We pin it so everyone resolves the same version.
- **`matplotlib`**: used for the confusion matrix and latency charts in `results/` (it is also a MediaPipe dependency).

**Transitive dependencies are not pinned by `requirements.txt`.** MediaPipe pulls in `jax`, `jaxlib`, `absl-py`, `flatbuffers`, `sentencepiece`, `sounddevice`, `attrs` and others with open version ranges. Two people installing on different days could get different versions. So:

- **`requirements.lock`** is the output of `pip freeze`, taken right after a successful `pip install -r requirements.txt` in a **clean** 64-bit Python 3.12 venv. It pins every package, transitive ones included.
- **Everyone installs from the lock file**, not from `requirements.txt`. `requirements.txt` is only the human-edited list that the lock is regenerated from.
- **One lock for every platform:** the lock is generated on **one** OS, but it must install cleanly on **Windows, macOS and, if CI is on, Ubuntu** (the CI runner).
- **Re-verification (Lead, Iteration 1):**
  - Install the lock into a fresh 3.12 venv on **Windows and macOS**, and on **Ubuntu** via the CI run if CI is enabled.
  - Run the MediaPipe check below.
  - Run `pytest`.
- **Platform-specific packages:** if a package exists only on some platforms, or one platform needs a different version, give that line an environment marker, for example `; sys_platform == "win32"`, `; sys_platform == "darwin"` or `; sys_platform == "linux"`. Never let the platforms silently drift.
- **When to regenerate:** whenever `requirements.txt` changes. The new lock goes in the same PR.

Setup (Iteration 1, everyone):

```bash
py -3.12 -m venv .venv          # Windows   |  python3.12 -m venv .venv   (macOS/Linux)
.venv\Scripts\activate           # Windows   |  source .venv/bin/activate
pip install -r requirements.lock
python -c "import mediapipe as mp, cv2; print(mp.__version__, cv2.__version__)"   # expect 0.10.21 4.11.0
```

Regenerating the lock (Lead only):

```bash
python3.12 -m venv .venv-clean && source .venv-clean/bin/activate   # Windows: py -3.12 ... / .venv-clean\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements.lock
```

---

## 4. Interface specifications

Conventions used in every interface:

- **Units.** Time is in seconds (`float`, from `time.perf_counter()`) unless a name ends in `_ms`. Screen positions and sizes are in pixels (`float`). Speeds are in pixels/second. MediaPipe coordinates are normalised (see `HandLandmarks`).
- **Errors.** Invalid arguments raise `ValueError` (wrong shape or value) or `TypeError` (wrong type). Hardware or file problems raise `RuntimeError` or `FileNotFoundError` with a message that says how to fix it. Game control methods **never raise** on calls that are valid but have no effect (for example `pause()` while already paused). They just do nothing, which keeps the live loop robust.
- **Threads.** [Section 2.1](DESIGN.md#21-component-diagram) rule 7 says which thread may call what. Classes that are shared between threads (`Camera`, `GestureWorker`, `KeyboardState`) say so explicitly and are internally locked.
- **Interface additions.** The brief allows additions beyond the fixed interfaces when they are justified. Each one is marked **[added]** with its reason.

### 4.1 Shared: `src/config.py` (Lead)

This file contains constants only (full list in [Section 6.5](game-logic.md#65-srcconfigpy--all-named-constants-lead-creates-in-iteration-1)). Label constants:

```python
LABEL_FIST: str = "fist"
LABEL_V_SIGN: str = "v_sign"
LABEL_OPEN_PALM: str = "open_palm"
LABEL_NONE: str = "none"
LABELS: tuple[str, ...] = (LABEL_FIST, LABEL_V_SIGN, LABEL_OPEN_PALM, LABEL_NONE)
NO_HAND: str = "no_hand"   # [added] internal pipeline token, NEVER a classifier output or CSV label
```

**[added] `NO_HAND`:** the "no hand → pause" rule (LI.4) has to pass through the same smoothing and mapping as real gestures. That way a single dropped frame does not pause the game, and the rule lives in the one mapping function (I6).

### 4.2 Triet (Hand tracking) – `src/capture`

#### `Frame` and `Camera` (`camera.py`) — provided by Triet, consumed by Shobita (`collect.py`) and Evan (`worker.py`, `main.py`)

```python
@dataclass(frozen=True)
class Frame:
    image: np.ndarray      # BGR uint8, shape (FRAME_HEIGHT, FRAME_WIDTH, 3), already mirrored if MIRROR_FRAME
    t_capture: float       # time.perf_counter() taken immediately after cv2.VideoCapture.read() returned (s)
    index: int             # 0-based counter of frames captured since the Camera was opened (gaps = dropped frames)

class Camera:
    def __init__(self, index: int = CAMERA_INDEX, width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT, mirror: bool = MIRROR_FRAME,
                 threaded: bool = CAMERA_THREADED) -> None: ...
    def read(self, timeout: float = CAMERA_READ_TIMEOUT_S) -> Frame: ...
    def latest(self) -> Frame | None: ...          # [added] newest frame without blocking (preview)
    @property
    def fps(self) -> float: ...                    # [added] measured capture FPS, rolling mean over 1 s
    def close(self) -> None: ...
    def __enter__(self) -> "Camera": ...
    def __exit__(self, *exc) -> None: ...          # calls close()
```

- **Threaded by default** (`CAMERA_THREADED = True`):
  - `__init__` starts a daemon capture thread. The thread reads, mirrors, resizes and timestamps frames in a loop, and stores **only the newest one** in a lock-protected slot.
  - `read()` blocks until a frame **newer than the last one `read()` returned** is available, and never returns the same frame twice. So a consumer slower than the camera drops frames instead of falling behind.
  - `latest()` returns the newest frame immediately, possibly the same frame again, and is used for preview drawing.
  - `Camera` is safe to share between threads.
- With `threaded=False`, `read()` captures synchronously in the caller's thread. Use it only for debugging or a quick script; the game and the collector use the default.
- **Errors:**
  - `__init__` raises `RuntimeError("Cannot open camera index 0 – close other apps using the webcam or set CAMERA_INDEX")` if `cv2.VideoCapture(index).isOpened()` is False.
  - `read()` raises `RuntimeError` if no new frame arrives within `timeout`, or after `CAMERA_MAX_FAILED_READS` (5) consecutive failed reads. The capture thread records the error and `read()` re-raises it in the caller's thread. `read()` never returns `None`.
- `close()` stops the thread (joins it within 1 s) and releases the device. It is idempotent.
- If the camera returns a different size, frames are resized to `(width, height)`, so downstream code always sees 640×480.
- **[added] `latest()` and `fps`:** the Pygame loop runs at 60 FPS and needs a non-blocking frame for the preview. The HUD/latency summary report the real camera FPS, which matters because dim light lowers it ([Section 6.2](game-logic.md#62-smoothing-gesturesmoother-evan)).
- **Why we mirror:** the player sees a mirror-like image, and MediaPipe's handedness labels assume a mirrored (selfie) image. With the mirror, a physical right hand is labelled `"Right"`.

#### `HandLandmarks` and `HandTracker` (`landmarks.py`) — provided by Triet, consumed by Shobita (`collect.py`, `dataset.py`) and Evan (`worker.py`)

```python
@dataclass(frozen=True)
class HandLandmarks:
    points: np.ndarray     # float, shape (21, 3): MediaPipe normalised (x, y, z) per landmark
                           #   x ∈ [0,1] fraction of image width, y ∈ [0,1] fraction of image height
                           #   (may slightly exceed [0,1] near edges), z = depth relative to wrist,
                           #   roughly the same scale as x. Index order = MediaPipe HandLandmark enum.
    handedness: str        # "Right" or "Left" (MediaPipe label on the MIRRORED frame)
    score: float           # MediaPipe handedness confidence in [0, 1]

class HandTracker:
    def __init__(self, max_num_hands: int = MP_MAX_NUM_HANDS,
                 model_complexity: int = MP_MODEL_COMPLEXITY,
                 min_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE) -> None: ...
    def detect(self, image_bgr: np.ndarray) -> HandLandmarks | None: ...
    def draw(self, image_bgr: np.ndarray, hand: HandLandmarks) -> None: ...   # in-place overlay for preview/collector
    def close(self) -> None: ...
```

- `detect()` converts BGR to RGB, runs `mp.solutions.hands.Hands.process`, and returns the **first** detected hand, or `None` if there is no hand. If `image_bgr` is not a `uint8` array of shape (H, W, 3) it raises `ValueError`.
- **Not thread-safe:** create and use one `HandTracker` per thread. The game creates it inside the `GestureWorker` thread.
- **[added] `HandLandmarks`:** I3 fixes the signature as `preprocess(landmarks)`, but mirroring the left hand (HT.3) needs the handedness. Bundling the points and the handedness in one object keeps the fixed one-argument signature.

#### `preprocess` (`preprocess.py`) — provided by Triet, consumed by Shobita (`dataset.py`, i.e. training data) and Evan (`worker.py`, live)

```python
PREPROCESS_VERSION: int  # imported from config; bump whenever the algorithm changes

def preprocess(landmarks: HandLandmarks) -> np.ndarray:
    """Return a float32 feature vector of shape (63,) — see Section 6.1 for the exact steps."""

def row_to_landmarks(features_raw: Sequence[float], handedness: str) -> HandLandmarks:
    """[added] Rebuild a HandLandmarks from a CSV row's f0..f62 + handedness (used by dataset.py)."""
```

| | |
|---|---|
| Input | `HandLandmarks` (raw MediaPipe values) |
| Output | `np.ndarray`, dtype `float32`, shape `(63,)`, layout `[x0, y0, z0, x1, y1, z1, …, x20, y20, z20]`, dimensionless (in units of palm size) |
| Errors | `ValueError` if `points.shape != (21, 3)`, if any value is non-finite, if `handedness` is not in `{"Left", "Right"}`, or if palm size < `MIN_PALM_SIZE` |
| Purity | Pure and deterministic. It never mutates its input. |

**[added] `row_to_landmarks`:** this guarantees that training reads a CSV row back into exactly the same object type that live prediction gets. Then **both paths call the same `preprocess()`** (HT.3).

### 4.3 Uy (Car game) – `src/game`: gameplay

#### `entities.py` — provided by Uy, consumed by Uy (`gameplay.py`, `render.py`, `ui.py`)

```python
@dataclass
class Car:
    lane: int              # 0 = left, 1 = middle, 2 = right
    width: float = CAR_WIDTH            # px
    height: float = CAR_HEIGHT          # px
    y: float = CAR_Y                    # px, top edge (fixed)
    @property
    def x(self) -> float: ...           # px, left edge = lane_center_x(lane) - width/2

@dataclass
class Obstacle:
    lane: int
    y: float                            # px, top edge; increases as it falls
    width: float = OBSTACLE_WIDTH
    height: float = OBSTACLE_HEIGHT
    @property
    def x(self) -> float: ...

def lane_center_x(lane: int) -> float:
    """Centre x (px) of a lane. ValueError if lane not in range(NUM_LANES)."""

def collides(car: Car, obstacle: Obstacle) -> bool:
    """Axis-aligned bounding-box overlap test (touching edges do NOT count as collision)."""
```

#### `Gameplay` (`gameplay.py`) — provided by Uy, consumed **only** inside `src/game` (`GameController`, `keyboard_demo.py`)

```python
class Gameplay:
    def __init__(self, rng: random.Random | None = None) -> None: ...
    car: Car
    obstacles: list[Obstacle]
    score: float           # points; += SCORE_PER_SECOND * dt while advancing
    speed: float           # px/s obstacle fall speed
    elapsed: float         # s of active (non-paused) play since reset

    def reset(self) -> None: ...                                # lane=1, no obstacles, score=0, speed=BASE_SPEED
    def move_left(self) -> bool: ...                            # True if lane changed; False at lane 0
    def move_right(self) -> bool: ...                           # True if lane changed; False at lane NUM_LANES-1
    def update(self, dt: float, collisions_enabled: bool = True) -> bool: ...
```

`update(dt)` advances the simulation by `dt` seconds:
1. `ValueError` if `dt < 0`. `dt` is clamped to `MAX_DT` (0.1 s) so a slow frame cannot teleport obstacles through the car.
2. `elapsed += dt`; `speed = min(MAX_SPEED, BASE_SPEED + SPEED_INCREASE_PER_SECOND * elapsed)`.
3. Move every obstacle down by `speed * dt`, and remove obstacles whose top is below `WINDOW_HEIGHT`.
4. Spawn based on **distance travelled**: once the newest obstacle has fallen `SPAWN_GAP_PX`, spawn a new row at `y = -OBSTACLE_HEIGHT`. Each row has 1 obstacle, or 2 with probability `DOUBLE_OBSTACLE_PROB`, in distinct random lanes. A row **never** has 3, so there is always a way through.
5. `score += SCORE_PER_SECOND * dt`.
6. If `collisions_enabled`, return `True` when `collides(car, o)` for any obstacle, otherwise `False`. `Gameplay` itself does not stop on collision. The decision belongs to `GameController`.

`Gameplay` does not import pygame, so it is fully unit-testable. Pass a seeded `random.Random` to get deterministic tests.

#### `render.py` and `keyboard_demo.py` — Uy

```python
def draw_world(surface: pygame.Surface, gameplay: Gameplay, blink_car: bool = False) -> None:
    """Draw road, lane dividers, obstacles and the car. blink_car=True draws the car semi-transparent (grace period)."""
```

`keyboard_demo.py` is a 60 FPS Pygame loop: ←/→ call `move_left`/`move_right`, and a collision prints the score and resets. It makes the gameplay demonstrable on its own (GM.6) before the state machine, UI and live integration exist.

**[added] `--controller` flag:** with it, the demo drives `GameController` instead of `Gameplay`. ←/→ move, `P` pauses/resumes, `R` restarts, and `1`/`2`/`3`/`0`/`H` feed fake values to `set_gesture_info`. Uy can then demonstrate every state, overlay and the HUD before Evan's integration exists, which the Iteration 2 acceptance check ([Section 11](workflow.md#11-development-plan-5-iterations)) needs.

### 4.4 Uy (Car game) – `src/game`: state + UI

#### `GameState` and `StateMachine` (`states.py`) — provided by Uy, consumed by Uy and Evan (`actions.py` reads `GameState`)

```python
class GameState(Enum):
    PAUSED = "paused"        # initial state; also after open palm / no hand
    COUNTDOWN = "countdown"  # after resume/restart: obstacles frozen, car CAN change lanes
    PLAYING = "playing"
    GAME_OVER = "game_over"

class StateMachine:
    def __init__(self) -> None: ...            # state = PAUSED
    state: GameState
    countdown_left: float                      # s remaining while in COUNTDOWN, else 0.0
    grace_left: float                          # s of collision immunity remaining after COUNTDOWN ends
    def pause(self) -> bool: ...               # PLAYING/COUNTDOWN → PAUSED; returns True if transitioned
    def resume(self) -> bool: ...              # PAUSED → COUNTDOWN (countdown_left = RESUME_COUNTDOWN_S)
    def restart(self) -> bool: ...             # GAME_OVER (or any state) → COUNTDOWN
    def crash(self) -> bool: ...               # PLAYING → GAME_OVER
    def tick(self, dt: float) -> None: ...     # counts down; COUNTDOWN → PLAYING (grace_left = GRACE_PERIOD_S)
```

#### `GameController` (`controller.py`) — provided by Uy, consumed by Evan (and tests)

```python
class GameController:
    def __init__(self, rng: random.Random | None = None) -> None: ...

    # ---- Fixed interface (I5) ----
    def move_left(self) -> bool: ...
    def move_right(self) -> bool: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def is_paused(self) -> bool: ...

    # ---- [added] needed by the live loop ----
    @property
    def state(self) -> GameState: ...
    @property
    def score(self) -> int: ...                 # int(gameplay.score)
    def restart(self) -> None: ...
    def update(self, dt: float) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...
    def set_gesture_info(self, label: str, confidence: float, hand_present: bool) -> None: ...
```

| Method | Behaviour by state | Returns / errors |
|---|---|---|
| `move_left()` / `move_right()` | Changes lane in `PLAYING` **and** `COUNTDOWN` (so the player can get out of the way before obstacles start moving). Ignored in `PAUSED` and `GAME_OVER`. | `True` if the lane actually changed, else `False`. Never raises. **[added return value]:** the latency logger only records actions that had an effect. |
| `pause()` | `PLAYING`/`COUNTDOWN` → `PAUSED`. Otherwise no-op. | `None` |
| `resume()` | `PAUSED` → `COUNTDOWN` (`RESUME_COUNTDOWN_S`). After the countdown → `PLAYING` with `GRACE_PERIOD_S` collision immunity. Otherwise no-op. | `None` |
| `is_paused()` | `True` only in `PAUSED`. (`COUNTDOWN` is not paused: the car can move.) | `bool` |
| `restart()` **[added]** | Resets `Gameplay`, → `COUNTDOWN`. Needed for GM.8. | `None` |
| `update(dt)` **[added]** | Calls `state_machine.tick(dt)`. In `PLAYING` it calls `gameplay.update(dt, collisions_enabled=grace_left <= 0)` and on collision `state_machine.crash()`. In other states `Gameplay` is frozen. `ValueError` if `dt < 0`. | `None` |
| `render(surface)` **[added]** | `draw_world(...)` (blinking car during grace), then `ui.draw_hud(...)`, then the overlay for the current state. | `None` |
| `set_gesture_info(...)` **[added]** | Stores the values for the HUD (GM.9/LI.6). `ValueError` if `confidence` ∉ [0, 1]. `label` may be any of `LABELS` or `NO_HAND`. | `None` |

`GameController` is **not thread-safe** and is used only on the main thread ([Section 2.1](DESIGN.md#21-component-diagram), rule 7). The live loop needs `restart`, `update`, `render`, `state` and `set_gesture_info`. Without them, Evan would have to reach into `Gameplay`/`StateMachine`, which would break GM.10 ("the only API").

#### `ui.py` — Uy (Lead/Triet backup, see [Section 10](workflow.md#10-per-person-task-breakdown))

```python
def draw_hud(surface: pygame.Surface, score: int, label: str, confidence: float,
             hand_present: bool, fps: float | None = None) -> None: ...
def draw_overlay(surface: pygame.Surface, state: GameState, countdown_left: float, score: int) -> None: ...
```

- **HUD (top bar, 48 px):** `Score: 123`, `Gesture: V sign (0.92)` (or `NO HAND` in red), and optionally `FPS: 29`. If the camera FPS is below `LOW_FPS_WARNING`, the FPS is shown in orange.
- **Overlays:**
  - `PAUSED`: dim the screen and show "PAUSED – show FIST or V SIGN to resume" (plain text, not emoji, so we don't depend on a font with emoji support).
  - `COUNTDOWN`: a large `2`, `1`.
  - `GAME_OVER`: "Game over – score N – open palm or R to restart".

### 4.5 Dataset & classifier – `src/data` (Shobita), `src/model` (Siri)

#### `collect.py` — provided by Shobita (built on Triet's `Camera`, `HandTracker`), used by everyone during Iteration 3

```python
def key_to_label(key: int) -> str | None:
    """Map an OpenCV waitKey code to a label: '1'→fist, '2'→v_sign, '3'→open_palm, '0'→none, else None."""

def landmarks_to_row(user_id: str, condition: str, label: str, hand: HandLandmarks) -> list[str | float]:
    """Return one CSV row: [user_id, condition, label, f0..f62 (raw x,y,z), handedness] (67 values)."""

class RecordingSession:
    """Pure start/stop recording logic (no OpenCV), so it is unit-testable."""
    def __init__(self, user_id: str, condition: str,
                 target_per_label: int = RECORD_TARGET_PER_LABEL,
                 sample_hz: float = RECORD_SAMPLE_HZ,
                 start_delay_s: float = RECORD_START_DELAY_S) -> None: ...
    active_label: str | None            # label currently being recorded, or None (stopped)
    counts: dict[str, int]              # rows kept per label in this session
    rows: list[list[str | float]]
    def handle_key(self, key: int, now: float) -> str: ...      # toggles start/stop; returns a status message
    def offer(self, hand: HandLandmarks | None, now: float) -> bool: ...   # maybe record one row
    def undo_last_burst(self) -> int: ...                       # removes rows of the last start→stop burst

def main(argv: list[str] | None = None) -> None:
    """CLI: --user uXX --condition <lighting>_<distance> [--target 80] [--out-dir data/raw]."""
```

Tool behaviour. **Press to start, press to stop:** OpenCV's `waitKey` reports key *presses*, not whether a key is *held*, so recording is a toggle.
- **Start:** pressing a label key (`1` fist, `2` v_sign, `3` open_palm, `0` none) while stopped starts recording that label after `RECORD_START_DELAY_S` (1 s), so the hand is in pose before the first sample. The overlay shows `GET READY` and then a red `REC fist 37/80`.
- **Stop:** pressing the **same label key again or Space** stops. Pressing a *different* label key while recording is **ignored**, with an orange warning, so a stray key can never mislabel samples. Recording also stops automatically when that label reaches `--target` (`RECORD_TARGET_PER_LABEL`).
- **Sampling:** while recording, `offer()` keeps at most one row per `1 / RECORD_SAMPLE_HZ` s (10 Hz), so consecutive rows are not near-duplicates. Frames without a hand are skipped, and the overlay shows "NO HAND".
- **Undo:** `Backspace` while stopped removes the last start→stop burst.
- **Saving:** rows are appended to the session file at every stop, so a crash loses at most one burst. `Q` stops and quits.
- **Validation:** `--user` must match `^u\d{2}$` and `--condition` must be in `CONDITIONS`. Otherwise the tool raises `SystemExit(2)` with a usage message.
- **Output:** it writes `data/raw/<user_id>_<condition>_<YYYYmmdd-HHMMSS>.csv` ([Section 7.4](data-eval.md#74-files-naming-privacy)), or into `--out-dir` for test recordings that must not enter the dataset. It never writes image data.

**CSV schema (I2 + one justified column):**

| Column | Type | Example | Notes |
|---|---|---|---|
| `user_id` | str | `u03` | Anonymous ID. The ID → name mapping is kept **off** the repo. |
| `condition` | str | `bright_near` | One of `CONDITIONS` ([Section 7.2](data-eval.md#72-conditions)) |
| `label` | str | `v_sign` | One of `LABELS` |
| `f0 … f62` | float | `0.5123` | **Raw** MediaPipe landmarks: `f(3i)=x_i`, `f(3i+1)=y_i`, `f(3i+2)=z_i`, i = 0..20 |
| `handedness` | str | `Right` | **[added]** Needed so that `preprocess()` can mirror left hands at training time (HT.3). It is appended **last** so that the first 66 columns match the agreed I2 order exactly. |

We store **raw** values rather than preprocessed ones. If `preprocess()` improves later (for example a new normalisation), we just retrain: nobody has to record again.

#### `dataset.py` — provided by Shobita, consumed by Shobita (`split.py`) and Siri (via `load_split`)

```python
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray           # float32 (n, 63) — preprocess() output
    y: np.ndarray           # str (n,) labels ∈ LABELS
    groups: np.ndarray      # str (n,) user_id (for user-based splitting / leave-one-user-out)
    conditions: np.ndarray  # str (n,) condition
    sessions: np.ndarray    # str (n,) source CSV file name (one recording session)
    handedness: np.ndarray  # str (n,) "Left"/"Right"

@dataclass(frozen=True)
class CleaningReport:
    rows_read: int
    rows_kept: int
    dropped: dict[str, int]          # reason → count (reasons listed in Section 7.5)
    excluded_files: dict[str, str]   # file name → reason (from splits.json)

def load_dataset(paths: Sequence[Path] | None = None,
                 excluded_files: Mapping[str, str] | None = None) -> tuple[Dataset, CleaningReport]:
    """Read CSVs (default: data/raw/*.csv), validate schema, drop invalid samples, apply preprocess()."""

def quality_report(dataset: Dataset, cleaning: CleaningReport) -> str:
    """Plain-text report: counts per user × condition × label, flags, dropped rows (Section 7.5)."""

def main(argv: list[str] | None = None) -> None:
    """CLI: --report [--data DIR] → prints and writes results/dataset_report.txt."""
```

Errors:
- `FileNotFoundError` if no CSV matches.
- `ValueError` naming the file if the header is not exactly `CSV_COLUMNS`, or if `user_id`/`condition` inside a file do not match its file name. A wrong header means a broken tool, so the whole file fails loudly.
- Invalid **rows** are not errors. They are dropped and counted in `CleaningReport.dropped` using the rules in [Section 7.5](data-eval.md#75-cleaning-quality-report-and-trainvalidationtest-split).

#### `split.py` — provided by Shobita, consumed by Siri

```python
@dataclass(frozen=True)
class Splits:
    data_dir: Path                      # directory the file names are relative to (data/raw or data/sample)
    train_users: tuple[str, ...]
    test_users: tuple[str, ...]
    validation: str                     # always "leave_one_user_out" (over train_users), see Section 7.5
    files: dict[str, tuple[str, ...]]   # "train" / "test" → CSV file names
    excluded_files: dict[str, str]      # file name → reason; in NO split

def make_splits(data_dir: Path = DATA_DIR, test_users: Sequence[str] | None = None,
                n_test_users: int = N_TEST_USERS, seed: int = SEED,
                excluded_files: Mapping[str, str] | None = None) -> Splits: ...
def save_splits(splits: Splits, path: Path = SPLITS_PATH) -> None: ...
def load_splits(path: Path = SPLITS_PATH) -> Splits: ...
def load_split(name: Literal["train", "test"], path: Path = SPLITS_PATH) -> Dataset:
    """Load only the files of one split (cleaned + preprocessed via load_dataset)."""
def main(argv: list[str] | None = None) -> None:
    """CLI: [--test u02 u05] [--exclude FILE=REASON ...] [--data DIR] [--out data/splits.json]"""
```

- If `test_users` is `None`, `make_splits` picks `n_test_users` users at random with `seed`, from users that have all `CORE_CONDITIONS`.
- **Raises `ValueError` if:**
  - a test user is unknown or lacks a core condition;
  - fewer than 3 train users remain;
  - any user or any file would land in two splits.
- `load_splits` re-checks the same invariants, so a hand-edited `splits.json` cannot leak.
- The `splits.json` format is in [Section 7.5](data-eval.md#75-cleaning-quality-report-and-trainvalidationtest-split).

#### `train.py` — provided by Siri

```python
def build_pipeline(C: float = SVM_C, gamma: float | str = SVM_GAMMA) -> sklearn.pipeline.Pipeline:
    """StandardScaler → SVC(kernel='rbf', C, gamma, probability=True, class_weight='balanced', random_state=SEED)."""

def tune(train: Dataset) -> tuple[dict, list[dict]]:
    """Grid SVM_GRID_C × SVM_GRID_GAMMA, LeaveOneGroupOut over train users, macro-F1.
    Returns (best_params, per-fold/per-param records)."""

def train_model(train: Dataset, C: float = SVM_C, gamma: float | str = SVM_GAMMA) -> Pipeline: ...

def save_model(pipeline: Pipeline, splits: Splits, path: Path = MODEL_PATH) -> None:
    """joblib.dump a bundle dict (see below)."""

def main(argv: list[str] | None = None) -> None:   # CLI: [--grid] [--splits data/splits.json] [--out models/gesture_svm.joblib]
```

`train.py` only ever calls `load_split("train")`. It never loads the test split.

Model bundle saved to `models/gesture_svm.joblib`:

```python
{
    "pipeline": Pipeline,                 # fitted on ALL train users with the tuned parameters
    "labels": list[str],                  # == list(pipeline.classes_)
    "preprocess_version": int,            # == config.PREPROCESS_VERSION at training time
    "sklearn_version": "1.6.1",
    "trained_on_users": ["u01", ...],     # == splits.train_users
    "test_users": ["u02", "u05"],         # == splits.test_users (never trained on)
    "params": {"C": 10.0, "gamma": "scale"},
    "created_utc": "2026-10-20T10:31:00Z",
}
```

#### `predict.py` — provided by Siri, consumed by Evan

```python
class Classifier(Protocol):
    """Anything the live loop can use: real SVM or keyboard mock (LI.8)."""
    def predict(self, features: np.ndarray) -> tuple[str, float]: ...

class GestureClassifier:   # implements Classifier
    def __init__(self, model_path: Path = MODEL_PATH,
                 threshold: float = CONFIDENCE_THRESHOLD) -> None: ...
    def predict(self, features: np.ndarray) -> tuple[str, float]: ...
    def predict_proba(self, features: np.ndarray) -> dict[str, float]: ...   # [added] for evaluation/debug
```

| | |
|---|---|
| Input | `features`: output of `preprocess()`, `float32`/`float64`, shape `(63,)` |
| Output | `(label, confidence)`. `label ∈ LABELS`. `confidence ∈ [0, 1]` = the highest class probability (Platt-scaled SVC `predict_proba`). The label is the argmax of the probabilities, not `SVC.predict`, so the two always agree. If `confidence < threshold`, the result is `("none", confidence)`. |
| Errors | `__init__`: `FileNotFoundError("models/gesture_svm.joblib not found – run python -m src.model.train")`. `ValueError` if the bundle's `preprocess_version != config.PREPROCESS_VERSION` (stops a stale model from silently using different features). `predict`: `ValueError` if shape ≠ (63,) or the input contains non-finite values. |
| Threads | Called only from the `GestureWorker` thread. |
| Performance | < 2 ms per call on a laptop CPU (Siri checks this with `evaluate.py --timing`). |

**[added] a `Classifier` Protocol with `predict` as a method:** I4 is satisfied by the method signature. A class lets us load the model once instead of on every frame, and lets `MockClassifier` be a drop-in replacement (LI.8). Evaluation uses `threshold=0.0` to get raw predictions.

#### `evaluate.py` — provided by Siri

```python
def validation_predictions(train: Dataset, C: float, gamma: float | str) -> list[dict]:
    """Out-of-fold predictions via leave-one-user-out over the TRAIN users (threshold tuning, per-user spread).
    Records: {user_id, condition, true, pred, confidence}."""

def test_predictions(bundle_path: Path = MODEL_PATH, splits_path: Path = SPLITS_PATH) -> list[dict]:
    """Predictions of the saved model on the TEST users. Raises ValueError if any test user is in
    bundle['trained_on_users'] or if bundle['test_users'] != splits.test_users (leakage guard)."""

def main(argv: list[str] | None = None) -> None:   # CLI: [--splits data/splits.json] [--timing] [--nice]; writes Section 8.5 Siri files
```

We deliberately do **not** add pandas as a dependency, so results are plain `list[dict]` and are written with the `csv` module. `--timing` measures the mean `predict()` time over 1,000 calls. `--nice` also produces the [Nice] outputs ([Section 8.5](data-eval.md#85-who-measures-what-and-output-formats)).

### 4.6 Evan (Live integration) – `src/app`

#### `worker.py` — provided by Evan, consumed by Evan (`main.py`)

```python
@dataclass(frozen=True)
class GestureObservation:
    token: str             # raw: classifier label (after threshold) or NO_HAND
    confidence: float      # [0, 1]; 0.0 for NO_HAND
    stable: str            # GestureSmoother output
    hand_present: bool
    frame_index: int
    t_capture: float       # Frame.t_capture (s)
    t_landmarks: float     # perf_counter() after HandTracker.detect (s)
    t_predict: float       # perf_counter() after Classifier.predict (= t_landmarks if no hand) (s)

@dataclass(frozen=True)
class GestureEvent:
    token: str             # debounced event token ∈ LABELS ∪ {NO_HAND}, never "none"
    t_onset: float         # t_capture of the FIRST frame of the raw-token streak that led to it (s)
    t_trigger: float       # t_capture of the frame whose processing emitted it (s)
    t_landmarks: float
    t_predict: float

class GestureWorker:
    """Background thread: camera → HandTracker → preprocess → Classifier → smoother → debouncer."""
    def __init__(self, camera: Camera | MockCamera,
                 tracker_factory: Callable[[], HandTracker | MockHandTracker],
                 classifier: Classifier,
                 smoother: GestureSmoother | None = None,
                 debouncer: Debouncer | None = None) -> None: ...
    def start(self) -> None: ...                         # starts the daemon thread
    def stop(self, timeout: float = 1.0) -> None: ...    # signals and joins; idempotent
    def latest(self) -> GestureObservation | None: ...   # non-blocking, for the HUD
    def poll_events(self) -> list[GestureEvent]: ...     # drains the event queue (main thread)
    @property
    def fps(self) -> float: ...                          # processed frames per second, rolling 1 s
    @property
    def error(self) -> BaseException | None: ...         # set if the thread died; main.py shows it and exits
```

- `tracker_factory` is called **inside** the worker thread, so MediaPipe lives on one thread only ([Section 2.1](DESIGN.md#21-component-diagram), rule 7).
- If `classifier.predict` or `tracker.detect` raises, the worker stores the exception in `error` and stops. It never swallows exceptions silently.
- The event queue is a `queue.Queue` and is unbounded in principle. In practice the debouncer emits at most about 3 events/s.

#### `actions.py` — provided by Evan, consumed by Evan (and tested by Evan)

```python
class Action(Enum):
    NOOP = "noop"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PAUSE = "pause"
    RESUME_AND_LEFT = "resume_and_left"
    RESUME_AND_RIGHT = "resume_and_right"
    RESTART = "restart"

def gesture_to_action(token: str, state: GameState) -> Action:
    """THE single gesture→action mapping (I6). Pure: no side effects, no I/O.
    token ∈ LABELS ∪ {NO_HAND}; raises ValueError for any other token."""

def apply_action(controller: GameController, action: Action) -> bool:
    """Execute an Action via GameController methods only. Returns True if the game changed."""
```

The mapping (implemented as a dict lookup, so the table **is** the code):

| token ↓ / state → | `PLAYING` | `COUNTDOWN` | `PAUSED` | `GAME_OVER` |
|---|---|---|---|---|
| `fist` | `MOVE_LEFT` | `MOVE_LEFT` | `RESUME_AND_LEFT` | `NOOP` |
| `v_sign` | `MOVE_RIGHT` | `MOVE_RIGHT` | `RESUME_AND_RIGHT` | `NOOP` |
| `open_palm` | `PAUSE` | `PAUSE` | `NOOP` | `RESTART` |
| `none` | `NOOP` | `NOOP` | `NOOP` | `NOOP` |
| `no_hand` | `PAUSE` | `PAUSE` | `NOOP` | `NOOP` |

`apply_action` for `RESUME_AND_LEFT` calls `controller.resume()` and then `controller.move_left()`. The move works because the state is now `COUNTDOWN`. That is the "resume play **and** perform that movement" rule (R6, LI.5).

**Why `GAME_OVER` + `open_palm` → `RESTART`:** the player has to be able to restart without the keyboard. Open palm is the only non-steering gesture, and the game is already stopped, so there is nothing for it to pause.

#### `smoothing.py` — provided by Evan

```python
class GestureSmoother:
    def __init__(self, window_s: float = SMOOTHING_WINDOW_S,
                 min_fraction: float = SMOOTHING_MIN_FRACTION,
                 min_frames: int = SMOOTHING_MIN_FRAMES) -> None: ...
    def update(self, token: str, t: float) -> str: ...   # t = Frame.t_capture; returns current STABLE token
    def reset(self) -> None: ...
    stable: str                                         # initial value: NO_HAND

class Debouncer:
    def __init__(self, cooldown_s: float = ACTION_COOLDOWN_S) -> None: ...
    def update(self, stable: str, now: float) -> str | None: ...   # event token or None
    def reset(self) -> None: ...
```

Exact algorithms are in [Section 6.2](game-logic.md#62-smoothing-gesturesmoother-evan) and [Section 6.3](game-logic.md#63-debounce-debouncer-evan). Constructors raise `ValueError` unless `window_s > 0`, `0.5 < min_fraction ≤ 1`, `min_frames ≥ 1` and `cooldown_s ≥ 0`. `update` raises `ValueError` if `t` goes backwards.

#### `latency.py` — provided by Evan

```python
@dataclass
class ActionTiming:
    action: Action
    token: str
    t_onset: float        # from GestureEvent (s)
    t_trigger: float      # from GestureEvent (s)
    t_landmarks: float    # from GestureEvent (s)
    t_predict: float      # from GestureEvent (s)
    t_action: float       # perf_counter() right after the GameController call returned, main thread (s)

class LatencyLogger:
    def __init__(self, out_path: Path | None) -> None: ...      # None → in-memory only
    def log_action(self, timing: ActionTiming) -> None: ...
    def summary(self) -> dict[str, float]: ...                  # median/p95/mean/max of each metric, ms
    def close(self) -> None: ...                                # flush CSV
```

`time.perf_counter()` is a single process-wide clock, so timestamps from the camera, worker and main threads can be compared directly.

#### `mock_classifier.py` — provided by Evan (Iteration 1, with the Lead)

```python
class KeyboardState:
    """Thread-safe snapshot of pressed keys. main.py writes it every frame; mocks read it."""
    def update(self, pressed: Sequence[bool]) -> None: ...   # main thread only
    def get(self) -> Sequence[bool]: ...                     # any thread

class MockClassifier:   # implements Classifier
    def __init__(self, key_state: Callable[[], Sequence[bool]]) -> None: ...
    def predict(self, features: np.ndarray) -> tuple[str, float]: ...

class MockCamera:        # same public API as Camera (read, latest, fps, close); black 640×480 Frames at CAMERA_FPS
class MockHandTracker:   # same public API as HandTracker; fixed valid HandLandmarks unless key H is held
    def __init__(self, key_state: Callable[[], Sequence[bool]]) -> None: ...
```

- `MockClassifier.predict` checks the shape exactly like the real one (`ValueError` if not (63,)) and then ignores the values. Keys **1** → `fist`, **2** → `v_sign`, **3** → `open_palm`, otherwise `none`. Confidence is `1.0`.
- **[added] `KeyboardState`:** the mocks run on the worker thread, and Pygame may only be touched from the main thread. So `main.py` copies `pygame.key.get_pressed()` into this snapshot every frame, and the mocks read the snapshot. Tests inject a plain function instead.
- Holding **H** in `MockHandTracker` simulates "no hand", so the auto-pause can be tested without a camera.

#### `main.py` — Evan

```python
def build_components(args: argparse.Namespace, keys: KeyboardState) -> tuple[
        Camera | MockCamera, Callable[[], HandTracker | MockHandTracker], Classifier]: ...
def run(args: argparse.Namespace) -> None: ...
def main(argv: list[str] | None = None) -> None:
    """--classifier {svm,mock} (default svm) --camera {webcam,mock} (default webcam)
       --log-latency (write results/latency_<YYYYmmdd-HHMMSS>.csv) --no-preview"""
```

- If `GestureClassifier` fails to load, `main` prints the error and suggests `--classifier mock`. It does **not** silently fall back, because a silent fallback could invalidate measurements.
- If `worker.error` is set, `main` shows it in the window title and on the console, then exits cleanly. It stops the worker and closes the camera.

### 4.7 Provider / consumer summary

| Interface | Provider | Consumers |
|---|---|---|
| `config.*` | Lead | everyone |
| `Camera`, `Frame` | Triet | Shobita (`collect`), Evan (`worker`, `main`) |
| `HandTracker`, `HandLandmarks` | Triet | Shobita (`collect`, `dataset`), Evan (`worker`) |
| `preprocess`, `row_to_landmarks` | Triet | Shobita (`dataset` → training data), Evan (`worker`, live) |
| `RecordingSession`, `collect.py` tool | Shobita | everyone (recording sessions) |
| CSV files `data/raw/*.csv` | Shobita (tool + sessions) + everyone (recording) | Shobita (`dataset`) |
| `Dataset`, `load_dataset`, `quality_report` | Shobita | Shobita (`split`), Siri (via `load_split`) |
| `Splits`, `load_splits`, `load_split`, `data/splits.json` | Shobita | Siri (`train`, `evaluate`) |
| `Car`, `Obstacle`, `collides`, `lane_center_x`, `Gameplay`, `draw_world` | Uy | Uy (inside `src/game`) |
| `GameState`, `StateMachine`, `draw_hud`, `draw_overlay` | Uy | Uy; `GameState` also Evan |
| `GameController` | Uy | Evan |
| `tune`, `train_model`, model bundle | Siri | Siri |
| `Classifier`, `GestureClassifier.predict` | Siri | Evan |
| `GestureWorker`, `gesture_to_action`, `apply_action`, `GestureSmoother`, `Debouncer`, `LatencyLogger`, `KeyboardState`, mocks | Evan | Evan |
