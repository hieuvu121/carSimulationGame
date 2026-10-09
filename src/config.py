"""Central configuration. Every tunable number lives here — never hard-code these elsewhere."""
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data" / "raw"
SAMPLE_DIR: Path = ROOT_DIR / "data" / "sample"
SPLITS_PATH: Path = ROOT_DIR / "data" / "splits.json"
SAMPLE_SPLITS_PATH: Path = SAMPLE_DIR / "splits.json"
MODEL_PATH: Path = ROOT_DIR / "models" / "gesture_svm.joblib"
RESULTS_DIR: Path = ROOT_DIR / "results"
SEED: int = 218

# ---------- Labels ----------
LABEL_FIST, LABEL_V_SIGN, LABEL_OPEN_PALM, LABEL_NONE = "fist", "v_sign", "open_palm", "none"
LABELS: tuple[str, ...] = (LABEL_FIST, LABEL_V_SIGN, LABEL_OPEN_PALM, LABEL_NONE)
NO_HAND: str = "no_hand"                    # pipeline token only (not a classifier label)

# ---------- Camera (Triet) ----------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640                      # px
FRAME_HEIGHT: int = 480                     # px
CAMERA_FPS: int = 30                        # requested; actual may be lower (≈15 in dim light)
MIRROR_FRAME: bool = True
CAMERA_THREADED: bool = True                # capture in a background thread (default)
CAMERA_READ_TIMEOUT_S: float = 1.0          # s, Camera.read() waits this long for a new frame
CAMERA_MAX_FAILED_READS: int = 5

# ---------- MediaPipe (Triet) ----------
MP_MAX_NUM_HANDS: int = 1
MP_MODEL_COMPLEXITY: int = 0                # 0 = fastest model; switch to 1 only if accuracy demands it
MP_MIN_DETECTION_CONFIDENCE: float = 0.6
MP_MIN_TRACKING_CONFIDENCE: float = 0.5

# ---------- Landmarks / preprocessing (Triet) ----------
NUM_LANDMARKS: int = 21
NUM_COORDS: int = 3
NUM_FEATURES: int = NUM_LANDMARKS * NUM_COORDS   # 63
WRIST_IDX: int = 0
MIDDLE_MCP_IDX: int = 9
MIN_PALM_SIZE: float = 1e-6                 # in aspect-corrected normalised units
PREPROCESS_VERSION: int = 1

# ---------- Data collection & preparation (Shobita) ----------
LIGHTING_CONDITIONS: tuple[str, ...] = ("bright", "dim", "backlit")
DISTANCE_CONDITIONS: tuple[str, ...] = ("near", "far")
CONDITIONS: tuple[str, ...] = tuple(f"{l}_{d}" for l in LIGHTING_CONDITIONS for d in DISTANCE_CONDITIONS)
CORE_CONDITIONS: tuple[str, ...] = ("bright_near", "bright_far", "dim_near", "dim_far")
RECORD_SAMPLE_HZ: float = 10.0              # rows per second while recording is ON
RECORD_START_DELAY_S: float = 1.0           # s between pressing a label key and the first sample
RECORD_TARGET_PER_LABEL: int = 80           # rows per label per condition per user (auto-stop)
MIN_ROWS_PER_CELL: int = 60                 # quality report flags user×condition×label cells below this
OUT_OF_FRAME_MARGIN: float = 0.05           # landmark x/y outside [-margin, 1+margin] counts as out of frame
MAX_OUT_OF_FRAME_LANDMARKS: int = 5         # gesture rows with more out-of-frame landmarks are dropped
N_TEST_USERS: int = 2                       # users held out entirely as the test set
CSV_FEATURE_COLUMNS: tuple[str, ...] = tuple(f"f{i}" for i in range(NUM_FEATURES))
CSV_COLUMNS: tuple[str, ...] = ("user_id", "condition", "label", *CSV_FEATURE_COLUMNS, "handedness")

# ---------- Classifier (Siri) ----------
SVM_C: float = 10.0
SVM_GAMMA: float | str = "scale"
SVM_GRID_C: tuple[float, ...] = (1.0, 10.0, 100.0)
SVM_GRID_GAMMA: tuple[float | str, ...] = ("scale", 0.01, 0.1)
CONFIDENCE_THRESHOLD: float = 0.70

# ---------- Live pipeline (Evan) ----------
SMOOTHING_WINDOW_S: float = 0.25            # s of capture time in the majority-vote window
SMOOTHING_MIN_FRACTION: float = 0.6         # share of the window a token needs to become stable (> 0.5)
SMOOTHING_MIN_FRAMES: int = 3               # window never shrinks below this many frames (low-FPS guard)
ACTION_COOLDOWN_S: float = 0.30             # s between lane-change events
LOW_FPS_WARNING: float = 10.0               # HUD shows FPS in orange below this
SHOW_CAMERA_PREVIEW: bool = True
PREVIEW_SIZE: tuple[int, int] = (160, 120)  # px, drawn top-right below HUD

# ---------- Game window & timing (Uy) ----------
WINDOW_WIDTH: int = 480                     # px (3 lanes × 160 px)
WINDOW_HEIGHT: int = 720                    # px
HUD_HEIGHT: int = 48                        # px
TARGET_FPS: int = 60                        # Pygame render rate, independent of camera FPS
MAX_DT: float = 0.1                         # s, clamp per update

# ---------- Gameplay (Uy) ----------
NUM_LANES: int = 3
LANE_WIDTH: float = WINDOW_WIDTH / NUM_LANES   # 160 px
CAR_WIDTH: float = 70.0
CAR_HEIGHT: float = 110.0
CAR_Y: float = WINDOW_HEIGHT - CAR_HEIGHT - 30.0
OBSTACLE_WIDTH: float = 90.0
OBSTACLE_HEIGHT: float = 70.0
BASE_SPEED: float = 220.0                   # px/s
SPEED_INCREASE_PER_SECOND: float = 6.0      # px/s per second of play
MAX_SPEED: float = 650.0                    # px/s
SPAWN_GAP_PX: float = 300.0                 # vertical distance between obstacle rows
DOUBLE_OBSTACLE_PROB: float = 0.25
SCORE_PER_SECOND: float = 10.0

# ---------- Game state (Uy) ----------
RESUME_COUNTDOWN_S: float = 2.0
GRACE_PERIOD_S: float = 0.75
