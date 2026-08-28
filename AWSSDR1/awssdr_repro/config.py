from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent.parent

FS = 16000
N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 128
N_MELS = 40
FMIN = 400
FMAX = 6000
FIT_LEN = 40
FIT_HOP = 2
EPS = 1e-10
MIN_SAMPLES = N_FFT + (FIT_LEN - 1) * HOP_LENGTH

TRAIN_RIR_DIR = WORKSPACE_ROOT / "ALL_RIRS2_train"
TRAIN_LABEL_CSV = TRAIN_RIR_DIR / "T30_ground_truth.csv"
EARS_DIR = WORKSPACE_ROOT / "Speech" / "EARS"
ACE_RIRN_DIR = WORKSPACE_ROOT / "Eval_data" / "ACE_Corpus_RIRN_Single" / "Single"
AID_DIR = WORKSPACE_ROOT / "Eval_data" / "AID" / "wavs"
ACE_DEV_DIR = WORKSPACE_ROOT / "Eval_data" / "Dev" / "Speech" / "Single"
ACE_EVAL_DIR = WORKSPACE_ROOT / "Eval_data" / "Eval" / "Speech" / "Single"
ACE_DEV_CSV = ACE_DEV_DIR / "20260707T234354_test_gen_corpus_dataset_results.csv"
ACE_EVAL_CSV = ACE_EVAL_DIR / "20260707T214531_test_gen_corpus_dataset_results.csv"

GENERATED_DIR = REPO_ROOT / "generated_features"
REPORTS_DIR = REPO_ROOT / "reports" / "AWSSDR_T30_EARS_NOISY"

TRAIN_FEATURE_DIR = GENERATED_DIR / "data-train-t30-ears-noisy"
ACE_DEV_FEATURE_DIR = GENERATED_DIR / "data-ACEdev-fb-t30"
ACE_EVAL_FEATURE_DIR = GENERATED_DIR / "data-ACEeval-fb-t30"

CHECKPOINT_ID = "AWSSDR_T30_EARS_NOISY"

SEED = 42
K_UTT_PER_RIR = 34
NOISE_TYPES = ("Ambient", "Fan", "Babble")
SNRS_DB = (0, 10, 20)

NOISE_ROOMS = ("Office_1", "Building_Lobby")
NOISE_SEG_S = 10.0
N_BABBLE_TALKERS = (4, 7)
BABBLE_AMBIENT_DB = -15.0
N_TRANSIENTS = (1, 3)
TRANSIENT_GAIN_DB = (-5.0, 10.0)
TRANSIENT_MIC = "SH_MKH800"
TRANSIENT_CLASSES = (
    "foot_steps",
    "cough",
    "paper",
    "book",
    "keys",
    "pen_case",
    "scissors",
    "rubber_band",
    "roll_stool",
    "zipper",
    "snapping",
    "clapping",
)
STFT_NPERSEG = 1024

TRAINING = {
    "batch_size": 16,
    "lr": 1e-3,
    "fc_dim": 512,
    "max_epochs": 100,
    "decay_epoch": 50,
    "decay_lr": 0.99,
    "eval_check": 2,
    "update_period": 16,
    "weight_decay": 1e-3,
    "grad_clip": 1.0,
    "num_workers": 4,
}
