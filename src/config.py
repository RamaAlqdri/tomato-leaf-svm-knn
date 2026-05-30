"""Project configuration.

Change DATASET_DIR if your local dataset is stored somewhere else.
All relative paths are resolved from the project root.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Required by the project specification.
DATASET_DIR = "data/raw"

PROCESSED_DIR = "data/processed"
FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"

IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_IMAGES_PER_CLASS = 1000
CV_FOLDS = (5, 10, 20)
N_JOBS = -1

TARGET_CLASSES = [
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Mosaic Virus",
    "Yellow Leaf Curl Virus",
]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

RGB_HIST_BINS = 16
HSV_HIST_BINS = 16
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2, 3)
GLCM_ANGLES = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
LBP_RADIUS = 3
LBP_POINTS = 24
LBP_METHOD = "uniform"

SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

KNN_DEFAULT_K = 5
KNN_K_VALUES = (3, 5, 7, 9)
KNN_METRIC = "euclidean"

