"""Configurações centralizadas do pipeline de Risco de Crédito."""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_URL = os.getenv(
    "DATASET_URL",
    "https://drive.google.com/file/d/1R4WCMc_56lv3fMaDalUBAz5jSxeZOcwI/view?usp=sharing",
)
RAW_DATA_PATH = PROJECT_ROOT / os.getenv(
    "RAW_DATA_PATH", "data/raw/credit_risk_dataset.csv"
)
FIGURES_DIR = PROJECT_ROOT / os.getenv("FIGURES_DIR", "reports/figures")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.20"))
BALANCING_METHOD = os.getenv("BALANCING_METHOD", "rus").lower()

TARGET_COL = "loan_status"
FEATURE_ENGINEERED_COL = "comprometimento_renda"

CONTINUOUS_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    FEATURE_ENGINEERED_COL,
]

ORDINAL_GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
BINARY_DEFAULT_MAP = {"N": 0, "Y": 1}
NOMINAL_COLS = ["person_home_ownership", "loan_intent"]

MAX_VALID_AGE = 100
MAX_VALID_EMP_LENGTH = 60
INCOME_CLIP_QUANTILE = 0.99

KNN_K_VALUES = [3, 5, 7, 9, 11, 15]
TREE_MAX_DEPTHS = [3, 5, 7, 8, 10, None]
THRESHOLD_GRID = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
DEFAULT_LGD = 0.60
CV_FOLDS = 5
