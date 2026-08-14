from datetime import datetime
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = r"C:\Users\borgesd8828\OneDrive - ARCADIS\Dokumente\GitHub\Real-Estate-Valuation-ImmoScout24\apr20_rental_no_duplicates_for_python.csv"
CSV_SEP = ","

TARGET = "obj_totalRent"

BASE_FEATURES = [
    "obj_heatingType", "obj_ExclusiveExpose", "obj_regio1", "obj_balcony", "obj_cellar",
    "obj_hasKitchen", "obj_picturecount", "obj_lift", "obj_petsAllowed", "obj_condition",
    "obj_livingSpace", "obj_typeOfFlat", "obj_garden", "obj_barrierFree"
]

ENGINEERED_FEATURES = [
    "obj_energyType_cat", "obj_thermalChar_num", "obj_hasNumberOfFloorsInfo",
    "obj_numberOfFloors_num", "obj_hasParkingInfo", "obj_noParkSpaces_num",
    "obj_hasLastRefurbishInfo", "obj_yearsSinceLastRefurbish", "obj_buildingAge",
    "obj_yearsBetweenConstructionAndRefurbish"
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

BINARY_FEATURES = [
    "obj_ExclusiveExpose", "obj_balcony", "obj_cellar", "obj_hasKitchen", "obj_lift",
    "obj_garden", "obj_barrierFree", "obj_hasNumberOfFloorsInfo", "obj_hasParkingInfo",
    "obj_hasLastRefurbishInfo"
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VIF_SAMPLE_SIZE = 5000
RIDGE_ALPHAS = np.logspace(-3, 3, 50)
STEPWISE_P_THRESHOLD = 0.05
STEPWISE_MAX_ITER = 100

MISSING_THRESHOLD_SCENARIOS = [
    {"scenario_code": "N0", "max_missing_pct": 0.00},
    {"scenario_code": "N3", "max_missing_pct": 0.03},
    {"scenario_code": "N5", "max_missing_pct": 0.05},
]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_EXCEL = f"Linear_Model_Results_Thesis_Tables_{timestamp}.xlsx"
DOCX_OUTPUT = f"Linear_Model_Debug_Report_{timestamp}.docx"

# Assumption gate
ASSUMPTION_ALPHA = 0.05
DW_OK_MIN = 1.5
DW_OK_MAX = 2.5
MAX_ALLOWED_VIF = 10.0

# Regra: se >= 2 checks falharem, dispara remediação
FAILED_CHECKS_TRIGGER = 99
MAX_REMEDIATION_ROUNDS = 0

# Remediação (ligar/desligar)
ENABLE_TARGET_LOG = False
ENABLE_WINSORIZATION = False
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99