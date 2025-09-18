from pathlib import Path

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJ_ROOT / "src" / "multipathogen_sero"

# CMDSTAN_PATHS = [
#     "C:\\Users\\alexy\\.cmdstan\\RTools40\\mingw64\\bin",
#     "C:\\Users\\alexy\\.cmdstan\\RTools40\\usr\\bin",
#     "C:\\Users\\alexy\\.cmdstan\\cmdstan-2.36.0\\stan\\lib\\stan_math\\lib\\tbb"
# ]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


STAN_DIR = SRC_DIR / "models" / "stan"
HPC_OUTPUTS_DIR = PROJ_ROOT / "outputs" / "from_hpc"
LOCAL_OUTPUTS_DIR = PROJ_ROOT / "outputs" / "from_local"
HPC_MODEL_FITS_DIR = HPC_OUTPUTS_DIR / "model_fits"
LOCAL_MODEL_FITS_DIR = LOCAL_OUTPUTS_DIR / "model_fits"

# If tqdm is installed, configure loguru with tqdm.write
