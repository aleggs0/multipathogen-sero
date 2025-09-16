from pathlib import Path


# from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
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

OUTPUTS_DIR = PROJ_ROOT / "outputs"

STAN_DIR = SRC_DIR / "models" / "stan"


# If tqdm is installed, configure loguru with tqdm.write
