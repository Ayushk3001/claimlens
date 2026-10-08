import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "ClaimLens — AI-Powered Insurance Claims Assistant"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Environment & Host
    ENVIRONMENT: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://aicredits.in/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "data" / "sample_data"
    PARQUET_FILE: Path = PROJECT_ROOT / "data" / "sample_data" / "claims_sample.parquet"
    CSV_FILE: Path = PROJECT_ROOT / "data" / "sample_data" / "claims_sample.csv"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"
    FEEDBACK_FILE: Path = PROJECT_ROOT / "data" / "feedback_store.json"
    MODELS_DIR: Path = PROJECT_ROOT / "data" / "models"
    
    # Hugging Face Dataset
    HF_DATASET_ID: str = "ziadatalabs/FreeInsuranceClaims100M"
    HF_PARQUET_URL: str = (
        "https://huggingface.co/datasets/ziadatalabs/FreeInsuranceClaims100M/resolve/main/insurance_claims_100M.parquet"
    )
    SAMPLE_SIZE: int = 10000
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()
