import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Air-Gapped Sovereign AI Workbench"
    API_V1_STR: str = "/api/v1"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    WORKSPACE_DIR: Path = BASE_DIR / "data" / "workspace"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CONFIG_FILE: Path = BASE_DIR / "model.yaml"
    LOG_DIR: Path = BASE_DIR / "data" / "logs"
    SANDBOX_DIR: Path = BASE_DIR / "data" / "sandbox"

    # DB
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/app_state.db"
    VECTOR_DB_PATH: Path = BASE_DIR / "data" / "chroma_db"
    EMBEDDING_MODEL: str = "all-minilm"
    EMBEDDING_API_BASE: str = "http://localhost:11434"

    # Local speech-to-text (Hugging Face Whisper checkpoint)
    WHISPER_MODEL_ID: str = "openai/whisper-base"
    WHISPER_MODEL_DIR: Path = BASE_DIR / "models" / "whisper-base"
    WHISPER_DEVICE: str = "auto"
    WHISPER_MAX_AUDIO_SECONDS: int = 120
    WHISPER_MAX_AUDIO_MB: int = 20
    KNOWLEDGE_DIR: Path = BASE_DIR / "data" / "knowledge_documents"

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: set = {
        ".pdf",
        ".docx",
        ".doc",
        ".png",
        ".jpg",
        ".jpeg",
        ".txt",
        ".log",
        ".py",
        ".json",
        ".csv",
        ".xlsx",
        ".xlsm",
        ".xltx",
        ".xltm",
    }
    WORKSPACE_RETENTION_HOURS: int = 24

    def load_model_registry(self) -> dict:
        if not self.CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.CONFIG_FILE}"
            )

        with open(self.CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)


settings = Settings()

settings.WORKSPACE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.SANDBOX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.VECTOR_DB_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

settings.KNOWLEDGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.WHISPER_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
