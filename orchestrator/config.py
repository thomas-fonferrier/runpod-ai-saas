import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    RUNPOD_API_KEY: str = os.environ["RUNPOD_API_KEY"]
    FLUX_ENDPOINT_ID: str = os.environ["FLUX_ENDPOINT_ID"]
    LTXVIDEO_ENDPOINT_ID: str = os.environ["LTXVIDEO_ENDPOINT_ID"]
    XTTS_ENDPOINT_ID: str = os.environ["XTTS_ENDPOINT_ID"]
    POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))
    MAX_WAIT: float = float(os.getenv("MAX_WAIT_SECONDS", "300"))


settings = Settings()
