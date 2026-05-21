import os
from dotenv import load_dotenv

load_dotenv()


def load_env(env_key: str):
    return os.getenv(env_key, "")

