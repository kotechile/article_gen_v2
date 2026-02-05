
from src.core.config import Settings
import os
from dotenv import load_dotenv

load_dotenv()
print(f"DEBUG: ALLOWED_ORIGINS env var: {os.getenv('ALLOWED_ORIGINS')}")

try:
    s = Settings()
    print("Settings loaded successfully!")
    print(f"Allowed Origins: {s.allowed_origins}")
except Exception as e:
    print(f"Error loading settings: {e}")
