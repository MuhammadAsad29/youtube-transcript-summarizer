import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b-instruct-q4_K_M")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 3000))
OVERLAP = int(os.getenv("OVERLAP", 200))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
TOP_P = float(os.getenv("TOP_P", 0.9))
NUM_CTX = int(os.getenv("NUM_CTX", 4096))
NUM_THREAD = int(os.getenv("NUM_THREAD", os.cpu_count() or 4))
TIMEOUT = int(os.getenv("TIMEOUT", 120))
MAX_SUMMARY_WORDS = int(os.getenv("MAX_SUMMARY_WORDS", 200))
MIN_KEY_POINTS = int(os.getenv("MIN_KEY_POINTS", 5))
MAX_KEY_POINTS = int(os.getenv("MAX_KEY_POINTS", 10))