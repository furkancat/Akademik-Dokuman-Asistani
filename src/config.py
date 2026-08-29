"""
Yerel Akademik Doküman Asistanı - Yapılandırma Modülü
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle (varsa)
load_dotenv()

# === PROJE DİZİNLERİ ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"

# Dizinleri oluştur
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORES_DIR.mkdir(parents=True, exist_ok=True)

# === OLLAMA YAPILANDIRMASI ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

# Desteklenen modeller (VRAM kullanımına göre)
SUPPORTED_MODELS = {
    "qwen2.5:7b": {"name": "Qwen 2.5 7B", "vram_gb": 5.0, "default": False},
    "llama3.1:8b": {"name": "Llama 3.1 8B", "vram_gb": 5.5, "default": True},
    "llama3.1:8b-instruct-q4_K_M": {"name": "Llama 3.1 8B Q4_K_M", "vram_gb": 5.5, "default": False},
    "mistral:7b": {"name": "Mistral 7B", "vram_gb": 5.0, "default": False},
    "mistral:7b-instruct-q4_K_M": {"name": "Mistral 7B Instruct Q4_K_M", "vram_gb": 5.0, "default": False},
    "gemma2:9b": {"name": "Gemma 2 9B", "vram_gb": 6.0, "default": False},
    "phi3:14b": {"name": "Phi-3 14B", "vram_gb": 8.0, "default": False},
}

# === EMBEDDING YAPILANDIRMASI ===
# sentence-transformers modeli (yerel, internet gerektirmez)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"  # 80MB, CPU'da da çalışır
)
# Alternatif embedding modelleri:
# "sentence-transformers/all-MiniLM-L6-v2"     # 80MB  - Hızlı, hafif (ÖNERİLEN)
# "sentence-transformers/all-mpnet-base-v2"    # 420MB - Daha iyi kalite
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 420MB - Çok dilli

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" veya "cpu"

# === FAISS YAPILANDIRMASI ===
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "Flat")  # "Flat" veya "IVF"
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "100"))  # IVF için cluster sayısı

# === DOKÜMAN BÖLÜMLEME (CHUNKING) ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # Her bir chunk'ın karakter sayısı
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # Chunk'lar arası örtüşme

# === ARXIV API YAPILANDIRMASI ===
ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
ARXIV_DELAY_SECONDS = int(os.getenv("ARXIV_DELAY_SECONDS", "3"))  # Rate limiting

# === RAG YAPILANDIRMASI ===
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))  # Kaç doküman getirilecek
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.0"))

# === UYGULAMA YAPILANDIRMASI ===
APP_TITLE = "Yerel Akademik Doküman Asistanı"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
LangChain ve FAISS ile vektör veritabanı oluşturup, 4-bit quantization ile 
yerel LLM entegrasyonu sağlayan; PDF ve arXiv makaleleri üzerinde 
kaynak bazlı soru-cevap (RAG) yapabilen doküman asistanı.
"""

# === LOGGING ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
