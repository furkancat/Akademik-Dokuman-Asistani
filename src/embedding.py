"""
Yerel Akademik Doküman Asistanı - Embedding Modülü

sentence-transformers kullanarak yerel embedding oluşturur.
İnternet bağlantısı gerektirmez (model önbelleğe alındıktan sonra).
"""

import logging
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

# Embedding modeli singleton (tekrar tekrar yüklenmesini önler)
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
    cache_folder: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    HuggingFace embedding modelini yükle (singleton pattern).

    Args:
        model_name: sentence-transformers model adı
        device: "cuda" veya "cpu"
        cache_folder: Model önbellek dizini

    Returns:
        HuggingFaceEmbeddings: Yüklenen embedding modeli
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Embedding modeli yükleniyor: {model_name} (device={device})")

        try:
            model_kwargs = {"device": device}
            encode_kwargs = {
                "normalize_embeddings": True,  # Kosinüs benzerliği için normalize et
                "batch_size": 32,
            }

            _embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=cache_folder,
            )

            logger.info(f"Embedding modeli yüklendi: {model_name}")
            logger.info(f"Embedding boyutu: {get_embedding_dimension()}")

        except Exception as e:
            logger.error(f"Embedding modeli yüklenirken hata: {e}")
            logger.info("CPU modunda tekrar deneniyor...")

            # GPU hatası durumunda CPU'ya düş
            _embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
                cache_folder=cache_folder,
            )

    return _embedding_model


def get_embedding_dimension() -> int:
    """Embedding vektör boyutunu döndür."""
    model = get_embedding_model()
    # Test embedding'i ile boyutu bul
    test_embedding = model.embed_query("test")
    return len(test_embedding)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Metin listesini embedding vektörlerine dönüştür.

    Args:
        texts: Embedding'e dönüştürülecek metin listesi

    Returns:
        List[List[float]]: Embedding vektörleri
    """
    if not texts:
        return []

    model = get_embedding_model()
    logger.debug(f"{len(texts)} metin embedding'e dönüştürülüyor...")

    embeddings = model.embed_documents(texts)
    logger.debug(f"Embedding oluşturuldu: {len(embeddings)} vektör")

    return embeddings


def embed_query(query: str) -> List[float]:
    """
    Tek bir sorguyu embedding vektörüne dönüştür.

    Args:
        query: Sorgu metni

    Returns:
        List[float]: Embedding vektörü
    """
    model = get_embedding_model()
    return model.embed_query(query)


def list_available_models() -> List[dict]:
    """
    Kullanılabilir embedding modellerini listele.

    Returns:
        Liste içinde model bilgileri
    """
    return [
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Hızlı ve hafif (80MB), CPU'da da çalışır",
            "dimensions": 384,
            "recommended": True,
        },
        {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "Daha iyi kalite (420MB), GPU önerilir",
            "dimensions": 768,
            "recommended": False,
        },
        {
            "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Çok dilli destek (420MB)",
            "dimensions": 384,
            "recommended": False,
        },
        {
            "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "description": "Soru-cevap görevleri için optimize (80MB)",
            "dimensions": 384,
            "recommended": False,
        },
    ]


# === DOĞRUDAN ÇALIŞTIRMA ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Embedding Modülü Test ===\n")

    # Model yükle
    print("1. Embedding modeli yükleniyor...")
    model = get_embedding_model()

    # Boyut kontrolü
    dim = get_embedding_dimension()
    print(f"   Embedding boyutu: {dim}")

    # Test
    print("\n2. Test embedding'i oluşturuluyor...")
    test_texts = [
        "Makine öğrenmesi, yapay zekanın bir alt dalıdır.",
        "Derin öğrenme, yapay sinir ağları kullanır.",
        "Python programlama dili popülerdir.",
    ]

    embeddings = embed_texts(test_texts)
    print(f"   {len(embeddings)} embedding oluşturuldu")
    print(f"   Her biri {len(embeddings[0])} boyutunda")

    # Benzerlik kontrolü
    print("\n3. Kosinüs benzerliği testi:")
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])  # ML - DL (benzer)
    sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])  # ML - Python (farklı)

    print(f"   'ML' vs 'DL' benzerliği: {sim_0_1:.4f} (yüksek beklenir)")
    print(f"   'ML' vs 'Python' benzerliği: {sim_0_2:.4f} (düşük beklenir)")

    print("\n✅ Embedding modülü çalışıyor!")
