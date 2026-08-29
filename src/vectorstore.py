"""
Yerel Akademik Doküman Asistanı - Vektör Veritabanı Modülü

FAISS kullanarak yerel vektör veritabanı yönetimi.
Tüm veriler yerel diskte saklanır, bulut bağlantısı gerektirmez.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import VECTORSTORES_DIR, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD
from embedding import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """FAISS vektör veritabanını yöneten sınıf."""

    def __init__(
        self,
        index_name: str = "academic_docs",
        embedding_model=None,
    ):
        self.index_name = index_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.index_path = VECTORSTORES_DIR / index_name
        self.vectorstore: Optional[FAISS] = None

        # Var olan index'i yükle
        if self.index_path.exists():
            self.load()

    def create_from_documents(
        self,
        documents: List[Document],
        save: bool = True,
    ) -> FAISS:
        """
        Dokümanlardan yeni FAISS index'i oluştur.

        Args:
            documents: Vektörize edilecek dokümanlar
            save: Oluşturulan index'i diske kaydet

        Returns:
            FAISS: Oluşturulan vektör veritabanı
        """
        if not documents:
            raise ValueError("Doküman listesi boş olamaz")

        logger.info(f"FAISS index oluşturuluyor: {len(documents)} doküman")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )

        logger.info(f"FAISS index oluşturuldu: {len(documents)} doküman")

        if save:
            self.save()

        return self.vectorstore

    def add_documents(
        self,
        documents: List[Document],
        save: bool = True,
    ) -> None:
        """
        Mevcut index'e yeni dokümanlar ekle.

        Args:
            documents: Eklenecek dokümanlar
            save: Güncellenen index'i diske kaydet
        """
        if not documents:
            logger.warning("Eklenecek doküman yok")
            return

        if self.vectorstore is None:
            logger.info("Mevcut index bulunamadı, yeni oluşturuluyor...")
            self.create_from_documents(documents, save=save)
            return

        logger.info(f"Index'e {len(documents)} doküman ekleniyor...")
        self.vectorstore.add_documents(documents)
        logger.info(f"Dokümanlar eklendi: {len(documents)} adet")

        if save:
            self.save()

    def load(self) -> Optional[FAISS]:
        """
        Diskten FAISS index'i yükle.

        Returns:
            FAISS: Yüklenen vektör veritabanı veya None
        """
        if not self.index_path.exists():
            logger.warning(f"Index dosyası bulunamadı: {self.index_path}")
            return None

        try:
            logger.info(f"FAISS index yükleniyor: {self.index_path}")
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

            # Doküman sayısını kontrol et
            doc_count = self.vectorstore.index.ntotal
            logger.info(f"FAISS index yüklendi: {doc_count} vektör")
            return self.vectorstore

        except Exception as e:
            logger.error(f"Index yükleme hatası: {e}")
            return None

    def save(self) -> None:
        """FAISS index'ini diske kaydet."""
        if self.vectorstore is None:
            logger.warning("Kaydedilecek index yok")
            return

        self.index_path.mkdir(parents=True, exist_ok=True)

        try:
            self.vectorstore.save_local(str(self.index_path))
            doc_count = self.vectorstore.index.ntotal
            logger.info(f"FAISS index kaydedildi: {self.index_path} ({doc_count} vektör)")
        except Exception as e:
            logger.error(f"Index kaydetme hatası: {e}")

    def search(
        self,
        query: str,
        k: int = RETRIEVER_K,
        score_threshold: float = RETRIEVER_SCORE_THRESHOLD,
    ) -> List[Tuple[Document, float]]:
        """
        Sorguya göre en benzer dokümanları bul.

        Args:
            query: Arama sorgusu
            k: Döndürülecek sonuç sayısı
            score_threshold: Minimum benzerlik skoru

        Returns:
            List[Tuple[Document, float]]: (doküman, skor) çiftleri
        """
        if self.vectorstore is None:
            logger.error("Vektör veritabanı boş! Önce doküman ekleyin.")
            return []

        logger.debug(f"Arama: '{query}' (k={k})")

        # Benzerlik araması (kosinüs benzerliği)
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )

        # Skor filtresi uygula (FAISS'te düşük skor = daha yakın)
        if score_threshold > 0:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        logger.debug(f"{len(results)} sonuç bulundu")
        return results

    def get_retriever(self, k: int = RETRIEVER_K):
        """
        LangChain retriever nesnesi döndür.

        Args:
            k: Getirilecek doküman sayısı

        Returns:
            BaseRetriever: LangChain retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vektör veritabanı boş! Önce doküman ekleyin.")

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_document_count(self) -> int:
        """Index'teki toplam doküman sayısını döndür."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal

    def delete_index(self) -> None:
        """Index'i sil."""
        import shutil

        if self.index_path.exists():
            shutil.rmtree(self.index_path)
            logger.info(f"Index silindi: {self.index_path}")

        self.vectorstore = None

    def list_indices(self) -> List[str]:
        """Mevcut tüm index'leri listele."""
        if not VECTORSTORES_DIR.exists():
            return []

        indices = []
        for item in VECTORSTORES_DIR.iterdir():
            if item.is_dir() and (item / "index.faiss").exists():
                indices.append(item.name)

        return indices


# === KOLAY KULLANIM FONKSİYONLARI ===

def create_vectorstore(
    documents: List[Document],
    index_name: str = "academic_docs",
) -> VectorStoreManager:
    """
    Dokümanlardan vektör veritabanı oluştur (kolay kullanım).

    Args:
        documents: Vektörize edilecek dokümanlar
        index_name: Index adı

    Returns:
        VectorStoreManager: Oluşturulan yönetici
    """
    manager = VectorStoreManager(index_name=index_name)
    manager.create_from_documents(documents)
    return manager


def load_vectorstore(index_name: str = "academic_docs") -> Optional[VectorStoreManager]:
    """
    Mevcut vektör veritabanını yükle.

    Args:
        index_name: Index adı

    Returns:
        VectorStoreManager veya None
    """
    manager = VectorStoreManager(index_name=index_name)
    if manager.vectorstore is not None:
        return manager
    return None


# === DOĞRUDAN ÇALIŞTIRMA ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Vektör Veritabanı Modülü Test ===\n")

    # Test dokümanları
    test_docs = [
        Document(
            page_content="Makine öğrenmesi, bilgisayarların verilerden öğrenmesini sağlayan bir yapay zeka dalıdır.",
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="Derin öğrenme, yapay sinir ağları kullanarak karmaşık desenleri öğrenen bir makine öğrenmesi tekniğidir.",
            metadata={"source": "test2.txt"},
        ),
        Document(
            page_content="Doğal dil işleme, bilgisayarların insan dilini anlamasını ve işlemesini sağlayan bir alandır.",
            metadata={"source": "test3.txt"},
        ),
    ]

    # Vektör veritabanı oluştur
    print("1. Vektör veritabanı oluşturuluyor...")
    manager = VectorStoreManager(index_name="test_index")
    manager.create_from_documents(test_docs)

    # Arama testi
    print("\n2. Arama testi:")
    query = "yapay sinir ağları nedir"
    results = manager.search(query, k=2)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n   Sonuç {i} (skor: {score:.4f}):")
        print(f"   {doc.page_content[:100]}...")

    # Temizlik
    manager.delete_index()
    print("\n✅ Vektör veritabanı modülü çalışıyor!")
