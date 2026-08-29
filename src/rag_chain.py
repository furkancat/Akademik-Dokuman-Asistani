"""
Yerel Akademik Doküman Asistanı - RAG Pipeline Modülü

LangChain ile Retrieval-Augmented Generation pipeline'ı.
Soru sor, ilgili dokümanları getir, LLM ile yanıt oluştur, kaynakları göster.
"""

import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import RETRIEVER_K, OLLAMA_LLM_MODEL
from embedding import get_embedding_model
from vectorstore import VectorStoreManager
from llm import get_llm, ACADEMIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RagResponse:
    """RAG yanıt veri yapısı."""
    answer: str
    sources: List[Dict]  # Kaynak dokümanlar
    query: str
    documents_retrieved: int
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> Dict:
        """Sözlüğe dönüştür."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "documents_retrieved": self.documents_retrieved,
            "confidence": self.confidence,
        }

    def format_sources(self) -> str:
        """Kaynakları formatlı string olarak döndür."""
        if not self.sources:
            return "Kaynak bulunamadı."

        lines = ["\n📚 **Kaynaklar:**\n"]
        for i, source in enumerate(self.sources, 1):
            lines.append(f"[{i}] **{source.get('title', 'Bilinmiyor')}**")
            lines.append(f"    Yazar: {source.get('authors', 'Bilinmiyor')}")
            lines.append(f"    Kaynak: {source.get('source', 'Bilinmiyor')}")
            if source.get('score'):
                lines.append(f"    Benzerlik: {source['score']:.4f}")
            lines.append("")

        return "\n".join(lines)


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) Pipeline."""

    # Özel RAG prompt şablonu
    RAG_PROMPT_TEMPLATE = """Aşağıdaki akademik doküman parçalarını kullanarak kullanıcının sorusunu Türkçe yanıtla.
SADECE sağlanan dokümanlardaki bilgilere dayanarak cevap ver. Dokümanlarda yoksa "Bu konuda sağlanan dokümanlarda yeterli bilgi bulunmuyor." de.

=== DOKÜMANLAR ===
{context}

=== SORU ===
{question}

=== TALİMATLAR ===
1. Cevabı SADECE yukarıdaki dokümanlara dayanarak ver
2. Her önemli bilgi için parantez içinde kaynak belirt: [Kaynak: makale başlığı]
3. Yanıtı maddeler halinde ve düzenli bir şekilde ver
4. Teknik terimleri açıkla
5. Emin değilsen "Bilmiyorum" de, uydurma
6. Cevabı Türkçe yaz

=== CEVAP ==="""

    def __init__(
        self,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        llm_model: str = OLLAMA_LLM_MODEL,
        retriever_k: int = RETRIEVER_K,
    ):
        self.vectorstore = vectorstore_manager or VectorStoreManager()
        self.llm = get_llm(model=llm_model)
        self.retriever_k = retriever_k

        # RAG Prompt
        self.prompt = PromptTemplate(
            template=self.RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        logger.info("RAG Pipeline başlatıldı")

    def _format_context(self, documents: List[Document]) -> str:
        """
        Dokümanları LLM için bağlam metnine dönüştür.

        Args:
            documents: Kaynak dokümanlar

        Returns:
            str: Formatlanmış bağlam
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            # Metadata'dan kaynak bilgisi al
            source = doc.metadata.get("source", "Bilinmiyor")
            title = doc.metadata.get("title", "")
            authors = doc.metadata.get("authors", "")
            filename = doc.metadata.get("filename", "")
            file_type = doc.metadata.get("file_type", "")

            # Kaynak başlığını belirle
            if title:
                source_header = f"[{i}] {title}"
                if authors:
                    source_header += f" - {authors}"
            elif filename:
                source_header = f"[{i}] {filename}"
            else:
                source_header = f"[{i}] {source}"

            context_parts.append(f"{source_header}\n{doc.page_content}\n")

        return "\n---\n".join(context_parts)

    def _extract_sources(self, documents: List[Document]) -> List[Dict]:
        """
        Dokümanlardan kaynak bilgilerini çıkar.

        Args:
            documents: Kaynak dokümanlar

        Returns:
            List[Dict]: Kaynak bilgileri
        """
        sources = []
        seen = set()

        for doc in documents:
            title = doc.metadata.get("title", "")
            source = doc.metadata.get("source", "")

            # Tekrarları önle
            key = f"{title}_{source}"
            if key in seen:
                continue
            seen.add(key)

            source_info = {
                "title": title or doc.metadata.get("filename", "Bilinmiyor"),
                "authors": doc.metadata.get("authors", "Bilinmiyor"),
                "source": source,
                "published": doc.metadata.get("published", ""),
                "primary_category": doc.metadata.get("primary_category", ""),
                "pdf_url": doc.metadata.get("pdf_url", ""),
                "file_type": doc.metadata.get("file_type", ""),
            }

            sources.append(source_info)

        return sources

    def _calculate_confidence(self, documents: List[Document], answer: str) -> str:
        """
        Yanıtın güven seviyesini hesapla.

        Args:
            documents: Getirilen dokümanlar
            answer: LLM yanıtı

        Returns:
            str: "high", "medium", "low"
        """
        # Belirsizlik ifadeleri
        uncertainty_phrases = [
            "bilmiyorum", "emin değilim", "yeterli bilgi yok",
            "sağlanan dokümanlarda", "belirtilmemiş", "açıklanmamış",
            "i don't know", "not sure", "not mentioned", "insufficient",
        ]

        answer_lower = answer.lower()

        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return "low"

        if len(documents) >= 3:
            return "high"
        elif len(documents) >= 1:
            return "medium"

        return "low"

    def query(self, question: str) -> RagResponse:
        """
        RAG pipeline ile soru sor.

        Args:
            question: Kullanıcı sorusu

        Returns:
            RagResponse: Yanıt ve kaynaklar
        """
        logger.info(f"Soru: {question}")

        # 1. Vektör veritabanını kontrol et
        if self.vectorstore.vectorstore is None:
            logger.error("Vektör veritabanı boş!")
            return RagResponse(
                answer="HATA: Vektör veritabanı boş. Lütfen önce doküman ekleyin.",
                sources=[],
                query=question,
                documents_retrieved=0,
                confidence="low",
            )

        # 2. Benzer dokümanları getir (Retrieval)
        logger.info("İlgili dokümanlar aranıyor...")
        retrieved_docs = self.vectorstore.search(question, k=self.retriever_k)

        if not retrieved_docs:
            logger.warning("Hiç doküman bulunamadı")
            return RagResponse(
                answer="Üzgünüm, bu soruyla ilgili doküman bulunamadı. "
                       "Lütfen farklı bir soru sorun veya daha fazla doküman ekleyin.",
                sources=[],
                query=question,
                documents_retrieved=0,
                confidence="low",
            )

        documents = [doc for doc, _ in retrieved_docs]
        logger.info(f"{len(documents)} doküman getirildi")

        # 3. Bağlamı formatla
        context = self._format_context(documents)

        # 4. LLM ile yanıt oluştur (Generation)
        logger.info("Yanıt oluşturuluyor...")

        try:
            # Prompt'u doldur
            formatted_prompt = self.prompt.format(
                context=context,
                question=question,
            )

            # LLM'den yanıt al
            response = self.llm.invoke(formatted_prompt)
            answer = response.content

        except Exception as e:
            logger.error(f"LLM yanıt oluşturma hatası: {e}")
            return RagResponse(
                answer=f"Yanıt oluşturulurken hata: {str(e)}",
                sources=[],
                query=question,
                documents_retrieved=len(documents),
                confidence="low",
            )

        # 5. Kaynakları çıkar
        sources = self._extract_sources(documents)

        # 6. Güven seviyesi
        confidence = self._calculate_confidence(documents, answer)

        logger.info(f"Yanıt oluşturuldu (confidence: {confidence})")

        return RagResponse(
            answer=answer,
            sources=sources,
            query=question,
            documents_retrieved=len(documents),
            confidence=confidence,
        )

    def query_stream(self, question: str):
        """
        RAG pipeline ile soru sor (stream yanıt).

        Args:
            question: Kullanıcı sorusu

        Yields:
            str: Kısmi yanıt parçaları
        """
        logger.info(f"Stream sorgusu: {question}")

        # 1. Vektör veritabanını kontrol et
        if self.vectorstore.vectorstore is None:
            yield "HATA: Vektör veritabanı boş. Lütfen önce doküman ekleyin."
            return

        # 2. Benzer dokümanları getir
        retrieved_docs = self.vectorstore.search(question, k=self.retriever_k)

        if not retrieved_docs:
            yield "Üzgünüm, bu soruyla ilgili doküman bulunamadı."
            return

        documents = [doc for doc, _ in retrieved_docs]
        context = self._format_context(documents)

        # 3. Stream yanıt
        formatted_prompt = self.prompt.format(
            context=context,
            question=question,
        )

        try:
            for chunk in self.llm.stream(formatted_prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Stream hatası: {e}")
            yield f"\n[Hata: {str(e)}]"

    def add_documents(self, documents: List[Document]) -> None:
        """
        Pipeline'a yeni dokümanlar ekle.

        Args:
            documents: Eklenecek dokümanlar
        """
        if not documents:
            logger.warning("Eklenecek doküman yok")
            return

        if self.vectorstore.vectorstore is None:
            logger.info("Yeni vektör veritabanı oluşturuluyor...")
            self.vectorstore.create_from_documents(documents)
        else:
            self.vectorstore.add_documents(documents)

        logger.info(f"{len(documents)} doküman eklendi")

    def get_stats(self) -> Dict:
        """
        Pipeline istatistiklerini döndür.

        Returns:
            Dict: İstatistikler
        """
        doc_count = self.vectorstore.get_document_count()

        return {
            "total_documents": doc_count,
            "retriever_k": self.retriever_k,
            "llm_model": self.llm.model,
            "has_vectorstore": self.vectorstore.vectorstore is not None,
        }


# === KOLAY KULLANIM FONKSİYONLARI ===

def create_rag_pipeline(
    index_name: str = "academic_docs",
    llm_model: str = OLLAMA_LLM_MODEL,
) -> RAGPipeline:
    """
    RAG pipeline oluştur (kolay kullanım).

    Args:
        index_name: Vektör veritabanı adı
        llm_model: LLM model adı

    Returns:
        RAGPipeline: Oluşturulan pipeline
    """
    from vectorstore import VectorStoreManager

    vectorstore = VectorStoreManager(index_name=index_name)
    return RAGPipeline(
        vectorstore_manager=vectorstore,
        llm_model=llm_model,
    )


def ask_question(question: str, pipeline: Optional[RAGPipeline] = None) -> RagResponse:
    """
    Soru sor (kolay kullanım).

    Args:
        question: Soru
        pipeline: Mevcut pipeline (None = yeni oluştur)

    Returns:
        RagResponse: Yanıt
    """
    if pipeline is None:
        pipeline = create_rag_pipeline()

    return pipeline.query(question)


# === DOĞRUDAN ÇALIŞTIRMA ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== RAG Pipeline Test ===\n")

    # Vektör veritabanı kontrolü
    from vectorstore import VectorStoreManager
    from document_loader import DocumentLoader

    manager = VectorStoreManager(index_name="test_rag")

    if manager.vectorstore is None:
        print("1. Test veritabanı oluşturuluyor...")

        # Test dokümanları
        loader = DocumentLoader()
        test_docs = [
            Document(
                page_content="Transformer mimarisi, 2017 yılında Vaswani et al. tarafından 'Attention Is All You Need' makalesinde tanıtılmıştır. "
                           "Öz-dikkat (self-attention) mekanizması kullanarak paralel işleme sağlar. "
                           "BERT, GPT ve T5 gibi modellerin temelini oluşturur.",
                metadata={
                    "source": "test_transformer.pdf",
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani et al.",
                    "file_type": "pdf",
                },
            ),
            Document(
                page_content="BERT (Bidirectional Encoder Representations from Transformers), Google tarafından 2018'de geliştirilmiştir. "
                           "Maskeli dil modeli (MLM) ve sonraki cümle tahmini (NSP) görevleriyle ön eğitim yapılır. "
                           "Doğal dil anlama (NLU) görevlerinde çığır açan başarılar elde etmiştir.",
                metadata={
                    "source": "test_bert.pdf",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": "Devlin et al.",
                    "file_type": "pdf",
                },
            ),
            Document(
                page_content="GPT (Generative Pre-trained Transformer), OpenAI tarafından geliştirilen bir dil modeli serisidir. "
                           "Tek yönlü (auto-regressive) yapısıyla metin üretiminde mükemmel sonuçlar verir. "
                           "GPT-3 175 milyar parametre içerir ve few-shot learning yeteneğine sahiptir.",
                metadata={
                    "source": "test_gpt.pdf",
                    "title": "Language Models are Few-Shot Learners",
                    "authors": "Brown et al.",
                    "file_type": "pdf",
                },
            ),
        ]

        # Chunk'la
        chunks = loader.split_documents(test_docs)
        manager.create_from_documents(chunks)

    # RAG Pipeline oluştur
    print("\n2. RAG Pipeline oluşturuluyor...")
    rag = RAGPipeline(vectorstore_manager=manager)

    # Sorular sor
    questions = [
        "Transformer mimarisi nedir?",
        "BERT ve GPT arasındaki fark nedir?",
        "Öz-dikkat mekanizması nasıl çalışır?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"SORU: {q}")
        print(f"{'='*60}")

        response = rag.query(q)
        print(f"\nYANIT:\n{response.answer}")
        print(f"\nGetirilen doküman: {response.documents_retrieved}")
        print(f"Güven: {response.confidence}")

        if response.sources:
            print(f"\nKAYNAKLAR:")
            for s in response.sources:
                print(f"  - {s['title']} ({s['authors']})")

    # Temizlik
    manager.delete_index()
    print("\n✅ RAG Pipeline testi tamamlandı!")
