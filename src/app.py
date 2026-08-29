"""
Yerel Akademik Doküman Asistanı - Streamlit Arayüzü

Özellikler:
- PDF/TXT/DOCX dosya yükleme
- arXiv'den makale çekme
- Vektör veritabanı yönetimi
- Kaynak gösteren soru-cevap
- Yerel çalışma (veri dışarı çıkmaz)
"""

import logging
import sys
from pathlib import Path

# Src dizinini Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import (
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
    DOCUMENTS_DIR,
    SUPPORTED_MODELS,
    OLLAMA_LLM_MODEL,
)
from document_loader import DocumentLoader
from arxiv_fetcher import ArxivFetcher
from vectorstore import VectorStoreManager
from embedding import get_embedding_model, list_available_models
from llm import check_ollama_status, get_model_info
from rag_chain import RAGPipeline

# === SAYFA YAPILANDIRMASI ===
st.set_page_config(
    page_title=f"{APP_TITLE} v{APP_VERSION}",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === OTURUM DURUMU (Session State) ===
def init_session_state():
    """Streamlit oturum durumunu başlat."""
    defaults = {
        "vectorstore": None,
        "rag_pipeline": None,
        "chat_history": [],
        "uploaded_files": [],
        "arxiv_articles": [],
        "processing": False,
        "vectorstore_name": "academic_docs",
        "selected_model": OLLAMA_LLM_MODEL,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retriever_k": 5,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# === YARDIMCI FONKSİYONLAR ===

def get_vectorstore_manager() -> VectorStoreManager:
    """Vektör veritabanı yöneticisini al veya oluştur."""
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStoreManager(
            index_name=st.session_state.vectorstore_name,
        )
    return st.session_state.vectorstore


def get_rag_pipeline() -> RAGPipeline:
    """RAG pipeline'ını al veya oluştur."""
    if st.session_state.rag_pipeline is None:
        vs_manager = get_vectorstore_manager()
        st.session_state.rag_pipeline = RAGPipeline(
            vectorstore_manager=vs_manager,
            llm_model=st.session_state.selected_model,
            retriever_k=st.session_state.retriever_k,
        )
    return st.session_state.rag_pipeline


def reset_pipeline():
    """Pipeline'ı sıfırla (yeni model veya ayar seçildiğinde)."""
    st.session_state.rag_pipeline = None


def process_uploaded_files(uploaded_files):
    """Yüklenen dosyaları işle."""
    if not uploaded_files:
        return 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    loader = DocumentLoader(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )

    all_chunks = []
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"İşleniyor: {uploaded_file.name} ({i+1}/{total_files})")

        # Dosyayı kaydet
        file_path = DOCUMENTS_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Yükle ve chunk'la
        chunks = loader.load_and_split(file_path)
        all_chunks.extend(chunks)

        progress_bar.progress((i + 1) / total_files)

    # Vektör veritabanına ekle
    if all_chunks:
        vs_manager = get_vectorstore_manager()

        if vs_manager.vectorstore is None:
            vs_manager.create_from_documents(all_chunks)
        else:
            vs_manager.add_documents(all_chunks)

        # Pipeline'ı güncelle
        reset_pipeline()

    progress_bar.empty()
    status_text.empty()

    return len(all_chunks)


def process_arxiv_articles(articles):
    """arXiv makalelerini vektör veritabanına ekle."""
    if not articles:
        return 0

    fetcher = ArxivFetcher()
    documents = fetcher.articles_to_documents(articles)

    # Chunk'la
    loader = DocumentLoader(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    chunks = loader.split_documents(documents)

    # Vektör veritabanına ekle
    if chunks:
        vs_manager = get_vectorstore_manager()

        if vs_manager.vectorstore is None:
            vs_manager.create_from_documents(chunks)
        else:
            vs_manager.add_documents(chunks)

        reset_pipeline()

    return len(chunks)


# === SIDEBAR (YAN MENÜ) ===
with st.sidebar:
    st.title("📚 Kontrol Paneli")
    st.markdown("---")

    # === SEKME 1: SİSTEM DURUMU ===
    with st.expander("🔧 Sistem Durumu", expanded=True):
        # Ollama durumu
        ollama_status = check_ollama_status()

        if ollama_status["status"] == "running":
            st.success("✅ Ollama çalışıyor")
            st.caption(f"URL: {ollama_status['url']}")

            if ollama_status["available_models"]:
                st.caption(f"Modeller: {', '.join(ollama_status['available_models'][:3])}")
        else:
            st.error("❌ Ollama çalışmıyor!")
            st.info("Terminalde çalıştır:\n```\nollama serve\n```")

        # Embedding durumu
        try:
            embedding_model = get_embedding_model()
            st.success("✅ Embedding modeli hazır")
        except Exception as e:
            st.error(f"❌ Embedding hatası: {e}")

        # Vektör veritabanı durumu
        vs_manager = get_vectorstore_manager()
        doc_count = vs_manager.get_document_count()
        if doc_count > 0:
            st.success(f"✅ {doc_count} doküman indekslendi")
        else:
            st.warning("⚠️ Henüz doküman yok")

    st.markdown("---")

    # === SEKME 2: MODEL AYARLARI ===
    with st.expander("🤖 Model Ayarları", expanded=True):
        # LLM Model seçimi
        available_models = ollama_status.get("available_models", [])

        if available_models:
            selected = st.selectbox(
                "LLM Modeli:",
                options=available_models,
                index=0 if OLLAMA_LLM_MODEL not in available_models else available_models.index(OLLAMA_LLM_MODEL),
                key="selected_model",
                on_change=reset_pipeline,
            )
        else:
            st.text_input(
                "LLM Modeli:",
                value=st.session_state.selected_model,
                key="selected_model",
                on_change=reset_pipeline,
            )
            st.info("Model indirmek için:\n```\nollama pull llama3.1:8b\n```")

        # Embedding modeli
        embedding_models = list_available_models()
        embedding_options = [m["name"] for m in embedding_models]
        st.selectbox(
            "Embedding Modeli:",
            options=embedding_options,
            index=0,
            disabled=True,  # Şimdilik değiştirilemez
        )
        st.caption(f"Boyut: {embedding_models[0]['dimensions']}d | {embedding_models[0]['description']}")

        # RAG parametreleri
        st.markdown("**RAG Parametreleri:**")
        st.slider(
            "Getirilecek Doküman (k):",
            min_value=1,
            max_value=10,
            value=st.session_state.retriever_k,
            key="retriever_k",
            on_change=reset_pipeline,
        )

    st.markdown("---")

    # === SEKME 3: GELİŞMİŞ AYARLAR ===
    with st.expander("⚙️ Gelişmiş Ayarlar", expanded=False):
        st.slider(
            "Chunk Boyutu:",
            min_value=200,
            max_value=2000,
            value=st.session_state.chunk_size,
            step=100,
            key="chunk_size",
        )
        st.slider(
            "Chunk Örtüşmesi:",
            min_value=0,
            max_value=500,
            value=st.session_state.chunk_overlap,
            step=50,
            key="chunk_overlap",
        )

        # Vektör veritabanı yönetimi
        st.markdown("**Vektör Veritabanı:****")
        indices = vs_manager.list_indices()
        if indices:
            st.caption(f"Mevcut index'ler: {', '.join(indices)}")
        else:
            st.caption("Henüz index yok")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Index'i Sil", use_container_width=True):
                vs_manager.delete_index()
                st.session_state.vectorstore = None
                reset_pipeline()
                st.session_state.chat_history = []
                st.success("Index silindi!")
                st.rerun()

        with col2:
            if st.button("🔄 Yenile", use_container_width=True):
                st.rerun()

    st.markdown("---")
    st.caption(f"v{APP_VERSION} | Yerel RAG Asistanı")


# === ANA İÇERİK ===
st.title(f"📚 {APP_TITLE}")
st.caption(APP_DESCRIPTION)
st.markdown("---")

# === SEKMELER ===
tab_chat, tab_upload, tab_arxiv, tab_sources = st.tabs([
    "💬 Soru-Cevap",
    "📁 Dosya Yükle",
    "🔬 arXiv",
    "📊 Kaynaklar",
])

# === SEKME 1: SORU-CEVAP ===
# === SEKME 1: SORU-CEVAP ===
with tab_chat:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Akademik Soru-Cevap")
        st.info("Yüklediğiniz dokümanlar ve arXiv makaleleri üzerinde soru sorun.")
    with col2:
        # Sohbeti Temizle butonunu GİRİŞ KUTUSUNUN ÜSTÜNE, sayfanın başına alıyoruz.
        if st.session_state.chat_history:
            if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Mesajların render edileceği sabit alan
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📚 Kaynaklar"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**[{i}] {src.get('title', 'Bilinmiyor')}**")
                            st.caption(f"Yazar: {src.get('authors', 'Bilinmiyor')}")
                            if src.get('source'):
                                st.caption(f"🔗 {src['source']}")

    # Giriş kutusu (Kodun en sonunda olduğu için her zaman ekranın en altında sabit kalacak)
    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    vs_manager = get_vectorstore_manager()
                    if vs_manager.vectorstore is None:
                        st.error("⚠️ Henüz doküman yok! Lütfen önce sekmeden doküman ekleyin.")
                    else:
                        rag = get_rag_pipeline()
                        with st.spinner("Düşünüyor..."):
                            response = rag.query(prompt)

                        st.markdown(response.answer)
                        
                        if response.sources:
                            with st.expander("📚 Kaynaklar"):
                                for i, src in enumerate(response.sources, 1):
                                    st.markdown(f"**[{i}] {src.get('title', 'Bilinmiyor')}**")
                                    st.caption(f"Yazar: {src.get('authors', 'Bilinmiyor')}")
                                    if src.get('source'):
                                        st.caption(f"🔗 {src['source']}")

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                        })
                except Exception as e:
                    st.error(f"Hata: {str(e)}")


# === SEKME 2: DOSYA YÜKLE ===
with tab_upload:
    st.subheader("Doküman Yükle")
    st.info("PDF, TXT veya DOCX dosyalarını yükleyin. Dosyalar yerel olarak işlenir, dışarı çıkmaz.")

    uploaded_files = st.file_uploader(
        "Dosya seçin (çoklu seçim yapabilirsiniz):",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} dosya seçildi:**")
        for f in uploaded_files:
            st.write(f"  - {f.name} ({f.size / 1024:.1f} KB)")

        if st.button("📥 Dosyaları İşle", type="primary", use_container_width=True):
            with st.spinner("Dosyalar işleniyor..."):
                chunk_count = process_uploaded_files(uploaded_files)

            if chunk_count > 0:
                st.success(f"✅ {len(uploaded_files)} dosya yüklendi, {chunk_count} chunk oluşturuldu!")
            else:
                st.warning("⚠️ İşlenecek dosya bulunamadı")

    # Mevcut dosyaları göster
    st.markdown("---")
    st.subheader("Mevcut Dosyalar")

    if DOCUMENTS_DIR.exists():
        files = list(DOCUMENTS_DIR.rglob("*"))
        doc_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.docx']]

        if doc_files:
            for f in doc_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {f.name}")
                with col2:
                    if st.button("🗑️", key=f"del_{f.name}"):
                        f.unlink()
                        st.success(f"{f.name} silindi")
                        st.rerun()
        else:
            st.info("Henüz dosya yok")
    else:
        st.info("Henüz dosya yok")


# === SEKME 3: ARXIV ===
with tab_arxiv:
    st.subheader("arXiv'den Makale Çek")
    st.info("arXiv API'sinden makale başlığı ve özet çekin. PDF indirme seçeneği de mevcut.")

    col1, col2 = st.columns([3, 1])

    with col1:
        arxiv_query = st.text_input(
            "Arama sorgusu:",
            value="all:machine learning",
            placeholder="Örn: all:transformer, ti:BERT, cat:cs.CL",
            help="arXiv query syntax'ı kullanın",
        )

    with col2:
        arxiv_max_results = st.number_input(
            "Maksimum sonuç:",
            min_value=1,
            max_value=100,
            value=10,
        )

    # Kategoriler
    fetcher = ArxivFetcher()
    categories = fetcher.CATEGORIES

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        selected_category = st.selectbox(
            "Veya kategori seçin:",
            options=[""] + list(categories.keys()),
            format_func=lambda x: f"{x} - {categories.get(x, x)}" if x else "Kategori seçin...",
        )

    with col_cat2:
        sort_by = st.selectbox(
            "Sıralama:",
            options=["relevance", "submittedDate", "lastUpdatedDate"],
            format_func=lambda x: {"relevance": "İlgi Düzeyi", "submittedDate": "Yayın Tarihi", "lastUpdatedDate": "Güncelleme Tarihi"}.get(x, x),
        )

    # Arama butonu
    if st.button("🔍 arXiv'de Ara", type="primary", use_container_width=True):
        with st.spinner("Makaleler aranıyor..."):
            if selected_category:
                query = f"cat:{selected_category}"
            else:
                query = arxiv_query

            fetcher = ArxivFetcher(max_results=arxiv_max_results)
            articles = fetcher.search(query, sort_by=sort_by)

            st.session_state.arxiv_articles = articles

        if articles:
            st.success(f"✅ {len(articles)} makale bulundu!")
        else:
            st.warning("⚠️ Makale bulunamadı")

    # Sonuçları göster
    if st.session_state.arxiv_articles:
        st.markdown("---")
        st.subheader(f"Sonuçlar ({len(st.session_state.arxiv_articles)} makale)")

        # Tümünü ekle butonu
        col_all, col_sel = st.columns(2)
        with col_all:
            if st.button("📥 Tümünü Vektör DB'ye Ekle", use_container_width=True):
                with st.spinner("Makaleler ekleniyor..."):
                    chunk_count = process_arxiv_articles(st.session_state.arxiv_articles)
                st.success(f"✅ {len(st.session_state.arxiv_articles)} makale eklendi ({chunk_count} chunk)")

        with col_sel:
            selected_articles = []

        # Makaleleri listele
        for i, article in enumerate(st.session_state.arxiv_articles):
            with st.container():
                col_info, col_actions = st.columns([4, 1])

                with col_info:
                    st.markdown(f"**{i+1}. {article.title}**")
                    st.caption(f"Yazarlar: {', '.join(article.authors[:3])}{' et al.' if len(article.authors) > 3 else ''}")
                    st.caption(f"Kategori: {article.primary_category} | Tarih: {article.published[:10]}")

                    with st.expander("Özet"):
                        st.write(article.summary)
                        st.markdown(f"[🔗 arXiv Sayfası]({article.arxiv_url})")
                        if article.pdf_url:
                            st.markdown(f"[📄 PDF İndir]({article.pdf_url})")

                with col_actions:
                    if st.button("➕ Ekle", key=f"add_arxiv_{i}", use_container_width=True):
                        with st.spinner("Ekleniyor..."):
                            chunk_count = process_arxiv_articles([article])
                        st.success(f"Eklendi! ({chunk_count} chunk)")

                st.markdown("---")


# === SEKME 4: KAYNAKLAR ===
with tab_sources:
    st.subheader("İndekslenmiş Kaynaklar")

    vs_manager = get_vectorstore_manager()
    doc_count = vs_manager.get_document_count()

    if doc_count == 0:
        st.info("Henüz indekslenmiş kaynak yok. Dosya yükleyin veya arXiv'den makale çekin.")
    else:
        st.write(f"**Toplam {doc_count} doküman indekslendi**")

        # İstatistikler
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Doküman", doc_count)
        with col2:
            st.metric("LLM Model", st.session_state.selected_model)
        with col3:
            try:
                from embedding import get_embedding_dimension
                st.metric("Embedding Boyutu", f"{get_embedding_dimension()}d")
            except:
                st.metric("Embedding Boyutu", "?")

        # Kaynak listesi
        st.markdown("---")
        st.subheader("Dosya Listesi")

        if DOCUMENTS_DIR.exists():
            files = [f for f in DOCUMENTS_DIR.rglob("*") if f.suffix.lower() in ['.pdf', '.txt', '.docx']]
            for f in files:
                st.write(f"📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")


# === ALT BİLGİ ===
st.markdown("---")
st.caption(
    "🔒 Tüm veriler yerel olarak işlenir. Hiçbir bilgi dışarı çıkmaz. "
    f"| [Ollama](https://ollama.com) | [arXiv](https://arxiv.org) | v{APP_VERSION}"
)
