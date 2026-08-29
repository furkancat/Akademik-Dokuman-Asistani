# 📚 Yerel Akademik Doküman Asistanı (RAG Tabanlı)

> **LangChain ve FAISS ile vektör veritabanı oluşturup, 4-bit quantization ile yerel LLM entegrasyonu sağlayan; PDF ve arXiv makaleleri üzerinde kaynak bazlı soru-cevap (RAG) yapabilen doküman asistanı.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0%2B-orange)](https://pytorch.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen%202.5%207B-FF6C37)](https://ollama.com/)

---

## 🖼️ Arayüz

> ```markdown
> ![Streamlit Arayüzü](assets/screenshot.png)
> ```

---

## Özellikler

- 🔒 **Tamamen Yerel**: Verileriniz asla dışarı çıkmaz, internet bağlantısı gerektirmez
- 📄 **Çoklu Format**: PDF, TXT, DOCX dosyalarını destekler
- 🔬 **arXiv Entegrasyonu**: arXiv API'den canlı makale çekme ve indeksleme
- 📚 **Kaynak Gösterme**: Her yanıtta kaynak makaleyi belirtir
- 🤖 **Yerel LLM**: Ollama üzerinde Qwen/Llama/Mistral modelleri
- 🧠 **8GB VRAM Optimizasyonu**: 4-bit quantization ile düşük kaynakla çalışır
- ⚡ **Hızlı Embedding**: sentence-transformers/all-MiniLM-L6-v2 (sadece 80MB)
- 🎯 **FAISS Vektör Arama**: Alt milisaniyede benzerlik araması
- 💻 **Streamlit Arayüzü**: Kullanıcı dostu web arayüzü
- 🖥️ **CLI Arayüzü**: Komut satırından tam kontrol

---

## Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    KULLANICI ARAYÜZÜ                         │
│              (Streamlit / CLI)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              RAG PIPELINE (LangChain)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Soru        │  │  Vektör      │  │  LLM (Ollama)    │  │
│  │  Analizi     │->│  Arama       │->│  Yanıt Üretimi   │  │
│  │              │  │  (FAISS)     │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼────────┐ ┌──▼──────────────┐
│  Doküman     │ │  arXiv API  │ │  Yerel LLM      │
│  Yükleme     │ │  (XML)      │ │  (Ollama)       │
│  (PDF/TXT)   │ │  feedparser │ │  Q4_K_M         │
└──────────────┘ └─────────────┘ └─────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│           EMBEDDING (sentence-transformers)                 │
│           all-MiniLM-L6-v2 (80MB, CPU/GPU)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com/download) kurulu ve çalışıyor olmalı
- (Opsiyonel) NVIDIA GPU + CUDA 12.1

### 2. Ollama Modelini İndir

```bash
# Qwen 2.5 7B (~5GB VRAM, çok dilli, Türkçe performansı yüksek)
ollama pull qwen2.5:7b
```

### 3. Kurulum

```bash
# 1. Repoyu klonla
git clone <repo-url>
cd rag-akademik-asistani

# 2. Sanal ortam oluştur
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Çevre değişkenlerini ayarla
cp .env.example .env
```

> **Not:** `requirements.txt` içindeki PyTorch sürümü sisteminizdeki CUDA sürümüne göre ayarlanmalıdır. Detaylı kurulum için `.env.example` dosyasındaki yorumları inceleyin.

### 4. Çalıştırma

**Streamlit Arayüzü (Tavsiye Edilen):**

```bash
cd src
streamlit run streamlit_app.py
# Tarayıcıda otomatik açılır: http://localhost:8501
```

**CLI Arayüzü:**

```bash
cd src

# Sistem durumunu kontrol et
python main.py status

# Dosya yükle
python main.py upload makale.pdf

# arXiv'den makale çek
python main.py arxiv --query "all:transformer" --limit 20 --index

# Soru-cevap modu
python main.py chat
```

---

## Kullanım

### Streamlit Arayüzü

1. **Dosya Yükle** sekmesinden PDF/TXT/DOCX dosyalarınızı yükleyin
2. **arXiv** sekmesinden makale arayıp ekleyin
3. **Soru-Cevap** sekmesinde sorularınızı sorun
4. Yanıtlar kaynaklarıyla birlikte gelecektir

### Python API ile Kullanım

```python
from src.rag_chain import RAGPipeline
from src.document_loader import load_documents

# 1. Dokümanları yükle
docs = load_documents("./makaleler/")

# 2. RAG pipeline oluştur
rag = RAGPipeline()
rag.add_documents(docs)

# 3. Soru sor
response = rag.query("Transformer mimarisi nedir?")
print(response.answer)
print(response.format_sources())
```

---

## Teknik Detaylar

### VRAM Optimizasyonu (8GB)

| Bileşen | Bellek Kullanımı |
|---------|-----------------|
| Qwen 2.5 7B (Q4_K_M) | ~5.0 GB |
| Embedding Modeli (CPU) | ~0 MB (VRAM'de değil) |
| FAISS Vektör Arama | ~100-500 MB |
| KV Cache (4096 ctx) | ~1.1 GB |
| **Toplam** | **~6.5-7 GB** |

### Embedding Modelleri

| Model | Boyut | Boyut | Hız | Kalite |
|-------|-------|-------|-----|--------|
| `all-MiniLM-L6-v2` | 80 MB | 384d | Çok hızlı | İyi |
| `all-mpnet-base-v2` | 420 MB | 768d | Hızlı | Çok iyi |
| `multi-qa-MiniLM-L6-cos-v1` | 80 MB | 384d | Çok hızlı | QA için optimize |

### Desteklenen LLM Modelleri

| Model | VRAM | Özellikler |
|-------|------|-----------|
| `qwen2.5:7b` | ~5.0 GB | Çok dilli, Türkçe performansı yüksek, varsayılan |
| `llama3.1:8b` | ~5.5 GB | Genel amaçlı, çok yönlü |
| `mistral:7b` | ~5.0 GB | Hızlı, verimli |
| `llama3.2:3b` | ~3.0 GB | Düşük VRAM için |

---

## Proje Yapısı

```
rag-akademik-asistani/
├── src/
│   ├── config.py              # Yapılandırma ve sabitler
│   ├── document_loader.py     # PDF/TXT/DOCX okuma ve chunk\'leme
│   ├── arxiv_fetcher.py       # arXiv API'den makale çekme
│   ├── embedding.py           # sentence-transformers embedding
│   ├── vectorstore.py         # FAISS vektör veritabanı yönetimi
│   ├── llm.py                 # Ollama LLM entegrasyonu
│   ├── rag_chain.py           # RAG pipeline ve soru-cevap
│   ├── streamlit_app.py       # Streamlit web arayüzü
│   └── main.py                # CLI arayüzü
├── tests/                     # Pytest testleri
├── data/
│   ├── documents/             # Yüklenen dosyalar (yerel)
│   └── vectorstores/          # FAISS index\'leri (yerel)
├── assets/
│   └── screenshot.png         # Arayüz fotoğrafı (tek fotoğraf burada)
├── requirements.txt           # Python bağımlılıkları
├── .env.example               # Çevre değişkenleri şablonu
├── LICENSE                    # Apache 2.0
└── README.md                  # Bu dosya
```

---

## CLI Komutları

```bash
python main.py [komut] [seçenekler]

KOMUTLAR:
  status        Sistem durumunu kontrol et
  chat          İnteraktif soru-cevap modu
  upload        Dosya yükle ve indeksle
  index-dir     Tüm dizini indeksle
  arxiv         arXiv'den makale çek
  clear         Vektör veritabanını temizle
  streamlit     Web arayüzünü başlat

ÖRNEKLER:
  python main.py status
  python main.py upload makale.pdf
  python main.py index-dir ./makaleler/ --recursive
  python main.py arxiv --query "all:transformer" --limit 20 --index
  python main.py chat --model qwen2.5:7b --top-k 5
```

---

## Güvenlik ve Gizlilik

- ✅ **Verileriniz yerel kalır**: Hiçbir dosya buluta yüklenmez
- ✅ **LLM yerel çalışır**: Ollama internete bağlı olmadan çalışır
- ✅ **Embedding yerel çalışır**: Model indirildikten sonra internet gerektirmez
- ✅ **FAISS yerel çalışır**: Tüm vektör verileri yerel diskte
- ✅ **arXiv opsiyonel**: Sadece istediğinizde internete çıkar

---

## Performans

| İşlem | Süre (Tahmini) |
|-------|---------------|
| 100 sayfalık PDF yükleme | ~5-10 sn |
| Embedding oluşturma (1000 chunk) | ~10-30 sn (CPU) |
| Vektör arama | ~10-50 ms |
| LLM yanıtı (Qwen 2.5 7B) | ~5-15 sn |
| arXiv API sorgusu | ~3-10 sn |

---

## Test

```bash
# Tüm testleri çalıştır
pytest tests/ -v
```

---

**Kullanılan Teknolojiler:**
- LangChain (RAG pipeline, doküman işleme)
- FAISS (vektör veritabanı ve benzerlik araması)
- Ollama (yerel LLM çalıştırma, 4-bit quantization)
- sentence-transformers (embedding üretimi)
- Streamlit (web arayüzü)
- arXiv API (akademik veri toplama)

---

## Kaynaklar

- [LangChain Dokümantasyonu](https://python.langchain.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- [arXiv API](https://arxiv.org/help/api/)
- [PyTorch](https://pytorch.org/)

---

## Lisans

Apache License 2.0 - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---
