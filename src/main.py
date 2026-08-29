"""
Yerel Akademik Doküman Asistanı - CLI Arayüzü

Komut satırından kullanım için ana giriş noktası.
Streamlit arayüzüne alternatif olarak terminalden çalıştırılabilir.

Kullanım:
    python main.py --help
    python main.py chat
    python main.py upload dosya.pdf
    python main.py arxiv --query "machine learning" --limit 10
    python main.py status
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    APP_TITLE,
    APP_VERSION,
    DOCUMENTS_DIR,
    OLLAMA_LLM_MODEL,
    ARXIV_MAX_RESULTS,
)
from document_loader import DocumentLoader, load_documents
from arxiv_fetcher import ArxivFetcher
from vectorstore import VectorStoreManager
from llm import check_ollama_status, get_recommended_model
from rag_chain import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner():
    """Uygulama başlığını yazdır."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║           📚 YEREL AKADEMİK DOKÜMAN ASİSTANI v{APP_VERSION}          ║
╠══════════════════════════════════════════════════════════════╣
║  LangChain + FAISS + Ollama ile Yerel RAG Sistemi           ║
║  Verileriniz dışarı çıkmaz, tamamen yerel çalışır           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def cmd_status(args):
    """Sistem durumunu göster."""
    print("\n=== 🔧 Sistem Durumu ===\n")

    # Ollama durumu
    ollama = check_ollama_status()
    print(f"Ollama:")
    print(f"  Durum: {'✅ Çalışıyor' if ollama['status'] == 'running' else '❌ Çalışmıyor'}")
    print(f"  URL: {ollama['url']}")
    if ollama['available_models']:
        print(f"  Modeller: {', '.join(ollama['available_models'])}")
    else:
        print(f"  Modeller: Henüz indirilmiş model yok")
        print(f"  İndirmek için: ollama pull {OLLAMA_LLM_MODEL}")

    # Embedding
    print(f"\nEmbedding:")
    try:
        from embedding import get_embedding_model, get_embedding_dimension
        model = get_embedding_model()
        dim = get_embedding_dimension()
        print(f"  Model: ✅ Yüklü")
        print(f"  Boyut: {dim}d")
    except Exception as e:
        print(f"  Model: ❌ Hata - {e}")

    # Vektör veritabanı
    print(f"\nVektör Veritabanı:")
    manager = VectorStoreManager()
    doc_count = manager.get_document_count()
    print(f"  Doküman sayısı: {doc_count}")
    indices = manager.list_indices()
    if indices:
        print(f"  Index'ler: {', '.join(indices)}")

    # Önerilen model
    print(f"\nÖnerilen Model (8GB VRAM): {get_recommended_model(8.0)}")


def cmd_chat(args):
    """Interaktif sohbet modu."""
    print("\n=== 💬 Akademik Soru-Cevap ===\n")
    print("Çıkmak için 'quit', 'exit' veya 'q' yazın.\n")

    # Vektör veritabanı kontrolü
    manager = VectorStoreManager()
    if manager.vectorstore is None:
        print("❌ Vektör veritabanı boş!")
        print("Önce doküman ekleyin:")
        print(f"  python main.py upload {DOCUMENTS_DIR}/makaleniz.pdf")
        print(f"  python main.py arxiv --query 'machine learning'")
        return

    # RAG Pipeline oluştur
    model = args.model or OLLAMA_LLM_MODEL
    rag = RAGPipeline(
        vectorstore_manager=manager,
        llm_model=model,
        retriever_k=args.top_k,
    )

    print(f"Model: {model}")
    print(f"Doküman sayısı: {manager.get_document_count()}")
    print("-" * 50)

    while True:
        try:
            question = input("\n📝 Sorunuz: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Görüşmek üzere! 👋")
                break

            if not question:
                continue

            # Soru sor
            print("\n🤔 Düşünüyor...")
            response = rag.query(question)

            # Yanıtı göster
            print(f"\n📖 Yanıt:\n{response.answer}")
            print(f"\n📊 Getirilen doküman: {response.documents_retrieved}")

            # Kaynakları göster
            if response.sources:
                print(f"\n📚 Kaynaklar:")
                for i, src in enumerate(response.sources, 1):
                    print(f"  [{i}] {src['title']}")
                    print(f"      Yazar: {src['authors']}")

        except KeyboardInterrupt:
            print("\n\nGörüşmek üzere! 👋")
            break
        except Exception as e:
            logger.error(f"Hata: {e}")
            print(f"❌ Hata: {e}")


def cmd_upload(args):
    """Dosya yükle ve indeksle."""
    print(f"\n=== 📁 Dosya Yükleme ===\n")

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Dosya bulunamadı: {file_path}")
        return

    print(f"Dosya: {file_path.name}")
    print(f"Boyut: {file_path.stat().st_size / 1024:.1f} KB")

    # Yükle ve chunk'la
    print("\n📖 Dosya okunuyor...")
    loader = DocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = loader.load_and_split(file_path)

    if not chunks:
        print("❌ Dosyadan içerik çıkarılamadı!")
        return

    print(f"✅ {len(chunks)} chunk oluşturuldu")

    # Vektör veritabanına ekle
    print("\n🔢 Vektör veritabanına ekleniyor...")
    manager = VectorStoreManager()

    if manager.vectorstore is None:
        manager.create_from_documents(chunks)
    else:
        manager.add_documents(chunks)

    print(f"✅ İndekslendi! Toplam doküman: {manager.get_document_count()}")


def cmd_index_dir(args):
    """Bir dizindeki tüm dosyaları indeksle."""
    print(f"\n=== 📂 Dizin İndeksleme ===\n")

    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"❌ Dizin bulunamadı: {dir_path}")
        return

    print(f"Dizin: {dir_path}")
    print(f"Recursive: {args.recursive}")

    # Dosyaları yükle
    print("\n📖 Dosyalar okunuyor...")
    loader = DocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = loader.load_and_split(dir_path, recursive=args.recursive)

    if not chunks:
        print("❌ İşlenecek dosya bulunamadı!")
        return

    print(f"✅ {len(chunks)} chunk oluşturuldu")

    # Vektör veritabanına ekle
    print("\n🔢 Vektör veritabanına ekleniyor...")
    manager = VectorStoreManager()
    manager.create_from_documents(chunks)

    print(f"✅ İndekslendi! Toplam doküman: {manager.get_document_count()}")


def cmd_arxiv(args):
    """arXiv'den makale çek."""
    print(f"\n=== 🔬 arXiv Makale Çekme ===\n")

    query = args.query or "all:machine learning"
    limit = args.limit or ARXIV_MAX_RESULTS

    print(f"Sorgu: {query}")
    print(f"Limit: {limit}")

    # Makaleleri çek
    print("\n🔍 arXiv'de aranıyor...")
    fetcher = ArxivFetcher(max_results=limit)
    articles = fetcher.search(query, sort_by=args.sort_by)

    if not articles:
        print("❌ Makale bulunamadı!")
        return

    print(f"\n✅ {len(articles)} makale bulundu!\n")

    # Makaleleri göster
    for i, article in enumerate(articles, 1):
        print(f"[{i}] {article.title}")
        print(f"    Yazarlar: {', '.join(article.authors[:3])}")
        print(f"    Kategori: {article.primary_category}")
        print(f"    Tarih: {article.published[:10]}")
        print(f"    URL: {article.arxiv_url}")
        if args.show_abstract:
            print(f"    Özet: {article.summary[:200]}...")
        print()

    # PDF indir
    if args.download:
        print(f"\n📥 PDF'ler indiriliyor...")
        for article in articles:
            fetcher.download_pdf(article)

    # Vektör veritabanına ekle
    if args.index:
        print(f"\n🔢 Vektör veritabanına ekleniyor...")
        documents = fetcher.articles_to_documents(articles)

        loader = DocumentLoader(
            chunk_size=args.chunk_size or 1000,
            chunk_overlap=args.chunk_overlap or 200,
        )
        chunks = loader.split_documents(documents)

        manager = VectorStoreManager()
        if manager.vectorstore is None:
            manager.create_from_documents(chunks)
        else:
            manager.add_documents(chunks)

        print(f"✅ {len(articles)} makale indekslendi! ({len(chunks)} chunk)")

    # Sonuçları kaydet
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(f"Başlık: {article.title}\n")
                f.write(f"Yazarlar: {', '.join(article.authors)}\n")
                f.write(f"Özet: {article.summary}\n")
                f.write(f"URL: {article.arxiv_url}\n")
                f.write(f"PDF: {article.pdf_url}\n")
                f.write("-" * 80 + "\n\n")

        print(f"\n💾 Sonuçlar kaydedildi: {output_path}")


def cmd_clear(args):
    """Vektör veritabanını temizle."""
    print("\n=== 🗑️ Vektör Veritabanını Temizle ===\n")

    manager = VectorStoreManager()
    doc_count = manager.get_document_count()

    if doc_count == 0:
        print("Vektör veritabanı zaten boş.")
        return

    confirm = input(f"{doc_count} doküman silinecek. Emin misiniz? (evet/hayır): ")

    if confirm.lower() in ['evet', 'e', 'yes', 'y']:
        manager.delete_index()
        print("✅ Vektör veritabanı temizlendi!")
    else:
        print("İşlem iptal edildi.")


def cmd_streamlit(args):
    """Streamlit arayüzünü başlat."""
    import subprocess

    print("\n=== 🚀 Streamlit Arayüzü Başlatılıyor ===\n")

    streamlit_file = Path(__file__).parent / "app.py"

    try:
        subprocess.run(
            ["streamlit", "run", str(streamlit_file)],
            check=True,
        )
    except FileNotFoundError:
        print("❌ Streamlit bulunamadı!")
        print("Kurulum: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit hatası: {e}")


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description=f"{APP_TITLE} v{APP_VERSION} - Yerel RAG Asistanı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Sistem durumunu kontrol et
  python main.py status

  # Dosya yükle
  python main.py upload makale.pdf

  # Dizin indeksle
  python main.py index-dir ./makaleler/

  # arXiv'den makale çek
  python main.py arxiv --query "all:transformer" --limit 20 --index

  # Sohbet modu
  python main.py chat

  # Streamlit arayüzü
  python main.py streamlit
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # Status komutu
    status_parser = subparsers.add_parser("status", help="Sistem durumunu göster")
    status_parser.set_defaults(func=cmd_status)

    # Chat komutu
    chat_parser = subparsers.add_parser("chat", help="Soru-cevap modu")
    chat_parser.add_argument("--model", default=None, help="Kullanılacak LLM modeli")
    chat_parser.add_argument("--top-k", type=int, default=5, help="Getirilecek doküman sayısı")
    chat_parser.set_defaults(func=cmd_chat)

    # Upload komutu
    upload_parser = subparsers.add_parser("upload", help="Dosya yükle ve indeksle")
    upload_parser.add_argument("file", help="Yüklenecek dosya yolu")
    upload_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk boyutu")
    upload_parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk örtüşmesi")
    upload_parser.set_defaults(func=cmd_upload)

    # Index-dir komutu
    index_parser = subparsers.add_parser("index-dir", help="Dizin indeksle")
    index_parser.add_argument("directory", help="İndekslenecek dizin")
    index_parser.add_argument("--recursive", action="store_true", default=True, help="Alt dizinleri de tara")
    index_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk boyutu")
    index_parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk örtüşmesi")
    index_parser.set_defaults(func=cmd_index_dir)

    # ArXiv komutu
    arxiv_parser = subparsers.add_parser("arxiv", help="arXiv'den makale çek")
    arxiv_parser.add_argument("--query", default=None, help="Arama sorgusu")
    arxiv_parser.add_argument("--limit", type=int, default=None, help="Maksimum sonuç sayısı")
    arxiv_parser.add_argument("--sort-by", default="relevance", choices=["relevance", "submittedDate", "lastUpdatedDate"])
    arxiv_parser.add_argument("--show-abstract", action="store_true", help="Özetleri göster")
    arxiv_parser.add_argument("--download", action="store_true", help="PDF'leri indir")
    arxiv_parser.add_argument("--index", action="store_true", help="Vektör DB'ye ekle")
    arxiv_parser.add_argument("--output", default=None, help="Sonuçları dosyaya kaydet")
    arxiv_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk boyutu (index ile)")
    arxiv_parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk örtüşmesi (index ile)")
    arxiv_parser.set_defaults(func=cmd_arxiv)

    # Clear komutu
    clear_parser = subparsers.add_parser("clear", help="Vektör veritabanını temizle")
    clear_parser.set_defaults(func=cmd_clear)

    # Streamlit komutu
    streamlit_parser = subparsers.add_parser("streamlit", help="Streamlit arayüzünü başlat")
    streamlit_parser.set_defaults(func=cmd_streamlit)

    # Argümanları parse et
    args = parser.parse_args()

    # Banner göster
    if args.command is None:
        print_banner()
        parser.print_help()
        return

    if args.command != "streamlit":
        print_banner()

    # Komutu çalıştır
    args.func(args)


if __name__ == "__main__":
    main()
