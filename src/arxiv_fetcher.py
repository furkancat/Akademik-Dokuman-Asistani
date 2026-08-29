"""
Yerel Akademik Doküman Asistanı - arXiv Makale Çekme Modülü

arXiv API'den makale başlığı, özet ve PDF linki çeker.
feedparser kullanarak XML parse eder.

arXiv API Dökümantasyonu: https://arxiv.org/help/api/
API Endpoint: http://export.arxiv.org/api/query
"""

import logging
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from langchain_core.documents import Document

from config import (
    ARXIV_API_BASE_URL,
    ARXIV_MAX_RESULTS,
    ARXIV_DELAY_SECONDS,
    DOCUMENTS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class ArxivArticle:
    """arXiv makale veri yapısı."""
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: str
    arxiv_url: str
    primary_category: str
    categories: List[str]
    doi: Optional[str] = None
    journal_ref: Optional[str] = None

    def to_document(self) -> Document:
        """ArxivArticle'ı LangChain Document'a dönüştür."""
        content = f"Başlık: {self.title}\n\nÖzet: {self.summary}"

        metadata = {
            "source": self.arxiv_url,
            "title": self.title,
            "authors": ", ".join(self.authors),
            "published": self.published,
            "pdf_url": self.pdf_url,
            "primary_category": self.primary_category,
            "categories": ", ".join(self.categories),
            "doi": self.doi or "",
            "journal_ref": self.journal_ref or "",
            "file_type": "arxiv",
            "filename": f"arxiv_{self.arxiv_url.split('/')[-1]}.txt",
        }

        return Document(page_content=content, metadata=metadata)

    def __str__(self) -> str:
        return f"ArxivArticle(title='{self.title[:50]}...', authors={len(self.authors)}, cat={self.primary_category})"


class ArxivFetcher:
    """arXiv API'den makale çeken sınıf."""

    # arXiv kategorileri
    CATEGORIES = {
        "cs.AI": "Yapay Zeka",
        "cs.CL": "Hesaplamalı Dilbilim (NLP)",
        "cs.CV": "Bilgisayarlı Görü",
        "cs.LG": "Makine Öğrenmesi",
        "cs.RO": "Robotik",
        "cs.SE": "Yazılım Mühendisliği",
        "cs.DB": "Veritabanları",
        "cs.DC": "Dağıtık Hesaplama",
        "cs.DS": "Veri Yapıları",
        "cs.GT": "Oyun Teorisi",
        "cs.IR": "Bilgi Erişimi",
        "cs.MA": "Multi-Aracı Sistemler",
        "cs.NE": "Sinir ve Evrimsel Hesaplama",
        "cs.OS": "İşletim Sistemleri",
        "stat.ML": "İstatistik - Makine Öğrenmesi",
        "q-bio.NC": "Biyoloji - Sinir ve Bilişsel",
        "physics.comp-ph": "Hesaplamalı Fizik",
    }

    def __init__(
        self,
        base_url: str = ARXIV_API_BASE_URL,
        max_results: int = ARXIV_MAX_RESULTS,
        delay: int = ARXIV_DELAY_SECONDS,
    ):
        self.base_url = base_url
        self.max_results = max(1, min(max_results, 2000))  # arXiv limit: max 2000
        self.delay = delay
        logger.info(f"ArxivFetcher başlatıldı: max_results={max_results}, delay={delay}s")

    def _build_query_url(
        self,
        search_query: str,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> str:
        """
        arXiv API sorgu URL'si oluştur.

        Args:
            search_query: Arama sorgusu (arXiv query syntax)
            start: Başlangıç indeksi
            sort_by: Sıralama kriteri (relevance, lastUpdatedDate, submittedDate)
            sort_order: Sıralama yönü (ascending, descending)

        Returns:
            str: Tam URL
        """
        # Boşlukları URL encoding ile değiştir
        encoded_query = urllib.parse.quote(search_query)

        url = (
            f"{self.base_url}?"
            f"search_query={encoded_query}&"
            f"start={start}&"
            f"max_results={self.max_results}&"
            f"sortBy={sort_by}&"
            f"sortOrder={sort_order}"
        )

        return url

    def _parse_atom_feed(self, xml_content: str) -> List[ArxivArticle]:
        """
        Atom XML feed'i parse et.

        Args:
            xml_content: XML içeriği

        Returns:
            List[ArxivArticle]: Parse edilmiş makaleler
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            logger.error("xml.etree.ElementTree modülü bulunamadı")
            return []

        articles = []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse hatası: {e}")
            return []

        # Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Entry'leri bul
        entries = root.findall("atom:entry", ns)
        if not entries:
            # Namespace olmadan dene
            entries = root.findall(".//entry")

        logger.info(f"XML'de {len(entries)} makale bulundu")

        for entry in entries:
            try:
                article = self._parse_entry(entry, ns)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Makale parse hatası: {e}")
                continue

        return articles

    def _parse_entry(self, entry, ns: dict) -> Optional[ArxivArticle]:
        """Tek bir Atom entry'sini parse et."""
        def get_text(tag: str, default: str = "") -> str:
            elem = entry.find(f"atom:{tag}", ns)
            if elem is None:
                elem = entry.find(tag)
            return elem.text if elem is not None else default

        def get_attribute(tag: str, attr: str) -> str:
            elem = entry.find(f"atom:{tag}", ns)
            if elem is None:
                elem = entry.find(tag)
            return elem.get(attr, "") if elem is not None else ""

        # Başlık
        title = get_text("title").replace("\n", " ").strip()

        # Özet
        summary = get_text("summary").strip()

        # Yazarlar
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is None:
                name = author.find("name")
            if name is not None and name.text:
                authors.append(name.text)

        # Yayın tarihi
        published = get_text("published")

        # Linkler
        pdf_url = ""
        arxiv_url = ""
        for link in entry.findall("atom:link", ns):
            href = link.get("href", "")
            rel = link.get("rel", "")
            title_attr = link.get("title", "")

            if title_attr == "pdf":
                pdf_url = href
            elif rel == "alternate":
                arxiv_url = href

        # Eğer pdf_url bulunamazsa, arxiv_url'den oluştur
        if not pdf_url and arxiv_url:
            arxiv_id = arxiv_url.split("/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Kategoriler
        primary_category = ""
        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        # Primary category
        primary_elem = entry.find("atom:primary_category", ns)
        if primary_elem is None:
            primary_elem = entry.find("primary_category")
        if primary_elem is not None:
            primary_category = primary_elem.get("term", categories[0] if categories else "")
        elif categories:
            primary_category = categories[0]

        # DOI
        doi = get_text("doi")

        # Journal reference
        journal_ref = get_text("journal_ref")

        if not title or not summary:
            return None

        return ArxivArticle(
            title=title,
            summary=summary,
            authors=authors,
            published=published,
            pdf_url=pdf_url,
            arxiv_url=arxiv_url,
            primary_category=primary_category,
            categories=categories,
            doi=doi,
            journal_ref=journal_ref,
        )

    def search(
        self,
        query: str = "all:machine learning",
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[ArxivArticle]:
        """
        arXiv'de makale ara.

        Args:
            query: Arama sorgusu (arXiv query syntax)
                Örnekler:
                - "all:machine learning" (tüm alanlarda)
                - "ti:transformer" (başlıkta)
                - "au:Hinton" (yazara göre)
                - "cat:cs.LG" (kategoriye göre)
                - "all:neural network AND cat:cs.AI"
            start: Başlangıç indeksi (sayfalama için)
            sort_by: Sıralama kriteri
            sort_order: Sıralama yönü

        Returns:
            List[ArxivArticle]: Bulunan makaleler
        """
        url = self._build_query_url(query, start, sort_by, sort_order)
        logger.info(f"arXiv API sorgusu: {query}")

        try:
            # Rate limiting
            time.sleep(self.delay)

            # HTTP isteği
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "RAG-Akademik-Asistani/1.0 (Research Tool)",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                xml_content = response.read().decode("utf-8")

            articles = self._parse_atom_feed(xml_content)
            logger.info(f"{len(articles)} makale bulundu")

            return articles

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP hatası: {e.code} - {e.reason}")
            if e.code == 503:
                logger.warning("arXiv rate limit aşıldı, 10 saniye bekleniyor...")
                time.sleep(10)
                return self.search(query, start, sort_by, sort_order)
            return []

        except Exception as e:
            logger.error(f"arXiv arama hatası: {e}")
            return []

    def fetch_by_category(
        self,
        category: str = "cs.LG",
        max_results: Optional[int] = None,
    ) -> List[ArxivArticle]:
        """
        Belirli bir kategorideki makaleleri çek.

        Args:
            category: arXiv kategori kodu
            max_results: Maksimum sonuç sayısı

        Returns:
            List[ArxivArticle]: Makaleler
        """
        old_max = self.max_results
        if max_results:
            self.max_results = max_results

        query = f"cat:{category}"
        articles = self.search(query)

        self.max_results = old_max
        return articles

    def fetch_recent(
        self,
        days: int = 7,
        category: str = "cs.*",
    ) -> List[ArxivArticle]:
        """
        Son X gün içinde yayınlanan makaleleri çek.

        Args:
            days: Kaç gün geriye gidilecek
            category: Kategori filtresi

        Returns:
            List[ArxivArticle]: Makaleler
        """
        from datetime import datetime, timedelta

        # Tarih aralığını hesapla
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # arXiv tarih formatı: YYYYMMDDHHMMSS
        start_str = start_date.strftime("%Y%m%d%H%M%S")

        query = f"cat:{category} AND submittedDate:[{start_str} TO *]"
        articles = self.search(query, sort_by="submittedDate", sort_order="descending")

        return articles

    def download_pdf(self, article: ArxivArticle, output_dir: Path = DOCUMENTS_DIR) -> Optional[Path]:
        """
        Makalenin PDF'ini indir.

        Args:
            article: İndirilecek makale
            output_dir: Kaydedilecek dizin

        Returns:
            Path: İndirilen dosya yolu veya None
        """
        if not article.pdf_url:
            logger.warning(f"PDF URL bulunamadı: {article.title}")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dosya adını oluştur (güvenli)
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in article.title[:50])
        arxiv_id = article.arxiv_url.split("/")[-1]
        filename = f"{arxiv_id}_{safe_title}.pdf"
        filepath = output_dir / filename

        if filepath.exists():
            logger.info(f"PDF zaten mevcut: {filepath}")
            return filepath

        try:
            logger.info(f"PDF indiriliyor: {article.pdf_url}")
            time.sleep(self.delay)

            req = urllib.request.Request(
                article.pdf_url,
                headers={"User-Agent": "RAG-Akademik-Asistani/1.0"},
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                with open(filepath, "wb") as f:
                    f.write(response.read())

            logger.info(f"PDF indirildi: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"PDF indirme hatası: {e}")
            return None

    def articles_to_documents(self, articles: List[ArxivArticle]) -> List[Document]:
        """
        ArxivArticle listesini LangChain Document listesine dönüştür.

        Args:
            articles: Dönüştürülecek makaleler

        Returns:
            List[Document]: Dokümanlar
        """
        return [article.to_document() for article in articles]

    def get_category_name(self, category_code: str) -> str:
        """Kategori kodunu Türkçe isme dönüştür."""
        return self.CATEGORIES.get(category_code, category_code)


# === KOLAY KULLANIM FONKSİYONLARI ===

def fetch_arxiv_articles(
    query: str = "all:machine learning",
    max_results: int = 10,
) -> List[Document]:
    """
    arXiv'den makale çek ve LangChain Document'larına dönüştür.

    Args:
        query: Arama sorgusu
        max_results: Maksimum sonuç sayısı

    Returns:
        List[Document]: Makale dokümanları
    """
    fetcher = ArxivFetcher(max_results=max_results)
    articles = fetcher.search(query)
    return fetcher.articles_to_documents(articles)


def get_popular_categories() -> Dict[str, str]:
    """Popüler arXiv kategorilerini döndür."""
    return ArxivFetcher.CATEGORIES


# === DOĞRUDAN ÇALIŞTIRMA ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== arXiv Çekme Modülü Test ===\n")

    fetcher = ArxivFetcher(max_results=5, delay=1)

    # Test 1: Makine öğrenmesi ara
    print("1. 'machine learning' araması:")
    articles = fetcher.search("all:machine learning")

    for i, article in enumerate(articles[:3], 1):
        print(f"\n   [{i}] {article.title}")
        print(f"       Yazarlar: {', '.join(article.authors[:3])}")
        print(f"       Kategori: {article.primary_category}")
        print(f"       PDF: {article.pdf_url}")

    # Test 2: NLP makaleleri
    print("\n\n2. NLP makaleleri:")
    nlp_articles = fetcher.fetch_by_category("cs.CL", max_results=3)

    for i, article in enumerate(nlp_articles[:3], 1):
        print(f"\n   [{i}] {article.title[:60]}...")
        print(f"       Kategori: {article.primary_category}")

    # Test 3: Document dönüşümü
    if articles:
        print("\n\n3. Document dönüşümü testi:")
        docs = fetcher.articles_to_documents(articles[:2])
        for doc in docs:
            print(f"   - {doc.metadata['title'][:50]}...")
            print(f"     İçerik uzunluğu: {len(doc.page_content)} karakter")

    print("\n✅ arXiv modülü çalışıyor!")
