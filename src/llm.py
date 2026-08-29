"""
Yerel Akademik Doküman Asistanı - LLM (Ollama) Modülü

Ollama üzerinden yerel LLM ile iletişim kurar.
İnternet bağlantısı gerektirmez (model indirildikten sonra).
"""

import logging
from typing import Optional, List, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    SUPPORTED_MODELS,
)

logger = logging.getLogger(__name__)

# LLM singleton
_llm_instance: Optional[ChatOllama] = None


def get_llm(
    model: str = OLLAMA_LLM_MODEL,
    temperature: float = OLLAMA_TEMPERATURE,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
    base_url: str = OLLAMA_BASE_URL,
) -> ChatOllama:
    """
    Ollama LLM nesnesini oluştur (singleton pattern).

    Args:
        model: Ollama model adı
        temperature: Yaratıcılık parametresi (0.0 = deterministik)
        num_ctx: Context pencere boyutu (token)
        num_predict: Maksimum üretilecek token sayısı
        base_url: Ollama API URL'si

    Returns:
        ChatOllama: Yapılandırılmış LLM nesnesi
    """
    global _llm_instance

    # Eğer farklı bir model istenirse, yeni instance oluştur
    if _llm_instance is not None and _llm_instance.model == model:
        return _llm_instance

    logger.info(f"LLM başlatılıyor: {model} (temp={temperature}, ctx={num_ctx})")

    try:
        _llm_instance = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            base_url=base_url,
            # Düşük temperature = daha tutarlı, akademik yanıtlar
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            # Sistem mesajı davranışını kontrol et
            system=None,
        )

        logger.info(f"LLM başlatıldı: {model}")
        return _llm_instance

    except Exception as e:
        logger.error(f"LLM başlatma hatası: {e}")
        logger.info("Lütfen Ollama'nın çalıştığından emin olun:")
        logger.info("  1. Ollama'yı başlat: ollama serve")
        logger.info(f"  2. Modeli indir: ollama pull {model}")
        raise


def chat(
    message: str,
    system_prompt: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    model: str = OLLAMA_LLM_MODEL,
) -> str:
    """
    LLM ile sohbet et (basit kullanım).

    Args:
        message: Kullanıcı mesajı
        system_prompt: Sistem talimatı
        chat_history: Önceki mesajlar [{"role": "user|assistant", "content": "..."}]
        model: Kullanılacak model

    Returns:
        str: LLM yanıtı
    """
    llm = get_llm(model=model)

    messages = []

    # Sistem mesajı
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    # Sohbet geçmişi
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # Kullanıcı mesajı
    messages.append(HumanMessage(content=message))

    # Yanıt al
    response = llm.invoke(messages)
    return response.content


def check_ollama_status() -> Dict:
    """
    Ollama bağlantı durumunu kontrol et.

    Returns:
        Dict: Durum bilgisi
    """
    import urllib.request
    import json

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())

        models = [m["name"] for m in data.get("models", [])]

        return {
            "status": "running",
            "url": OLLAMA_BASE_URL,
            "available_models": models,
            "model_count": len(models),
        }

    except Exception as e:
        return {
            "status": "error",
            "url": OLLAMA_BASE_URL,
            "error": str(e),
            "available_models": [],
            "model_count": 0,
        }


def pull_model(model_name: str) -> bool:
    """
    Ollama modelini indir.

    Args:
        model_name: İndirilecek model adı

    Returns:
        bool: Başarılı mı
    """
    import urllib.request
    import json

    try:
        logger.info(f"Model indiriliyor: {model_name}")

        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/pull",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=300) as response:
            # Stream yanıtı oku
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    status = json.loads(line.decode())
                    if "completed" in status and "total" in status:
                        progress = status["completed"] / status["total"] * 100
                        logger.info(f"İndirme: %{progress:.1f}")
                except:
                    pass

        logger.info(f"Model indirildi: {model_name}")
        return True

    except Exception as e:
        logger.error(f"Model indirme hatası: {e}")
        return False


def get_recommended_model(vram_gb: float = 8.0) -> str:
    """
    Mevcut VRAM'e göre önerilen modeli döndür.

    Args:
        vram_gb: Mevcut VRAM (GB)

    Returns:
        str: Önerilen model adı
    """
    suitable_models = [
        name for name, info in SUPPORTED_MODELS.items()
        if info["vram_gb"] <= vram_gb
    ]

    if not suitable_models:
        # En hafif modeli döndür
        return "llama3.2:3b"

    # İlk uygun modeli döndür (varsayılan olanı tercih et)
    for model in suitable_models:
        if SUPPORTED_MODELS[model].get("default", False):
            return model

    return suitable_models[0]


def get_model_info() -> List[Dict]:
    """
    Desteklenen modellerin bilgilerini döndür.

    Returns:
        List[Dict]: Model bilgileri
    """
    return [
        {
            "name": name,
            "display_name": info["name"],
            "vram_required": info["vram_gb"],
            "default": info.get("default", False),
        }
        for name, info in SUPPORTED_MODELS.items()
    ]


# === AKADEMİK SORU-CEVAP İÇİN SİSTEM PROMPT'LARI ===

ACADEMIC_SYSTEM_PROMPT = """Sen akademik bir araştırma asistanısın. 
Görevin, sana sağlanan akademik dokümanların içeriğine dayanarak 
kullanıcının sorularını doğru ve kaynak göstererek yanıtlamaktır.

Kurallar:
1. Yanıtlarını SADECE sağlanan dokümanlardaki bilgilere dayanarak ver.
2. Her iddia için kaynak belirt: hangi makale/dokümandan geldiğini belirt.
3. Emin olmadığın konularda "Sağlanan dokümanlarda bu konuda yeterli bilgi bulunmuyor" de.
4. Teknik terimleri doğru kullan, gerektiğinde açıkla.
5. Yanıtların bilimsel, objektif ve öz olsun.
6. Gerekirse maddeler halinde sırala.
7. Türkçe yanıt ver. Türkçe karakterleri doğru kullan (ç, ğ, ı, ö, ş, ü).
8. Qwen modeli olarak en güncel ve doğru bilgiyi sağlamaya çalış."""


# === DOĞRUDAN ÇALIŞTIRMA ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== LLM Modülü Test ===\n")

    # Ollama durumunu kontrol et
    print("1. Ollama durumu kontrol ediliyor...")
    status = check_ollama_status()
    print(f"   Durum: {status['status']}")
    print(f"   URL: {status['url']}")

    if status['status'] == 'running':
        print(f"   Mevcut modeller: {status['available_models']}")

        if status['available_models']:
            # Test sorgusu
            print("\n2. Test sorgusu gönderiliyor...")
            response = chat(
                message="Makine öğrenmesi nedir? Kısaca açıkla.",
                system_prompt=ACADEMIC_SYSTEM_PROMPT,
            )
            print(f"   Yanıt: {response[:200]}...")
        else:
            print("\n   UYARI: Hiç model yok! Model indirmek için:")
            print(f"   ollama pull {OLLAMA_LLM_MODEL}")
    else:
        print(f"\n   HATA: Ollama çalışmıyor!")
        print("   Başlatmak için: ollama serve")

    # Önerilen model
    print("\n3. 8GB VRAM için önerilen model:")
    recommended = get_recommended_model(vram_gb=8.0)
    print(f"   {recommended}")

    print("\n✅ LLM modülü testi tamamlandı!")
