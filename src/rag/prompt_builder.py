from typing import Any


DEFAULT_SYSTEM_PROMPT = """
Sen kurum içi dokümanlara göre cevap veren bir yapay zeka asistanısın.
Cevapların kaynaklı, denetlenebilir ve bağlama sadık olmalıdır.
Kamu kurumu senaryosunda nihai karar veriyormuş gibi konuşmamalısın.
""".strip()


def build_grounded_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\\n\\n".join(
        [
            f"[Kaynak: {item['source']} | Skor: {item['score']:.3f}]\\n{item['text']}"
            for item in contexts
        ]
    )

    return f"""
Kurallar:
- Sadece aşağıdaki bağlam bilgisini kullan.
- Bağlamda cevap yoksa "Dokümanda bu bilgi bulunamadı." de.
- Cevabın sonunda kullandığın kaynakları belirt.
- Tahmin yapma.
- Uydurma madde, tarih, kurum adı veya sayı üretme.
- Karar destek sistemleri için nihai karar veriyormuş gibi konuşma.

Bağlam:
{context_text}

Soru:
{question}

Cevap:
""".strip()
