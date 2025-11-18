import os, re, json, time
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

import psycopg2

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", ""))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")     



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectIn(BaseModel):
    manufacturer: str
    product: str
    extra: Optional[str] = None

class AltItem(BaseModel):
    code: Optional[str] = None
    reason: Optional[str] = None

class Payments(BaseModel):
    duty: Optional[str] = None
    vat: Optional[str] = None
    excise: Optional[str] = None
    fees: Optional[str] = None

class DetectOut(BaseModel):
    code: str
    duty: str
    vat: str
    raw: Optional[str] = None
    description: Optional[str] = None
    tech31: Optional[str] = None
    decl31: Optional[str] = None
    classification_reason: Optional[str] = None
    alternatives: Optional[List[AltItem]] = None
    payments: Optional[Payments] = None
    requirements: Optional[List[str]] = None

class FeedbackIn(BaseModel):
    acc_code: int
    desc_31: int
    reason_clarity: int
    ui: int
    comment: Optional[str] = None

    manufacturer: Optional[str] = None
    product: Optional[str] = None
    extra: Optional[str] = None

    code: Optional[str] = None
    tech31: Optional[str] = None
    decl31: Optional[str] = None


TEN_DIGITS = re.compile(r"\b\d{10}\b")

def _take_10digits(s: str) -> str:
    if not s:
        return ""
    m = TEN_DIGITS.search(s.replace(" ", ""))
    return m.group(0) if m else ""

def _norm_percent(s: str) -> str:
    if not s:
        return ""
    s = s.strip().replace(",", ".")
    m = re.search(r"(\d+(\.\d+)?)\s*%?", s)
    return (m.group(1) + "%") if m else ""

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rfind("}")
        blob = text[start:end+1]
        return json.loads(blob)
    except Exception:
        return None

def _stringify_tech31(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            k_s = str(k).strip().capitalize()
            if isinstance(v, (list, tuple)):
                v_s = "; ".join(str(x).strip() for x in v if str(x).strip())
            elif isinstance(v, dict):
                v_s = "; ".join(f"{kk}: {vv}" for kk, vv in v.items())
            else:
                v_s = str(v).strip()
            if v_s:
                parts.append(f"- {k_s}: {v_s}")
        return "\n".join(parts)
    if isinstance(val, (list, tuple)):
        return "\n".join(f"- {str(x).strip()}" for x in val if str(x).strip())
    return str(val).strip()

def _normalize_alternatives(val):
    out = []
    if isinstance(val, dict):
        for k, v in val.items():
            out.append({"code": str(k), "reason": str(v)})
    elif isinstance(val, (list, tuple)):
        for it in val:
            if isinstance(it, dict):
                out.append({
                    "code": str(it.get("code", "") or it.get("код", "") or ""),
                    "reason": str(it.get("reason", "") or it.get("обоснование", "") or "")
                })
            else:
                out.append({"code": str(it), "reason": ""})
    elif val:
        out.append({"code": str(val), "reason": ""})
    return out

def _normalize_payments(val, fallback_duty: str, fallback_vat: str):
    d = {"duty": fallback_duty, "vat": fallback_vat, "excise": "—", "fees": "—"}
    if isinstance(val, dict):
        for k in ("duty", "vat", "excise", "fees"):
            if k in val and val[k] is not None:
                d[k] = str(val[k]).strip()
    return d

def get_db_connection():
    if not (DB_HOST and DB_NAME and DB_USER):
        print("[feedback] DB settings not configured, skip save")
        return None
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        dbname=DB_NAME,
    )

def save_feedback_to_db(fb: FeedbackIn, request: Request) -> None:
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    acc_code_rating,
                    tech31_rating,
                    reason_rating,
                    ui_rating,
                    req_manufacturer,
                    req_product,
                    req_extra,
                    res_code,
                    res_tech31,
                    res_decl31,
                    comment,
                    user_agent,
                    client_ip
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    fb.acc_code,
                    fb.desc_31,
                    fb.reason_clarity,
                    fb.ui,
                    fb.manufacturer or "",
                    fb.product or "",
                    fb.extra or "",
                    fb.code or "",
                    fb.tech31 or "",
                    fb.decl31 or "",
                    fb.comment or "",
                    request.headers.get("user-agent", ""),
                    request.client.host if request.client else None,
                ),
            )
    finally:
        conn.close()
# ===============================

def _normalize_requirements(val):
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        import re as _re
        items = [s.strip(" -•\t") for s in _re.split(r"[\n;]+", val) if s.strip()]
        return items or [val.strip()]
    if val:
        return [str(val)]
    return []

@app.post("/tnved/detect", response_model=DetectOut)
def detect(inp: DetectIn, request: Request):
    full = (inp.product or "").strip()
    if inp.extra and inp.extra.strip().lower() != "null":
        full += f" ({inp.extra.strip()})"
    if inp.manufacturer and inp.manufacturer.strip().lower() != "null":
        full += f" — Производитель: {inp.manufacturer.strip()}"

    if not full:
        raise HTTPException(status_code=400, detail="Поля пустые")

    system_msg = """Ты — эксперт по классификации товаров по ТН ВЭД ЕАЭС и по подготовке текстов для графы 31 декларации на товары.
    Ты — эксперт по классификации товаров по ТН ВЭД ЕАЭС и по подготовке текстов для графы 31 декларации на товары.
    Твоя задача: по краткому описанию товара определить наиболее вероятный 10-значный код ТН ВЭД ЕАЭС, указать ставки платежей и сформировать подробное техническое описание товара.
    Если предоставленной информации недостаточно для уверенной классификации (нет назначения, материалов, электрических параметров, области применения и т.п.), ты должен сначала получить недостающие сведения через web-поиск по типовым описаниям схожих товаров и уже на основе найденного сформировать итоговое описание. Используй только общедоступные и типовые характеристики, не выдумывай конкретные модели и бренды, если их нет во входных данных. Делай оговорки: «по типовым техническим характеристикам для такого вида товара».
    Результат верни строго в виде одного json-объекта
    Структура JSON (поля на русском):
    
    {
      "code": "10-значный код или \"UNKNOWN\"",
      "duty": "проценты или \"UNKNOWN\"",
      "vat": "проценты или \"UNKNOWN\"",
      "tech31": "подробное структурированное техописание: 1) назначение; 2) конструкция и материалы; 3) основные технические/электрические параметры (если применимо); 4) условия эксплуатации; 5) комплектация. Объем не меньше 100 слов. Если часть данных взята из типовых открытых источников — так и укажи.",
      "decl31": "готовая формулировка для графы 31 декларации на товары, краткая, без лишних пояснений, в одном абзаце, с указанием основных отличительных признаков и назначения. Без слов «примерно», «возможно», «как правило».",
      "classification_reason": "обоснование выбора позиции ТН ВЭД (ОПИ, примечания к группе/товарной позиции, признаки товара). Если есть неопределенность — укажи диапазон возможных кодов и чего не хватает.",
      "alternatives": [
        {"code": "возможный_код", "reason": "в каких случаях применим"}
      ],
      "payments": {
        "duty": "% или \"UNKNOWN\"",
        "vat": "% или \"UNKNOWN\"",
        "excise": "— или значение",
        "fees": "— или значение"
      },
      "requirements": [
        "ТР ЕАЭС, безопасность, лицензирование, сертификация — если применимо"
      ],
      "sources": [
        "краткие ссылки/названия найденных источников, если делался веб-поиск"
      ]
    }
    
    Требования:
    - не добавляй никаких комментариев вне JSON;
    - не меняй имена полей;
    - если веб-поиск не дал точных параметров — пиши «по типовым характеристикам для данного вида товара».
    
    """
    user_msg = (
        "Определи 10-значный код ТН ВЭД для товара и верни результат СТРОГО в виде JSON.\n"
        "Вход:\n"
        f"{json.dumps({'Наименование': full}, ensure_ascii=False)}"
    )

    try:
        resp = client.responses.create(
            model="gpt-5",
            tools=[{"type": "web_search"}],
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка GPT API: {e}")

    text = (resp.output_text or "").strip()
    data = _extract_json_block(text) or {}

    code = (data.get("code") or "").strip()
    if not _take_10digits(code):
        guessed = _take_10digits(text)
        code = guessed or (code if code.upper().startswith("UNKNOWN") else "")

    duty = _norm_percent(data.get("duty") or "")
    vat  = _norm_percent(data.get("vat") or "")

    code = code or "UNKNOWN"
    duty = duty or "UNKNOWN"
    vat  = vat or "UNKNOWN"

    tech31 = _stringify_tech31(data.get("tech31"))
    alternatives = _normalize_alternatives(data.get("alternatives"))
    payments = _normalize_payments(data.get("payments"), fallback_duty=duty, fallback_vat=vat)
    requirements = _normalize_requirements(data.get("requirements"))
    decl31 = (data.get("decl31") or "").strip()

    out = DetectOut(
        code=code,
        duty=duty,
        vat=vat,
        raw=text,
        description=(data.get("description") or ""),
        tech31=tech31,
        decl31=decl31,
        classification_reason=(data.get("classification_reason") or ""),
        alternatives=alternatives,
        payments=payments,
        requirements=requirements,
    )
    return out


@app.post("/feedback")
def feedback(fb: FeedbackIn, request: Request):
    try:
        save_feedback_to_db(fb, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения отзыва: {e}")
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "Проверка работоспособности"}
