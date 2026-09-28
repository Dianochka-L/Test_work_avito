import re
_WS = re.compile(r"\s+")
# адрес из параметров для текста бесполезен, локацию считаем отдельно
_ADDR = re.compile(r"Место оказания услуг.*?(?=Тип |Вид |$)")


def query_text(q, params):
    q = _WS.sub(" ", q or "").strip()
    params = _WS.sub(" ", params or "").strip()
    return f"query: {q}" + (f" | {params}" if params and params != "Вид услуги" else "")


def item_text(title, params, desc, desc_chars=300):
    params = _ADDR.sub("", params or "")
    s = f"passage: {title or ''} | {params} | {(desc or '')[:desc_chars]}"
    return _WS.sub(" ", s).strip()
