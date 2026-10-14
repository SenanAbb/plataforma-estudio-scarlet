"""metrics.py
Métricas de OCR: CER y WER, normalización conservadora, marcador [ilegible],
candidatos a errores críticos y CLI (par único o lote).

Solo biblioteca estándar. Todas las rutas entran por argumentos de línea de
comandos; el módulo nunca escribe en ellas.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

ILLEGIBLE = "[ilegible]"
_WILD = None  # elemento comodín en la referencia (zona [ilegible])

# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")
_EMPHASIS_RE = re.compile(r"(?<![\w*])(\*{1,3})(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_BULLET_RE = re.compile(r"(?m)^[ \t]*\*[ \t]+")
_HEADING_RE = re.compile(r"(?m)^[ \t]*#+[ \t]*")


def normalize(text: str, strict: bool = False) -> str:
    """Normaliza *text*.

    - Siempre: Unicode NFC.
    - ``strict=False``: comillas tipográficas -> ASCII; retira marcado Markdown
      (encabezados, énfasis y viñetas con ``*``, comillas invertidas, filas
      separadoras de tabla y delimitadores ``|``) sin eliminar contenido; colapsa
      espacios y saltos de línea.
    - ``strict=True``: solo NFC.

    No toca mayúsculas, tildes, cifras, separadores, signos (``*`` como
    multiplicación incluido), unidades, negaciones ni guiones.
    """
    text = unicodedata.normalize("NFC", text)
    if strict:
        return text

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("`", "")

    # Filas separadoras de tabla (|---|---|) y líneas de guiones: fuera enteras
    text = "\n".join(ln for ln in text.splitlines() if not _TABLE_SEP_RE.match(ln))

    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Distancia de edición
# ---------------------------------------------------------------------------

def levenshtein(a: Any, b: Any) -> int:
    """Distancia de Levenshtein entre dos secuencias (str o listas), dos filas."""
    s1, s2 = list(a), list(b)
    # Prefijo y sufijo comunes no cambian la distancia
    start = 0
    while start < len(s1) and start < len(s2) and s1[start] == s2[start]:
        start += 1
    end1, end2 = len(s1), len(s2)
    while end1 > start and end2 > start and s1[end1 - 1] == s2[end2 - 1]:
        end1 -= 1
        end2 -= 1
    s1, s2 = s1[start:end1], s2[start:end2]

    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (c1 != c2))
        previous = current
    return previous[-1]


def _align(ref: List[Optional[str]], hyp: List[str]) -> Tuple[int, int, List[List[str]]]:
    """Distancia de edición con comodines en la referencia.

    Un elemento ``_WILD`` de *ref* (zona [ilegible]) absorbe cero o más elementos
    de *hyp* a coste 0 y no cuenta en el denominador.

    Devuelve ``(distancia, n_evaluables, absorbido)`` donde ``absorbido[k]`` son
    los elementos de *hyp* alineados con el k-ésimo comodín.
    """
    wild_positions = [i for i, r in enumerate(ref) if r is _WILD]
    if not wild_positions:
        return levenshtein(ref, hyp), len(ref), []

    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        wild = ref[i - 1] is _WILD
        D[i][0] = D[i - 1][0] if wild else D[i - 1][0] + 1
        for j in range(1, m + 1):
            if wild:
                D[i][j] = min(D[i - 1][j], D[i][j - 1])
            else:
                D[i][j] = min(
                    D[i - 1][j] + 1,
                    D[i][j - 1] + 1,
                    D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                )

    wild_index = {pos: k for k, pos in enumerate(wild_positions)}
    absorbed: List[List[str]] = [[] for _ in wild_positions]
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and ref[i - 1] is _WILD:
            if j > 0 and D[i][j] == D[i][j - 1]:
                absorbed[wild_index[i - 1]].append(hyp[j - 1])
                j -= 1
            else:
                i -= 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    for bucket in absorbed:
        bucket.reverse()
    return D[n][m], n - len(wild_positions), absorbed


def _chars(text: str) -> List[Optional[str]]:
    seq: List[Optional[str]] = []
    parts = text.split(ILLEGIBLE)
    for k, part in enumerate(parts):
        seq.extend(part)
        if k < len(parts) - 1:
            seq.append(_WILD)
    return seq


def _words(text: str) -> List[Optional[str]]:
    return [_WILD if w == ILLEGIBLE else w for w in text.replace(ILLEGIBLE, f" {ILLEGIBLE} ").split()]


def _rate(dist: int, n: int) -> Optional[float]:
    return dist / n if n > 0 else None


def cer(reference: str, hypothesis: str) -> Optional[float]:
    """CER sobre los textos tal cual (sin normalizar). ``None`` si la referencia
    no tiene texto evaluable. Los tramos [ilegible] se excluyen."""
    d, n, _ = _align(_chars(reference), list(hypothesis))
    return _rate(d, n)


def wer(reference: str, hypothesis: str) -> Optional[float]:
    """WER sobre los textos tal cual. ``None`` si no hay palabras evaluables."""
    d, n, _ = _align(_words(reference), hypothesis.split())
    return _rate(d, n)


# ---------------------------------------------------------------------------
# Errores críticos (candidatos para revisión humana, no un veredicto)
# ---------------------------------------------------------------------------

_NUM = r"\d+(?:[.,]\d+)*"
_CUR = r"(?:€|\$|£|EUR|USD|GBP|CHF|JPY)"
_UNITS = r"(?:kg|g|t|l|ml|m2|m3|m²|m³|cm|mm|km|m|TEU)"
_INCOTERMS = ("EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP")

_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "cifras": re.compile(rf"(?<![\w.,]){_NUM}(?![\w])"),
    "importes": re.compile(
        rf"(?<![\w.,])(?:{_NUM}\s*{_CUR}(?![\w])|{_CUR}\s*{_NUM}(?![\w]))", re.IGNORECASE
    ),
    "porcentajes": re.compile(rf"(?<![\w.,]){_NUM}\s*%"),
    "unidades": re.compile(rf"(?<![\w.,]){_NUM}\s*{_UNITS}(?![\w])", re.IGNORECASE),
    "negaciones": re.compile(r"\b(?:no|ni|nunca|jamás|tampoco|sin)\b", re.IGNORECASE),
    "incoterms": re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_INCOTERMS) + r")(?![A-Za-z0-9])"),
}
CATEGORIES = tuple(_PATTERNS)


def _canon(cat: str, value: str) -> str:
    if cat == "negaciones":
        return value.lower()
    if cat in ("importes", "porcentajes", "unidades"):
        return re.sub(r"\s+", "", value).upper()
    return value


def _extract(text: str) -> Dict[str, List[str]]:
    return {cat: [m.group() for m in pat.finditer(text)] for cat, pat in _PATTERNS.items()}


def extract_critical_errors(reference: str, hypothesis: str) -> Dict[str, List[Dict[str, str]]]:
    """Elementos críticos de la referencia y si aparecen en la hipótesis.

    Se comparan por entidad completa (con límites de palabra) y como multiconjunto:
    cada aparición en la referencia consume una aparición en la hipótesis. Se
    trabaja sobre el texto normalizado. Resultado: ``{categoría: [{expected,
    found}]}`` con ``found == "ausente"`` cuando falta.
    """
    ref_items = _extract(normalize(reference))
    pool = {cat: Counter(_canon(cat, v) for v in vals)
            for cat, vals in _extract(normalize(hypothesis)).items()}
    errors: Dict[str, List[Dict[str, str]]] = {}
    for cat, vals in ref_items.items():
        out = []
        for v in vals:
            key = _canon(cat, v)
            if pool[cat][key] > 0:
                pool[cat][key] -= 1
                out.append({"expected": v, "found": v})
            else:
                out.append({"expected": v, "found": "ausente"})
        errors[cat] = out
    return errors


def extract_critical_extras(reference: str, hypothesis: str) -> Dict[str, List[str]]:
    """Elementos críticos que aparecen en la hipótesis y no en la referencia
    (posibles invenciones o sustituciones)."""
    pool = {cat: Counter(_canon(cat, v) for v in vals)
            for cat, vals in _extract(normalize(reference)).items()}
    extras: Dict[str, List[str]] = {}
    for cat, vals in _extract(normalize(hypothesis)).items():
        out = []
        for v in vals:
            key = _canon(cat, v)
            if pool[cat][key] > 0:
                pool[cat][key] -= 1
            else:
                out.append(v)
        extras[cat] = out
    return extras


# ---------------------------------------------------------------------------
# Métricas de un par de textos
# ---------------------------------------------------------------------------

def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Todas las métricas de un par referencia/hipótesis.

    ``status``: ``ok`` | ``empty_reference`` | ``empty_output``. En los casos
    vacíos cer/wer son ``None`` (nunca 0,0 ni 1,0 silenciosos).
    """
    ref_n, hyp_n = normalize(reference), normalize(hypothesis)
    ref_s, hyp_s = normalize(reference, strict=True), normalize(hypothesis, strict=True)

    d_cn, n_cn, absorbed = _align(_chars(ref_n), list(hyp_n))
    d_wn, n_wn, _ = _align(_words(ref_n), hyp_n.split())
    d_cs, n_cs, _ = _align(_chars(ref_s), list(hyp_s))
    d_ws, n_ws, _ = _align(_words(ref_s), hyp_s.split())

    n_markers = ref_n.count(ILLEGIBLE)
    if n_cn == 0:
        status, evaluable = "empty_reference", None
    else:
        status = "empty_output" if not hypothesis.strip() else "ok"
        evaluable = 100.0 * n_cn / (n_cn + n_markers * len(ILLEGIBLE))

    measured = status == "ok"
    result: Dict[str, Any] = {
        "status": status,
        "empty_output": status == "empty_output",
        "cer_normalized": _rate(d_cn, n_cn) if measured else None,
        "wer_normalized": _rate(d_wn, n_wn) if measured else None,
        "cer_strict": _rate(d_cs, n_cs) if measured else None,
        "wer_strict": _rate(d_ws, n_ws) if measured else None,
        "evaluable_percentage": evaluable,
        "illegible_content": ["".join(b) for b in absorbed],
        # Conteos brutos para agregar de forma ponderada en el lote
        "counts": {
            "cer_normalized": [d_cn, n_cn], "wer_normalized": [d_wn, n_wn],
            "cer_strict": [d_cs, n_cs], "wer_strict": [d_ws, n_ws],
        },
    }
    if status == "empty_reference":
        result["critical_errors"] = {c: [] for c in CATEGORIES}
        result["critical_extras"] = {c: [] for c in CATEGORIES}
    else:
        result["critical_errors"] = extract_critical_errors(reference, hypothesis)
        result["critical_extras"] = extract_critical_extras(reference, hypothesis)
    return result


# ---------------------------------------------------------------------------
# Agregación en lote
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("cer_normalized", "wer_normalized", "cer_strict", "wer_strict")


def _aggregate_batch(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ok = {k: v for k, v in results.items() if v["status"] == "ok"}
    overall: Dict[str, Any] = {
        "n_files": len(results),
        "n_ok": len(ok),
        "n_empty_output": sum(v["status"] == "empty_output" for v in results.values()),
        "n_empty_reference": sum(v["status"] == "empty_reference" for v in results.values()),
    }
    for key in _METRIC_KEYS:
        vals = [v[key] for v in ok.values()]
        overall[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        dist = sum(v["counts"][key][0] for v in ok.values())
        n = sum(v["counts"][key][1] for v in ok.values())
        overall[f"{key}_weighted"] = _rate(dist, n)
    overall["worst_file_cer"] = max(ok, key=lambda k: ok[k]["cer_normalized"]) if ok else None
    overall["worst_file_wer"] = max(ok, key=lambda k: ok[k]["wer_normalized"]) if ok else None
    evals = [v["evaluable_percentage"] for v in ok.values() if v["evaluable_percentage"] is not None]
    overall["evaluable_percentage_mean"] = sum(evals) / len(evals) if evals else None
    overall["critical"] = {
        cat: {
            "total": sum(len(v["critical_errors"][cat]) for v in results.values()),
            "ausentes": sum(e["found"] == "ausente" for v in results.values()
                            for e in v["critical_errors"][cat]),
            "extras": sum(len(v["critical_extras"][cat]) for v in results.values()),
        }
        for cat in CATEGORIES
    }
    return overall


# ---------------------------------------------------------------------------
# Salida legible
# ---------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "n/d" if x is None else f"{x:.2%}"


def _print_metrics(name: str, m: Dict[str, Any]) -> None:
    print(f"=== {name} ===")
    if m["status"] != "ok":
        print({"empty_output": "SALIDA VACÍA: no es un resultado válido",
               "empty_reference": "REFERENCIA VACÍA o sin texto evaluable: no se calcula"}[m["status"]])
    else:
        print(f"CER normalizado: {_pct(m['cer_normalized'])}   WER normalizado: {_pct(m['wer_normalized'])}")
        print(f"CER estricto:    {_pct(m['cer_strict'])}   WER estricto:    {_pct(m['wer_strict'])}")
    if m["evaluable_percentage"] is not None:
        print(f"% texto evaluable: {m['evaluable_percentage']:.2f}%")
    for i, content in enumerate(m["illegible_content"], 1):
        if content:
            print(f"  zona [ilegible] {i}: la salida escribió «{content[:40]}»")
    for cat in CATEGORIES:
        items = m["critical_errors"][cat]
        missing = [e["expected"] for e in items if e["found"] == "ausente"]
        extras = m["critical_extras"][cat]
        if items or extras:
            line = f"  {cat}: {len(items) - len(missing)}/{len(items)} presentes"
            if missing:
                line += f"; AUSENTES: {missing}"
            if extras:
                line += f"; EXTRA en salida: {extras}"
            print(line)
    print()


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def _stems(directory: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in sorted(os.listdir(directory)):
        full = os.path.join(directory, f)
        if os.path.isfile(full):
            stem = os.path.splitext(f)[0]
            if stem in out:
                raise ValueError(f"Nombre base duplicado en {directory}: {stem}")
            out[stem] = full
    return out


# ---------------------------------------------------------------------------
# CLI. Códigos de salida: 0 correcto; 1 error de lectura/E-S; 2 uso incorrecto
# o lote con problemas de emparejado (sin pares o archivos sin pareja).
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CER/WER y errores críticos para OCR.")
    parser.add_argument("--ref", help="Archivo de referencia")
    parser.add_argument("--hyp", help="Archivo de salida del OCR")
    parser.add_argument("--ref-dir", help="Directorio de referencias")
    parser.add_argument("--hyp-dir", help="Directorio de salidas")
    parser.add_argument("--json", action="store_true", help="Salida en JSON")
    args = parser.parse_args(argv)

    single = args.ref is not None or args.hyp is not None
    batch = args.ref_dir is not None or args.hyp_dir is not None
    if single == batch:
        parser.error("usa o bien --ref y --hyp, o bien --ref-dir y --hyp-dir")
    if single and not (args.ref and args.hyp):
        parser.error("--ref y --hyp van juntos")
    if batch and not (args.ref_dir and args.hyp_dir):
        parser.error("--ref-dir y --hyp-dir van juntos")

    if single:
        try:
            ref_text, hyp_text = _read(args.ref), _read(args.hyp)
        except (OSError, UnicodeDecodeError) as e:
            print(f"Error de lectura: {e}", file=sys.stderr)
            return 1
        result = compute_metrics(ref_text, hyp_text)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_metrics(os.path.basename(args.ref), result)
        return 0

    try:
        refs, hyps = _stems(args.ref_dir), _stems(args.hyp_dir)
        paired = sorted(set(refs) & set(hyps))
        results = {n: compute_metrics(_read(refs[n]), _read(hyps[n])) for n in paired}
    except (OSError, UnicodeDecodeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    missing_hyp = sorted(set(refs) - set(hyps))   # referencia sin salida
    missing_ref = sorted(set(hyps) - set(refs))   # salida sin referencia
    overall = _aggregate_batch(results)

    if args.json:
        print(json.dumps({"files": results, "overall": overall,
                          "missing_hypothesis": missing_hyp,
                          "missing_reference": missing_ref}, ensure_ascii=False, indent=2))
    else:
        for name, m in results.items():
            _print_metrics(name, m)
        if missing_hyp:
            print("Referencias SIN salida (no evaluadas):", ", ".join(missing_hyp))
        if missing_ref:
            print("Salidas SIN referencia (no evaluadas):", ", ".join(missing_ref))
        print("=== Resumen del lote ===")
        print(f"Archivos: {overall['n_files']} (ok {overall['n_ok']}, salida vacía "
              f"{overall['n_empty_output']}, referencia vacía {overall['n_empty_reference']})")
        for key in _METRIC_KEYS:
            print(f"{key}: media {_pct(overall[key + '_mean'])} | ponderado {_pct(overall[key + '_weighted'])}")
        print(f"Peor archivo por CER: {overall['worst_file_cer']} | por WER: {overall['worst_file_wer']}")
        for cat, c in overall["critical"].items():
            if c["total"] or c["extras"]:
                print(f"  {cat}: {c['total'] - c['ausentes']}/{c['total']} presentes, {c['extras']} extra")

    if not paired or missing_hyp or missing_ref:
        print("AVISO: lote sin pares o con archivos sin pareja (código de salida 2).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
