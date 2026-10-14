"""ocr_run.py
Lee fotos con motores de OCR LOCALES (Tesseract, GLM-OCR y Qwen3-VL por Ollama),
uno tras otro, y genera una página HTML para revisarlas a ojo: imagen a la
izquierda, texto de cada motor a la derecha.

- Las imágenes nunca salen del equipo: Ollama solo se admite en localhost.
- Sin fallback remoto. Un motor que falla queda registrado como fallo.
- Una salida vacía NO se da por buena (estado "vacia").
- No modifica las imágenes. Solo escribe dentro de --out.

Uso (desde src/tools/evaluacion-ocr/):
  python3 -B ocr_run.py --images ../../../fotos --out ../../../salidas/revision
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

PROMPT = (
    "Transcribe literalmente todo el texto visible en la imagen, en español. "
    "Respeta el orden de lectura, las cifras, los signos y los términos técnicos. "
    "No resumas, no corrijas ni completes nada. Si un fragmento no se puede leer, "
    "escribe [ilegible]. Devuelve solo la transcripción."
)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
OLLAMA_DEFAULT = "http://127.0.0.1:11434"
ENGINES = ("tesseract", "glm-ocr", "qwen3-vl:4b-instruct")
TIMEOUT_S = 600

# Configuración por modelo, fijada tras diagnosticar los fallos (2026-10-14):
# - qwen3-vl:4b es la variante "thinking" (no se puede apagar): se usa 4b-instruct.
#   Prompt corto: el largo con "[ilegible]" provocaba bucles. Imagen a tamaño completo.
# - glm-ocr: el modelo importado tiene la plantilla "{{ .Prompt }}" sin formato de
#   conversación y no emite fin de texto: se le da el formato GLM a mano, se reduce
#   la imagen (a tamaño completo falla) y se cortan los bucles de "```".
CONFIG: Dict[str, dict] = {
    "glm-ocr": {
        "prompt": "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>"
                  "Text Recognition:<|assistant|>\n",
        "options": {"temperature": 0, "num_predict": 3000,
                    "stop": ["\n```\n```", "<|endoftext|>", "<|user|>"]},
        "max_width": 1024,
    },
    "qwen3-vl:4b-instruct": {
        "prompt": "Transcribe literalmente todo el texto visible en la imagen, en español. "
                  "No resumas ni añadas nada.",
        "options": {"temperature": 0, "num_predict": 3000},
        "max_width": None,
    },
}


def safe(engine: str) -> str:
    return engine.replace(":", "-").replace("/", "-")


def check_local(host: str) -> str:
    """Solo se acepta Ollama en esta máquina (las imágenes no salen del equipo)."""
    from urllib.parse import urlparse
    u = urlparse(host)
    if u.scheme not in ("http",) or u.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"Host no local rechazado: {host}")
    return host.rstrip("/")


def list_images(folder: str) -> List[str]:
    def natural(name: str):  # 1, 2, ..., 10 en vez de 1, 10, 2
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]
    return sorted((f for f in os.listdir(folder)
                   if f.lower().endswith(EXTENSIONS) and os.path.isfile(os.path.join(folder, f))),
                  key=natural)


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.load(r)


def ollama_ps(host: str) -> list:
    try:
        with urllib.request.urlopen(host + "/api/ps", timeout=10) as r:
            return [m.get("name") for m in json.load(r).get("models", [])]
    except Exception:
        return ["(no se pudo consultar)"]


def run_tesseract(path: str) -> dict:
    t = time.time()
    p = subprocess.run(["tesseract", path, "stdout", "-l", "spa"], capture_output=True,
                       text=True, timeout=TIMEOUT_S)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:300] or f"tesseract salió con {p.returncode}")
    return {"text": p.stdout, "seconds": time.time() - t, "load_s": None}


def shrink(path: str, width: int) -> str:
    """Copia temporal con anchura máxima *width* (ImageMagick). Devuelve su ruta."""
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    p = subprocess.run(["magick", path, "-resize", f"{width}x>", tmp], capture_output=True, text=True)
    if p.returncode != 0:
        os.unlink(tmp)
        raise RuntimeError("no se pudo reducir la imagen: " + p.stderr.strip()[:200])
    return tmp


def trim_repetition(text: str, window: int = 10) -> tuple:
    """Corta el texto donde empieza un bucle (una ventana de *window* palabras que
    ya había aparecido) y quita las vallas ``` sobrantes. Devuelve (texto, recortado)."""
    trimmed = False
    words = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    seen: Dict[tuple, int] = {}
    for i in range(len(words) - window + 1):
        key = tuple(w for _, w in words[i:i + window])
        if key in seen:
            text, trimmed = text[:words[i][0]], True
            break
        seen[key] = i
    cleaned = re.sub(r"(\s*```\s*)+$", "", text.rstrip())
    cleaned = re.sub(r"^\s*```[a-z]*\n", "", cleaned)
    return cleaned, trimmed or cleaned != text.rstrip()


def run_ollama(host: str, model: str, path: str) -> dict:
    cfg = CONFIG.get(model, {"prompt": PROMPT, "options": {"temperature": 0}, "max_width": None})
    tmp = shrink(path, cfg["max_width"]) if cfg.get("max_width") else None
    try:
        with open(tmp or path, "rb") as f:
            img = base64.b64encode(f.read()).decode()
    finally:
        if tmp:
            os.unlink(tmp)
    payload = {"model": model, "prompt": cfg["prompt"], "images": [img], "stream": False,
               "options": dict(cfg["options"])}
    if "thinking" in model or model == "qwen3-vl:4b":
        payload["think"] = False
    t = time.time()
    r = post_json(host + "/api/generate", payload)
    text, trimmed = trim_repetition(r.get("response", "")) if model == "glm-ocr" else (r.get("response", ""), False)
    return {
        "text": text,
        "seconds": time.time() - t,
        "load_s": (r.get("load_duration") or 0) / 1e9,
        "done_reason": r.get("done_reason"),
        "eval_count": r.get("eval_count"),
        "thinking_chars": len(r.get("thinking") or ""),
        "recortada": trimmed,
    }


def unload(host: str, model: str) -> None:
    try:
        post_json(host + "/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def process(images_dir: str, out_dir: str, engines: List[str], host: str) -> None:
    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No hay imágenes en {images_dir}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "registro.jsonl")
    for engine in engines:
        edir = os.path.join(out_dir, safe(engine))
        os.makedirs(edir, exist_ok=True)
        print(f"== {engine} (Ollama cargado antes: {ollama_ps(host)}) ==", flush=True)
        for name in images:
            stem = os.path.splitext(name)[0]
            path = os.path.join(images_dir, name)
            entry: Dict[str, object] = {"engine": engine, "image": name}
            try:
                res = run_tesseract(path) if engine == "tesseract" else run_ollama(host, engine, path)
                text = res.pop("text")
                entry.update(res)
                if not text.strip():
                    entry["status"] = "vacia"
                else:
                    entry["status"] = "ok"
                    if res.get("recortada"):
                        entry["status"] = "ok_recortada"  # se cortó un bucle: revisar el final
                    if res.get("done_reason") == "length":
                        entry["status"] = "truncada"
                    with open(os.path.join(edir, stem + ".txt"), "w", encoding="utf-8") as f:
                        f.write(text)
                entry["chars"] = len(text.strip())
            except Exception as e:  # el fallo se registra, nunca se esconde
                entry.update(status="fallo", error=f"{type(e).__name__}: {e}"[:300])
            print(f"  {name}: {entry['status']} ({entry.get('seconds', 0):.1f}s, {entry.get('chars', 0)} car.)",
                  flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if engine != "tesseract":
            unload(host, engine)  # descarga solo el modelo propio de esta prueba


def make_report(images_dir: str, out_dir: str, engines: List[str]) -> str:
    log: Dict[tuple, dict] = {}
    log_path = os.path.join(out_dir, "registro.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                log[(e["engine"], e["image"])] = e  # la última ejecución manda
    rows = []
    for name in list_images(images_dir):
        stem = os.path.splitext(name)[0]
        img_rel = os.path.relpath(os.path.join(images_dir, name), out_dir)
        cols = []
        for eng in engines:
            e = log.get((eng, name), {})
            status = e.get("status", "sin ejecutar")
            txt_path = os.path.join(out_dir, safe(eng), stem + ".txt")
            text = ""
            if status in ("ok", "ok_recortada", "truncada") and os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    text = f.read()
            warn = "" if status == "ok" else f" <b class='bad'>{html.escape(status.upper())}</b>"
            if status == "fallo":
                warn += f" {html.escape(str(e.get('error', '')))}"
            secs = f" · {e['seconds']:.1f}s" if "seconds" in e else ""
            cols.append(f"<div class='col'><h3>{html.escape(eng)}{secs}{warn}</h3>"
                        f"<pre>{html.escape(text) if text else '(sin texto)'}</pre></div>")
        rows.append(f"<section><h2>{html.escape(name)}</h2><div class='row'>"
                    f"<a href='{html.escape(img_rel)}' target='_blank'><img src='{html.escape(img_rel)}'></a>"
                    f"<div class='cols'>{''.join(cols)}</div></div></section>")
    page = ("<!doctype html><meta charset='utf-8'><title>Revisión OCR</title><style>"
            "body{font:14px system-ui;margin:16px;background:#fafafa;color:#111}"
            ".row{display:flex;gap:16px;align-items:flex-start}"
            "img{width:340px;position:sticky;top:8px;border:1px solid #ccc}"
            ".cols{display:flex;gap:12px;flex:1;overflow-x:auto}.col{flex:1;min-width:300px}"
            "pre{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:8px;max-height:700px;overflow:auto}"
            ".bad{color:#b00}section{margin-bottom:40px}</style>"
            "<h1>Revisión visual de OCR (local, sin validar)</h1>"
            "<p>Compara cada texto con la imagen. Esto no es una medición.</p>" + "".join(rows))
    out = os.path.join(out_dir, "revision.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="OCR local de una carpeta de fotos + página de revisión.")
    ap.add_argument("--images", required=True, help="Carpeta con las fotos")
    ap.add_argument("--out", required=True, help="Carpeta de salida (se crea)")
    ap.add_argument("--engines", default=",".join(ENGINES), help="Lista separada por comas")
    ap.add_argument("--ollama", default=OLLAMA_DEFAULT, help="Solo localhost")
    ap.add_argument("--report-only", action="store_true", help="Regenera la página sin ejecutar OCR")
    args = ap.parse_args(argv)
    host = check_local(args.ollama)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if not os.path.isdir(args.images):
        print(f"No existe la carpeta de imágenes: {args.images}", file=sys.stderr)
        return 1
    if not args.report_only:
        process(args.images, args.out, engines, host)
    os.makedirs(args.out, exist_ok=True)
    print("Página de revisión:", os.path.abspath(make_report(args.images, args.out, engines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
