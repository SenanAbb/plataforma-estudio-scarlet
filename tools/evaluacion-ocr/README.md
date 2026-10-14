# Métricas de OCR (CER / WER)

Herramienta de evaluación (no forma parte de la aplicación). Vive en `src/tools/evaluacion-ocr/`; `src/` es la raíz de todo el código del proyecto (front, back, base de datos, Docker…).

Compara el texto de un motor OCR con una referencia humana. Solo biblioteca estándar (Python 3). Las rutas entran siempre por argumentos; el programa nunca escribe en ellas.

## Uso

```bash
# Desde src/tools/evaluacion-ocr/
python3 -B -m unittest test_metrics -v

python3 -B metrics.py --ref REF.txt --hyp SALIDA.txt [--json]
python3 -B metrics.py --ref-dir DIR_REFS --hyp-dir DIR_SALIDAS [--json]
```
(`-B` evita crear `__pycache__`.) Los archivos del lote se emparejan por nombre base (`a.txt` con `a.txt`).

**Códigos de salida:** `0` correcto · `1` error de lectura · `2` uso incorrecto o lote con problemas (sin pares o archivos sin pareja; se informa de cuáles).

## Qué calcula
- **CER** (caracteres) y **WER** (palabras separadas por espacio), **normalizados** y **estrictos** (solo NFC).
- **% de texto evaluable** (descontando `[ilegible]`).
- **Errores críticos**: cifras, importes, porcentajes, unidades, negaciones e Incoterms. Son candidatos a revisión humana, no un veredicto.
- **Extras**: elementos críticos que la salida tiene y la referencia no (posibles invenciones).
- Lote: media simple **y ponderada** (suma de distancias / suma de longitudes), peor archivo por CER y por WER, recuentos de críticos y de casos vacíos.

## Estados y vacíos
`status`: `ok`, `empty_output` (salida vacía) o `empty_reference` (referencia vacía o solo `[ilegible]`). En los casos no `ok`, CER/WER son `null` (nunca 0 % ni 100 % silenciosos) y no entran en los promedios; el lote los cuenta aparte.

## Normalización (`normalize`)
Siempre NFC. En modo normal además: comillas tipográficas a ASCII; se retiran encabezados `#`, énfasis `*x*`/`**x**` y viñetas `* `, comillas invertidas, filas separadoras de tabla (`|---|---|`) y los `|`; se colapsan espacios y saltos de línea.
**No se toca**: mayúsculas, tildes, cifras, separadores, signos (`3*4` se conserva), unidades, negaciones, guiones ni rayas.

## `[ilegible]`
La referencia puede contener `[ilegible]`. Se alinea por programación dinámica con traza hacia atrás (carácter para CER, palabra para WER): lo que cubre el marcador cuesta 0, no entra en el denominador, y lo que la salida escribe ahí se devuelve aparte (`illegible_content`). Se aplica igual en estricto.

## Errores críticos
Se comparan por entidad completa y como multiconjunto, sobre texto normalizado: cada aparición en la referencia consume una en la salida. `1.250,50` no está en `11.250,50`; `1.250,50` y `1,250.50` son distintos. Importes, porcentajes y unidades ignoran espacios (`100 €` = `100€`). Unidades: kg, g, t, l, ml, m, cm, mm, km, m2, m3, TEU. Monedas: €, $, £, EUR, USD, GBP, CHF, JPY.

## Limitaciones conocidas
- Con `[ilegible]` se usa una matriz completa O(n·m): unas 3 000 letras tardan ~3 s y varias decenas de MB. Sin marcador es más rápido (~0,2 s en la prueba).
- Las tablas se comparan como texto corrido en orden de lectura; su estructura hay que valorarla aparte.
- Las listas de unidades, monedas y negaciones son cerradas: algo fuera de ellas no se audita (p. ej. «euros» escrito en letras, «pallets»). «sin», «ni» y «no» también aparecen en usos que no son negaciones críticas.
- No comprende el significado: dos cifras distintas pero equivalentes (`1.000` y `1000`) se señalan como diferentes a propósito.
- Los criterios de aceptación siguen en `docs/pruebas-ocr.md`; este módulo solo mide.
