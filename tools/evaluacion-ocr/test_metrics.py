"""Tests de metrics.py. Todos los valores esperados están calculados a mano y
escritos como literales; los textos son sintéticos."""

import contextlib
import io
import json
import os
import tempfile
import unittest

import metrics
from metrics import cer, compute_metrics, extract_critical_errors, extract_critical_extras, levenshtein, normalize, wer


def founds(result, cat):
    return [e["found"] for e in result[cat]]


class TestDistancias(unittest.TestCase):
    def test_levenshtein_clasico(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_cer_sustitucion_insercion_omision(self):
        self.assertAlmostEqual(cer("abcdef", "abXdef"), 1 / 6)   # sustitución
        self.assertAlmostEqual(cer("abcdef", "abcdefg"), 1 / 6)  # inserción
        self.assertAlmostEqual(cer("abcdef", "abdef"), 1 / 6)    # omisión

    def test_wer_sustitucion_insercion_omision(self):
        self.assertAlmostEqual(wer("uno dos tres", "uno cuatro tres"), 1 / 3)
        self.assertAlmostEqual(wer("uno dos tres", "uno dos tres cuatro"), 1 / 3)
        self.assertAlmostEqual(wer("uno dos tres", "uno tres"), 1 / 3)
        self.assertAlmostEqual(wer("a b c", "a x c"), 1 / 3)

    def test_identidad(self):
        m = compute_metrics("Este es un texto de prueba.", "Este es un texto de prueba.")
        self.assertEqual(m["status"], "ok")
        for k in ("cer_normalized", "wer_normalized", "cer_strict", "wer_strict"):
            self.assertEqual(m[k], 0.0)

    def test_tilde_perdida_es_error(self):
        self.assertAlmostEqual(cer("canción", "cancion"), 1 / 7)
        self.assertAlmostEqual(wer("canción", "cancion"), 1.0)

    def test_precio_sin_punto_de_millares(self):
        # "El precio es 1.250,50 euros." mide 28 caracteres y 5 palabras
        m = compute_metrics("El precio es 1.250,50 euros.", "El precio es 1250,50 euros.")
        self.assertAlmostEqual(m["cer_normalized"], 1 / 28)
        self.assertAlmostEqual(m["wer_normalized"], 1 / 5)


class TestNormalizacion(unittest.TestCase):
    def test_markdown_sin_perder_contenido(self):
        self.assertEqual(normalize("# Título\n**Texto** con *énfasis* y `código`."),
                         "Título Texto con énfasis y código.")
        m = compute_metrics("# Título\n**Texto** con *énfasis*.", "Título Texto con énfasis.")
        self.assertEqual(m["cer_normalized"], 0.0)

    def test_asterisco_de_multiplicacion_se_conserva(self):
        self.assertEqual(normalize("Calcula 3*4 = 12"), "Calcula 3*4 = 12")
        self.assertEqual(normalize("3*4 y 5*6"), "3*4 y 5*6")

    def test_vinetas_con_asterisco(self):
        self.assertEqual(normalize("* uno\n* dos"), "uno dos")

    def test_fila_separadora_de_tabla(self):
        self.assertEqual(normalize("|a|b|\n|---|---|\n|1|2|"), "a b 1 2")
        self.assertEqual(normalize("| a | b |\n| :--- | ---: |\n| 1 | 2 |"), "a b 1 2")

    def test_no_normaliza_signos_ni_mayusculas(self):
        self.assertEqual(normalize("FOB  – 5 % −3"), "FOB – 5 % −3")
        self.assertNotEqual(normalize("Mar"), normalize("mar"))

    def test_estricto_distinto_del_normalizado(self):
        ref = "“cita” y ‘comilla’"          # 18 caracteres
        hyp = "\"cita\" y 'comilla'"
        m = compute_metrics(ref, hyp)
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertAlmostEqual(m["cer_strict"], 4 / 18)


class TestIlegible(unittest.TestCase):
    REF = "El precio es [ilegible] euros y la entrega FOB."
    HYP = "El presio es 12 euros y la entrega CIF."

    def test_valores_calculados_a_mano(self):
        # Evaluable: 13 + 24 = 37 caracteres; errores: c->s (1) y FOB->CIF (3) = 4
        # Palabras evaluables: 8; errores: precio->presio y FOB.->CIF. = 2
        # Total con marcador: 37 + 10 = 47
        m = compute_metrics(self.REF, self.HYP)
        self.assertAlmostEqual(m["cer_normalized"], 4 / 37)
        self.assertAlmostEqual(m["wer_normalized"], 2 / 8)
        self.assertAlmostEqual(m["evaluable_percentage"], 100 * 37 / 47)
        self.assertEqual(m["illegible_content"], ["12"])

    def test_mismas_cifras_en_estricto(self):
        m = compute_metrics(self.REF, self.HYP)
        self.assertAlmostEqual(m["cer_strict"], 4 / 37)
        self.assertAlmostEqual(m["wer_strict"], 2 / 8)

    def test_lo_escrito_en_la_zona_no_cuenta(self):
        m = compute_metrics("Hola [ilegible] fin", "Hola XYZ fin")
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertEqual(m["wer_normalized"], 0.0)
        self.assertEqual(m["illegible_content"], ["XYZ"])

    def test_marcador_al_inicio_y_al_final(self):
        self.assertEqual(compute_metrics("[ilegible] fin", "abc fin")["cer_normalized"], 0.0)
        self.assertEqual(compute_metrics("Hola [ilegible]", "Hola mundo")["cer_normalized"], 0.0)

    def test_error_fuera_del_marcador_se_cuenta(self):
        m = compute_metrics("Hola [ilegible] fin", "Hola XYZ fim")
        self.assertAlmostEqual(m["cer_normalized"], 1 / 9)  # "Hola " + " fin" = 9 evaluables

    def test_dos_marcadores(self):
        m = compute_metrics("a [ilegible] b [ilegible] c", "a 1 b 22 c")
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertEqual(m["illegible_content"], ["1", "22"])

    def test_solo_marcador_no_es_evaluable(self):
        self.assertEqual(compute_metrics("[ilegible]", "algo")["status"], "empty_reference")


class TestErroresCriticos(unittest.TestCase):
    def test_negacion_presente_y_ausente(self):
        ok = extract_critical_errors("No se permite la entrada.", "No se permite la entrada.")
        self.assertEqual(founds(ok, "negaciones"), ["No"])
        mal = extract_critical_errors("No se permite la entrada.", "Se permite la entrada.")
        self.assertEqual(founds(mal, "negaciones"), ["ausente"])

    def test_negaciones_repetidas(self):
        r = extract_critical_errors("No no sí", "No sí")
        self.assertEqual(founds(r, "negaciones"), ["No", "ausente"])

    def test_cifra_con_otro_separador(self):
        r = extract_critical_errors("Total 1.250,50 EUR", "Total 1,250.50 EUR")
        self.assertEqual(founds(r, "cifras"), ["ausente"])

    def test_cifra_no_se_encuentra_dentro_de_otra(self):
        r = extract_critical_errors("Total 1.250,50 EUR", "Total 11.250,50 EUR")
        self.assertEqual(founds(r, "cifras"), ["ausente"])
        self.assertEqual(founds(r, "importes"), ["ausente"])

    def test_cifra_de_un_digito(self):
        r = extract_critical_errors("Pagar 5 cajas", "Pagar 8 cajas")
        self.assertEqual([e["expected"] for e in r["cifras"]], ["5"])
        self.assertEqual(founds(r, "cifras"), ["ausente"])

    def test_importes_y_porcentajes(self):
        r = extract_critical_errors("Paga 100 € con el 21 % de IVA", "Paga 100 € con el 12 % de IVA")
        self.assertEqual(founds(r, "importes"), ["100 €"])
        self.assertEqual(founds(r, "porcentajes"), ["ausente"])

    def test_importe_con_formato_equivalente(self):
        r = extract_critical_errors("Paga 100 €", "Paga 100€")
        self.assertEqual(founds(r, "importes"), ["100 €"])

    def test_unidades(self):
        r = extract_critical_errors("Peso 20 kg", "Peso 20 g")
        self.assertEqual([e["expected"] for e in r["unidades"]], ["20 kg"])
        self.assertEqual(founds(r, "unidades"), ["ausente"])
        self.assertEqual(extract_critical_extras("Peso 20 kg", "Peso 20 g")["unidades"], ["20 g"])

    def test_unidad_no_es_prefijo_de_palabra(self):
        self.assertEqual(extract_critical_errors("Son 5 metros", "Son 5 metros")["unidades"], [])

    def test_incoterm_cambiado(self):
        r = extract_critical_errors("El término es FOB.", "El término es CIF.")
        self.assertEqual(founds(r, "incoterms"), ["ausente"])

    def test_incoterm_dentro_de_otra_palabra_no_cuenta(self):
        self.assertEqual(extract_critical_errors("CIFRA total", "CIFRA total")["incoterms"], [])


class TestVacios(unittest.TestCase):
    def test_salida_vacia(self):
        m = compute_metrics("Texto de referencia.", "   ")
        self.assertEqual(m["status"], "empty_output")
        self.assertTrue(m["empty_output"])
        self.assertIsNone(m["cer_normalized"])
        self.assertIsNone(m["wer_normalized"])

    def test_referencia_vacia(self):
        m = compute_metrics("", "texto inventado")
        self.assertEqual(m["status"], "empty_reference")
        self.assertIsNone(m["cer_normalized"])
        self.assertIsNone(cer("", "x"))
        self.assertIsNone(wer("", "x"))


def write(directory, name, text):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
        f.write(text)


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = metrics.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.refs = os.path.join(self.tmp.name, "refs")
        self.hyps = os.path.join(self.tmp.name, "hyps")
        os.mkdir(self.refs)
        os.mkdir(self.hyps)

    def test_par_con_archivo_inexistente(self):
        write(self.hyps, "h.txt", "algo")
        code, _, err = run_cli("--ref", os.path.join(self.refs, "no_existe.txt"),
                               "--hyp", os.path.join(self.hyps, "h.txt"))
        self.assertEqual(code, 1)
        self.assertIn("Error", err)

    def test_par_ok(self):
        write(self.refs, "a.txt", "abcd")
        write(self.hyps, "a.txt", "abxd")
        code, out, _ = run_cli("--ref", os.path.join(self.refs, "a.txt"),
                               "--hyp", os.path.join(self.hyps, "a.txt"), "--json")
        self.assertEqual(code, 0)
        self.assertAlmostEqual(json.loads(out)["cer_normalized"], 1 / 4)

    def test_lote_agregado_ponderado(self):
        write(self.refs, "a.txt", "abcd")
        write(self.hyps, "a.txt", "abxd")                       # 4 car., 1 error; 1 palabra, 1 error
        write(self.refs, "b.txt", "aaaaaaaaaa bbbbbbbbbb")       # 21 car., 2 palabras, sin errores
        write(self.hyps, "b.txt", "aaaaaaaaaa bbbbbbbbbb")
        write(self.refs, "c.txt", "x")
        write(self.hyps, "c.txt", "")                            # salida vacía: no entra en promedios
        code, out, _ = run_cli("--ref-dir", self.refs, "--hyp-dir", self.hyps, "--json")
        self.assertEqual(code, 0)
        o = json.loads(out)["overall"]
        self.assertEqual((o["n_files"], o["n_ok"], o["n_empty_output"]), (3, 2, 1))
        self.assertAlmostEqual(o["cer_normalized_mean"], 0.125)          # (1/4 + 0) / 2
        self.assertAlmostEqual(o["cer_normalized_weighted"], 1 / 25)     # 1 / (4 + 21)
        self.assertAlmostEqual(o["wer_normalized_mean"], 0.5)            # (1 + 0) / 2
        self.assertAlmostEqual(o["wer_normalized_weighted"], 1 / 3)      # 1 / (1 + 2)
        self.assertEqual(o["worst_file_cer"], "a")
        self.assertEqual(o["worst_file_wer"], "a")

    def test_lote_archivo_sin_pareja(self):
        write(self.refs, "solo_ref.txt", "Referencia")
        write(self.hyps, "solo_hyp.txt", "Hipótesis")
        code, out, _ = run_cli("--ref-dir", self.refs, "--hyp-dir", self.hyps, "--json")
        self.assertEqual(code, 2)
        o = json.loads(out)
        self.assertEqual(o["missing_hypothesis"], ["solo_ref"])
        self.assertEqual(o["missing_reference"], ["solo_hyp"])

    def test_lote_con_un_archivo_sin_pareja_y_pares_validos(self):
        write(self.refs, "a.txt", "abc")
        write(self.hyps, "a.txt", "abc")
        write(self.refs, "b.txt", "abc")
        code, _, _ = run_cli("--ref-dir", self.refs, "--hyp-dir", self.hyps, "--json")
        self.assertEqual(code, 2)

    def test_lote_sin_pares(self):
        code, _, _ = run_cli("--ref-dir", self.refs, "--hyp-dir", self.hyps)
        self.assertEqual(code, 2)

    def test_argumentos_incompatibles(self):
        self.assertEqual(run_cli("--hyp", "x.txt", "--ref-dir", self.refs, "--hyp-dir", self.hyps)[0], 2)
        self.assertEqual(run_cli("--ref", "x.txt", "--hyp-dir", self.hyps)[0], 2)
        self.assertEqual(run_cli("--ref", "x.txt")[0], 2)
        self.assertEqual(run_cli()[0], 2)

    def test_no_modifica_las_entradas(self):
        write(self.refs, "a.txt", "abcd")
        write(self.hyps, "a.txt", "abxd")
        run_cli("--ref-dir", self.refs, "--hyp-dir", self.hyps)
        with open(os.path.join(self.refs, "a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "abcd")
        self.assertEqual(sorted(os.listdir(self.refs)), ["a.txt"])


if __name__ == "__main__":
    unittest.main()
