from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import importlib.util
import json
import unittest
import zipfile
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd

from survey.core import (Question, SurveyError, read_sheets, definitions, map_questions,
                         build_dataset, aggregate, filter_index, membership)
from survey.charts import (FigureSpec, render, export_figure, batch_export,
                           multipage_pdf, save_settings, load_settings, validate_spec,
                           default_font)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    qs = [Question("1", "単一", "single", ["A", "B"]),
          Question("2", "複数", "multi", ["X", "Y"]),
          Question("3", "属性", "single", ["G1", "G2"]),
          Question("4", "時間", "numeric", []),
          Question("5", "重複属性", "multi", ["M", "N"])]
    frame = pd.DataFrame([
        ["No.1：単一", "No.2：複数", "No.2：複数", "No.3：属性", "No.4：時間", "No.5：重複属性", "No.5：重複属性"],
        ["", "2-X", "2-Y", "", "", "5-M", "5-N"],
        ["A", "X", "Y", "G1", 0, "M", "N"],
        ["B", "", "Y", "G1", 10, "M", ""],
        ["", "", "", "G2", "", "", "N"],
        ["A", "X", "", "G2", 20, "", "N"],
        ["", "", "", "", "", "", ""],
    ])
    map_questions(qs, frame)
    return build_dataset(qs, frame), frame


def marked_fixture():
    ds, frame = fixture()
    for q in ds.questions:
        if q.kind == "multi":
            for col in q.columns:
                for row in range(2, len(frame)):
                    if frame.iloc[row, col]:
                        frame.iloc[row, col] = ["〇", "○", " 〇 "][(row + col) % 3]
    return build_dataset(ds.questions, frame), frame


class CoreTests(unittest.TestCase):
    def test_supplied_structure(self):
        defs = next(iter(read_sheets(str(ROOT / "question_explanation.xlsx")).values()))
        frame = next(iter(read_sheets(str(ROOT / "sample.xlsx")).values()))
        qs = definitions(defs)
        groups = map_questions(qs, frame)
        ds = build_dataset(qs, frame)
        self.assertEqual((len(qs), len(groups), frame.shape[1], len(ds.index)), (128, 128, 348, 0))
        self.assertFalse([q.issue for q in qs if q.issue])
        self.assertEqual(qs[14].number, "15")
        self.assertNotEqual(qs[14].parent, qs[23].parent)
        self.assertEqual(qs[14].title, qs[23].title)

    def test_single_valid_all_and_missing(self):
        ds, _ = fixture()
        result = aggregate(ds, "1")
        self.assertEqual(result.counts.loc["全体"].tolist(), [2, 1])
        self.assertAlmostEqual(result.percents.loc["全体", "A"], 200 / 3)
        self.assertEqual(aggregate(ds, "1", denominator="all").percents.loc["全体", "A"], 50)
        missing = aggregate(ds, "1", include_missing=True)
        self.assertEqual(missing.counts.iloc[0].sum(), 4)
        self.assertEqual(missing.denominators.iloc[0, 0], 4)

    def test_multi_respondent_denominator(self):
        ds, _ = fixture()
        result = aggregate(ds, "2")
        self.assertEqual(result.counts.iloc[0].tolist(), [2, 2])
        self.assertEqual(result.denominators.iloc[0].tolist(), [3, 3])
        self.assertAlmostEqual(result.percents.iloc[0].sum(), 400 / 3)
        self.assertEqual(aggregate(ds, "2", denominator="all").percents.iloc[0].tolist(), [50, 50])

    def test_circle_answers_use_header_choices_and_valid_denominator(self):
        legacy, _ = fixture()
        ds, _ = marked_fixture()
        self.assertFalse(ds.problems)
        self.assertEqual(ds.values["2"].columns.tolist(), ["X", "Y"])
        for qid in ["2", "5"]:
            pd.testing.assert_frame_equal(ds.values[qid], legacy.values[qid])
            result = aggregate(ds, qid)
            pd.testing.assert_frame_equal(result.counts, aggregate(legacy, qid).counts)
            pd.testing.assert_frame_equal(result.percents, aggregate(legacy, qid).percents)
        self.assertEqual(aggregate(ds, "2").denominators.iloc[0].tolist(), [3, 3])
        self.assertEqual(aggregate(ds, "2", denominator="all").percents.iloc[0].tolist(), [50, 50])
        missing = aggregate(ds, "2", include_missing=True)
        self.assertEqual(missing.counts.iloc[0].tolist(), [2, 2, 1])
        self.assertEqual(missing.denominators.iloc[0, 0], 4)

    def test_circle_answers_attribute_cross_and_filters(self):
        ds, _ = marked_fixture()
        self.assertEqual(aggregate(ds, "2", "3").counts.values.tolist(), [[1, 2], [1, 0]])
        result = aggregate(ds, "2", "5")
        self.assertEqual(result.counts.values.tolist(), [[1, 2], [2, 1]])
        self.assertEqual(result.denominators.values.tolist(), [[2, 2], [2, 2]])
        self.assertEqual(filter_index(ds, [{"question": "2", "choices": ["X"]}]).tolist(), [0, 3])
        self.assertEqual(filter_index(ds, [{"question": "2", "choices": ["X", "Y"]}, {"question": "5", "choices": ["M"]}]).tolist(), [0, 1])

    def test_circle_unknown_values_and_legacy_other_text(self):
        ds, frame = marked_fixture()
        for value in ["×", "0", "1", "unexpected"]:
            invalid = frame.copy()
            invalid.iloc[2, 1] = value
            bad = build_dataset(ds.questions, invalid)
            self.assertIn(value, bad.problems["2"])
            self.assertFalse(bad.values["2"].iloc[0, 0])
            with self.assertRaises(SurveyError):
                aggregate(bad, "2")
        q = Question("other", "その他の例", "multi", ["その他"], columns=[0], column_choices={0: "その他"})
        other = build_dataset([q], pd.DataFrame([["No.1：その他の例"], ["1-その他"], ["自由記述の回答"], ["〇"]]))
        self.assertFalse(other.problems)
        self.assertEqual(aggregate(other, "other").counts.iloc[0, 0], 2)

    def test_cross_single_row_column_overall(self):
        ds, _ = fixture()
        row = aggregate(ds, "1", "3")
        self.assertEqual(row.counts.values.tolist(), [[1, 1], [1, 0]])
        self.assertEqual(row.percents.values.tolist(), [[50, 50], [100, 0]])
        col = aggregate(ds, "1", "3", percent_mode="column")
        self.assertEqual(col.percents.values.tolist(), [[50, 100], [50, 0]])
        total = aggregate(ds, "1", "3", percent_mode="overall")
        self.assertEqual(total.denominators.iloc[0, 0], 3)

    def test_multi_cross_unique_pairs_and_overlapping_groups(self):
        ds, _ = fixture()
        result = aggregate(ds, "2", "5")
        self.assertEqual(result.counts.values.tolist(), [[1, 2], [2, 1]])
        self.assertEqual(result.denominators.values.tolist(), [[2, 2], [2, 2]])
        self.assertIn("重複", result.note)
        grouped = aggregate(ds, "1", "5")
        self.assertEqual(grouped.counts.values.tolist(), [[1, 1], [2, 0]])

    def test_filters_and_empty_denominator(self):
        ds, _ = fixture()
        filters = [{"question": "1", "choices": ["A", "B"]}, {"question": "3", "choices": ["G2"]}]
        self.assertEqual(filter_index(ds, filters).tolist(), [3])
        self.assertEqual(filter_index(ds, [{"question": "4", "min": 0, "max": 10}]).tolist(), [0, 1])
        zero = aggregate(ds, "1", filters=[{"question": "1", "choices": []}])
        self.assertTrue(zero.percents.isna().all().all())
        self.assertEqual(zero.filtered_n, 0)

    def test_duplicate_mapping_and_unknown_values_block(self):
        ds, frame = fixture()
        map_questions(ds.questions, frame, {"1": "1", "2": "1"})
        self.assertTrue(ds.questions[1].issue)
        ds, frame = fixture()
        frame.iloc[2, 0] = "unexpected"
        bad = build_dataset(ds.questions, frame)
        self.assertIn("1", bad.problems)
        with self.assertRaises(SurveyError):
            membership(bad, "1")
        fixed = build_dataset(ds.questions, frame, {"1": {"unexpected": "A"}})
        self.assertNotIn("1", fixed.problems)
        self.assertEqual(aggregate(fixed, "1").counts.iloc[0, 0], 2)

    def test_numeric_invalid_and_valid_zero(self):
        ds, frame = fixture()
        self.assertEqual(ds.values["4"].dropna().tolist(), [0, 10, 20])
        for value in ["bad", -1, 1.5, "inf"]:
            frame.iloc[2, 4] = value
            self.assertIn("4", build_dataset(ds.questions, frame).problems)

    def test_fingerprint_does_not_depend_on_answers(self):
        ds, frame = fixture()
        frame.iloc[2, 0] = "B"
        self.assertEqual(ds.fingerprint, build_dataset(ds.questions, frame).fingerprint)

    def test_bad_path_and_wrong_format_safe_messages(self):
        with self.assertRaises(SurveyError):
            read_sheets(str(ROOT / "nonexistent.xlsx"))
        with self.assertRaises(SurveyError):
            definitions(pd.DataFrame([["wrong"]]))

    def test_xlsm_and_unicode_path(self):
        # OOXML sheets have the same representation in xlsx and xlsm;
        # a genuine macro-enabled fixture checks the openpyxl reader as well.
        directory = ROOT / "tests" / "artifacts" / "日本語 パス確認"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "★【関係者限り】回答 データ.xlsm"
        if importlib.util.find_spec("openpyxl"):
            from openpyxl import Workbook
            workbook = Workbook()
            workbook.active.append(["No.1：設問"])
            workbook.active.append([""])
            workbook.active.append(["回答"])
            buffer = BytesIO()
            workbook.save(buffer)
            output = BytesIO()
            with zipfile.ZipFile(buffer) as source, zipfile.ZipFile(output, "w") as dest:
                for name in source.namelist():
                    content = source.read(name)
                    if name == "[Content_Types].xml":
                        content = content.replace(b"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml", b"application/vnd.ms-excel.sheet.macroEnabled.main+xml")
                    dest.writestr(name, content)
            target.write_bytes(output.getvalue())
            self.assertEqual(next(iter(read_sheets(str(target)).values())).iloc[2, 0], "回答")
        else:
            target.write_bytes((ROOT / "sample.xlsx").read_bytes())
            self.assertEqual(next(iter(read_sheets(str(target)).values())).shape, (2, 348))


class ExportTests(unittest.TestCase):
    def test_all_charts_export_and_close(self):
        ds, _ = fixture()
        for spec in [FigureSpec("1", font=default_font()),
                     FigureSpec("1", group="3", mode="attribute", chart="stacked", font=default_font()),
                     FigureSpec("2", group="5", mode="cross", chart="heatmap", font=default_font()),
                     FigureSpec("4", chart="histogram", font=default_font())]:
            fig = render(ds, spec)
            try:
                self.assertTrue(export_figure(fig, spec, "png").startswith(b"\x89PNG"))
                svg = export_figure(fig, spec, "svg")
                self.assertIn(b"<svg", svg)
                self.assertIn(b"<path", svg)
                self.assertTrue(export_figure(fig, spec, "pdf").startswith(b"%PDF"))
            finally:
                plt.close(fig)
        self.assertFalse(plt.get_fignums())

    def test_svg_text_mode(self):
        ds, _ = fixture()
        spec = FigureSpec("1", svg_text=True, font=default_font())
        fig = render(ds, spec)
        try:
            self.assertIn(b"<text", export_figure(fig, spec, "svg"))
        finally:
            plt.close(fig)

    def test_batch_zip_and_pdf_page_count(self):
        ds, _ = fixture()
        specs = [FigureSpec("1", font=default_font()), FigureSpec("2", font=default_font())]
        with zipfile.ZipFile(BytesIO(batch_export(ds, specs))) as archive:
            self.assertEqual(len(archive.namelist()), 6)
        pdf = multipage_pdf(ds, specs)
        if importlib.util.find_spec("pypdf"):
            from pypdf import PdfReader
            self.assertEqual(len(PdfReader(BytesIO(pdf)).pages), 2)

    def test_settings_roundtrip_excludes_responses(self):
        ds, _ = fixture()
        spec = FigureSpec("1", title="保存テスト", filters=[{"question": "3", "choices": ["G1"]}], font=default_font())
        raw = save_settings(ds, [spec], {"1": "1"}, {})
        payload = json.loads(raw)
        self.assertEqual(set(payload), {"version", "fingerprint", "figures", "mappings", "choice_mappings"})
        got, mappings, _ = load_settings(raw, ds)
        self.assertEqual(asdict(got[0]), asdict(spec))
        self.assertEqual(mappings, {"1": "1"})
        validate_spec(got[0], ds)
        payload["fingerprint"] = "incompatible"
        with self.assertRaises(SurveyError):
            load_settings(json.dumps(payload).encode(), ds)
        payload["version"] = 100
        with self.assertRaises(SurveyError):
            load_settings(json.dumps(payload).encode(), ds)
        with self.assertRaises(SurveyError):
            load_settings(b"not json", ds)

    def test_disallow_misleading_stacks_and_bad_colors(self):
        ds, _ = fixture()
        for spec in [FigureSpec("2", chart="stacked"), FigureSpec("1", chart="stacked", denominator="all"), FigureSpec("1", colors={"A": "bad color"}), FigureSpec("1", width=0)]:
            with self.assertRaises(SurveyError):
                validate_spec(spec, ds)

    def test_malformed_settings_values_and_filters(self):
        ds, _ = fixture()
        for spec in [FigureSpec("1", colors=None), FigureSpec("1", labels=[]),
                     FigureSpec("1", title=["bad"]), FigureSpec("1", include_missing="yes"),
                     FigureSpec("1", filters=[{"question": "missing", "choices": ["A"]}]),
                     FigureSpec("1", filters=[{"question": "4", "min": 20, "max": 0}])]:
            with self.assertRaises(SurveyError):
                validate_spec(spec, ds)

    def test_zero_respondents_all_plot_types(self):
        ds, _ = fixture()
        for chart in ["bar", "stacked", "heatmap"]:
            spec = FigureSpec("1", group="3", mode="cross", chart=chart,
                              filters=[{"question": "1", "choices": []}], font=default_font())
            fig = render(ds, spec)
            try:
                self.assertTrue(export_figure(fig, spec, "png"))
            finally:
                plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("msoffcrypto") and importlib.util.find_spec("openpyxl"), "暗号化Excel用ライブラリが未インストール")
    def test_encrypted_excel_in_memory(self):
        import msoffcrypto
        from openpyxl import Workbook
        plain, encrypted = BytesIO(), BytesIO()
        workbook = Workbook()
        ds, frame = marked_fixture()
        for row in frame.itertuples(index=False, name=None):
            workbook.active.append(row)
        workbook.save(plain)
        plain.seek(0)
        msoffcrypto.OfficeFile(plain).encrypt("test-password", encrypted)
        # Python 3.13 creates TemporaryDirectory with restrictive Windows ACLs
        # that this managed execution profile cannot reopen. Use a normal
        # workspace directory and remove only this test's known file.
        directory = ROOT / "tests" / "artifacts" / f"encrypted_{uuid4().hex}"
        directory.mkdir(parents=True)
        path = directory / "encrypted.xlsx"
        try:
            path.write_bytes(encrypted.getvalue())
            before = path.read_bytes()
            loaded = next(iter(read_sheets(str(path), "test-password").values()))
            decrypted = build_dataset(ds.questions, loaded)
            self.assertFalse(decrypted.problems)
            self.assertEqual(aggregate(decrypted, "2").counts.iloc[0].tolist(), [2, 2])
            with self.assertRaises(SurveyError):
                read_sheets(str(path), "wrong")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in directory.iterdir()], ["encrypted.xlsx"])
        finally:
            path.unlink(missing_ok=True)
            directory.rmdir()


class AppTests(unittest.TestCase):
    def test_start_and_sample_load(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "読み込む / 再読込").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.dataset.questions), 128)
        self.assertEqual(len(app.session_state.dataset.index), 0)
        self.assertEqual(app.session_state.answer_password, "")
        next(b for b in app.button if b.label == "データと設定をクリア").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("sheets", app.session_state)

    def test_synthetic_data_editor_and_export_list(self):
        from streamlit.testing.v1 import AppTest
        defs = next(iter(read_sheets(str(ROOT / "question_explanation.xlsx")).values()))
        frame = next(iter(read_sheets(str(ROOT / "sample.xlsx")).values()))
        qs = definitions(defs)
        map_questions(qs, frame)
        row = [""] * frame.shape[1]
        for q in qs:
            if q.kind == "single":
                row[q.columns[0]] = q.choices[0]
            elif q.kind == "multi":
                row[q.columns[0]] = "〇"
            elif q.kind == "numeric":
                row[q.columns[0]] = 0
        frame.loc[2] = row
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
        app.session_state.sheets = ({"定義": defs}, {"回答": frame})
        app.run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "この図を一覧に追加").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.figures), 1)
        next(b for b in app.button if b.label == "一括出力を生成").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state.batch_zip.startswith(b"PK"))
        mode = next(s for s in app.selectbox if s.label == "集計方法")
        mode.select("cross").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.mode, "cross")
        group = next(s for s in app.selectbox if s.label == "属性 / クロス集計の行設問")
        group.select(qs[3].id).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.group, qs[3].id)
        next(b for b in app.button if b.label == "編集").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.mode, "single")
        palette = next(s for s in app.selectbox if s.label == "配色")
        palette.select("モノクロ").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.palette, "モノクロ")
        self.assertEqual(app.session_state.active_spec.colors, {})
        next(b for b in app.button if b.label == "一覧のこの図を更新").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.figures), 1)
        self.assertEqual(app.session_state.figures[0].palette, "モノクロ")


if __name__ == "__main__":
    unittest.main()
