from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import importlib.util
import json
import unittest
import zipfile
from uuid import uuid4
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

from survey.core import (Question, SurveyError, read_sheets, definitions, map_questions,
                         build_dataset, aggregate, filter_index, membership, _xml_sheets,
                         merged_choices, MISSING)
from survey.charts import (FigureSpec, render, export_figure, batch_export, data_for_spec,
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


def dense_fixture():
    choices = [f"回答{i + 1}" for i in range(6)]
    groups = [f"属性{i + 1}" for i in range(10)]
    questions = [Question("q", "比較する設問", "single", choices),
                 Question("g", "比較する属性", "single", groups)]
    frame = pd.DataFrame([["No.1：比較する設問", "No.2：比較する属性"], ["", ""]]
                         + [[choice, group] for group in groups for choice in choices])
    map_questions(questions, frame)
    return build_dataset(questions, frame)


class CoreTests(unittest.TestCase):
    def test_single_merges_into_existing_choice_and_keeps_missing(self):
        ds, _ = fixture()
        result = aggregate(ds, "1", question_merges={"A": "B"}, include_missing=True)
        self.assertEqual(result.counts.columns.tolist(), ["B", MISSING])
        self.assertEqual(result.counts.iloc[0].tolist(), [3, 2])
        self.assertEqual(result.percents.iloc[0].tolist(), [60, 40])
        self.assertEqual(result.filtered_n, 5)
        self.assertIn("B ← A / B", result.note)
        self.assertEqual(aggregate(ds, "1", question_merges={"A": "B"}).denominators.iloc[0, 0], 3)

    def test_multi_merges_deduplicate_respondents_on_both_axes(self):
        ds, _ = marked_fixture()
        before = ds.values["2"].copy(deep=True)
        question = {"X": "合計", "Y": "合計"}
        groups = {"M": "全属性", "N": "全属性"}
        result = aggregate(ds, "2", "5", question_merges=question, group_merges=groups)
        self.assertEqual(result.counts.values.tolist(), [[3]])
        self.assertEqual(result.denominators.values.tolist(), [[3]])
        self.assertEqual(result.percents.values.tolist(), [[100]])
        all_people = aggregate(ds, "2", "5", denominator="all", question_merges=question, group_merges=groups)
        self.assertEqual(all_people.denominators.values.tolist(), [[4]])
        self.assertEqual(all_people.percents.values.tolist(), [[75]])
        self.assertIn("回答側の集約", result.note)
        self.assertIn("属性側の集約", result.note)
        self.assertNotIn("グループ間に回答者の重複があります", result.note)
        pd.testing.assert_frame_equal(ds.values["2"], before)

    def test_merge_filters_use_original_choices_and_denominators(self):
        ds, _ = fixture()
        result = aggregate(ds, "2", filters=[{"question": "2", "choices": ["X"]}],
                           question_merges={"X": "まとめ", "Y": "まとめ"})
        self.assertEqual(result.filtered_n, 2)
        self.assertEqual(result.counts.iloc[0, 0], 2)
        self.assertEqual(result.denominators.iloc[0, 0], 2)
        for direction in ["row", "column", "overall"]:
            result = aggregate(ds, "1", "3", percent_mode=direction,
                               question_merges={"A": "全回答", "B": "全回答"},
                               group_merges={"G1": "全属性", "G2": "全属性"})
            self.assertEqual(result.counts.values.tolist(), [[3]])
            self.assertEqual(result.denominators.values.tolist(), [[3]])
            self.assertEqual(result.percents.values.tolist(), [[100]])

    def test_merge_order_and_invalid_mappings(self):
        ds, _ = fixture()
        self.assertEqual(merged_choices(ds.question("1"), {"B": "A"}), ["A"])
        self.assertEqual(merged_choices(ds.question("1"), {"A": "新しい名前"}), ["新しい名前", "B"])
        for mapping in [{"A": ""}, {"A": "   "}, {"A": MISSING},
                        {"missing": "その他"}, {"A": None}, []]:
            with self.assertRaises(SurveyError):
                aggregate(ds, "1", question_merges=mapping)
        with self.assertRaises(SurveyError):
            validate_spec(FigureSpec("4", chart="histogram", question_merges={"0": "まとめ"}), ds)
        with self.assertRaises(SurveyError):
            validate_spec(FigureSpec("1", group_merges={"G1": "まとめ"}), ds)

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
        self.assertEqual(len(ds.index), 5)
        self.assertEqual(aggregate(ds, "1", denominator="all").percents.loc["全体", "A"], 40)
        missing = aggregate(ds, "1", include_missing=True)
        self.assertEqual(missing.counts.iloc[0].sum(), 5)
        self.assertEqual(missing.denominators.iloc[0, 0], 5)

    def test_multi_respondent_denominator(self):
        ds, _ = fixture()
        result = aggregate(ds, "2")
        self.assertEqual(result.counts.iloc[0].tolist(), [2, 2])
        self.assertEqual(result.denominators.iloc[0].tolist(), [3, 3])
        self.assertAlmostEqual(result.percents.iloc[0].sum(), 400 / 3)
        self.assertEqual(aggregate(ds, "2", denominator="all").percents.iloc[0].tolist(), [40, 40])

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
        self.assertEqual(aggregate(ds, "2", denominator="all").percents.iloc[0].tolist(), [40, 40])
        missing = aggregate(ds, "2", include_missing=True)
        self.assertEqual(missing.counts.iloc[0].tolist(), [2, 2, 2])
        self.assertEqual(missing.denominators.iloc[0, 0], 5)

    def test_fully_blank_and_whitespace_responses_are_retained(self):
        ds, frame = fixture()
        frame.loc[3] = [None] * frame.shape[1]
        frame.loc[4] = [" \t　"] * frame.shape[1]
        rebuilt = build_dataset(ds.questions, frame)
        self.assertEqual(len(rebuilt.index), 5)
        self.assertEqual(rebuilt.values["1"].tolist(), ["A", "", "", "A", ""])
        self.assertEqual(aggregate(rebuilt, "1", denominator="all").denominators.iloc[0, 0], 5)
        self.assertEqual(aggregate(rebuilt, "1", include_missing=True).counts.iloc[0, -1], 3)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxlが未インストール")
    def test_1607_responses_including_middle_and_trailing_blank_rows(self):
        from openpyxl import Workbook
        workbook = Workbook()
        workbook.active.append(["No.1：単一"])
        workbook.active.append([None])
        for row in range(3, 1610):
            workbook.active.append([None if row in {5, 27, 1608, 1609} else "A"])
        buffer = BytesIO()
        workbook.save(buffer)
        raw = buffer.getvalue()
        for end_row in [None, 1609]:
            with patch("survey.core.Path.read_bytes", return_value=raw):
                frame = next(iter(read_sheets("synthetic.xlsx", preserve_blank_rows=True, end_row=end_row).values()))
            q = Question("q", "単一", "single", ["A"])
            map_questions([q], frame)
            ds = build_dataset([q], frame)
            self.assertEqual(len(frame), 1609)
            self.assertEqual(len(ds.index), 1607)
            self.assertEqual(aggregate(ds, "q", denominator="all").denominators.iloc[0, 0], 1607)
            self.assertEqual(aggregate(ds, "q").counts.iloc[0, 0], 1603)
            self.assertEqual(aggregate(ds, "q", include_missing=True).counts.iloc[0, -1], 4)
        fallback = next(iter(_xml_sheets(BytesIO(raw), preserve_blank_rows=True).values()))
        self.assertEqual(len(fallback), 1609)
        fallback_bound = next(iter(_xml_sheets(BytesIO(raw), preserve_blank_rows=True, end_row=1607).values()))
        self.assertEqual(len(fallback_bound), 1607)
        with patch("survey.core.Path.read_bytes", return_value=raw):
            bound = next(iter(read_sheets("synthetic.xlsx", preserve_blank_rows=True, end_row=1607).values()))
        self.assertEqual(len(bound), 1607)

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
    def test_merged_figures_settings_and_all_exports(self):
        ds, _ = fixture()
        spec = FigureSpec("1", group="3", mode="cross", chart="heatmap", dpi=72,
                          question_merges={"A": "全回答", "B": "全回答"},
                          group_merges={"G1": "全属性", "G2": "全属性"},
                          labels={"全回答": "回答の合計"}, colors={"全回答": "#0072B2"},
                          font=default_font())
        loaded, _, _ = load_settings(save_settings(ds, [spec], {}, {}), ds)
        self.assertEqual(asdict(spec), asdict(loaded[0]))
        result, note = data_for_spec(ds, loaded[0])
        self.assertEqual(result.counts.columns.tolist(), ["全回答"])
        self.assertEqual(result.counts.index.tolist(), ["全属性"])
        self.assertIn("回答側の集約", note)
        for chart in ["bar", "heatmap", "stacked"]:
            spec.chart = chart
            fig = render(ds, spec)
            try:
                tick_labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
                self.assertIn("回答の合計" if chart == "bar" else "全属性", tick_labels)
                for fmt in ["png", "svg", "pdf"]:
                    self.assertTrue(export_figure(fig, spec, fmt))
            finally:
                plt.close(fig)
        with zipfile.ZipFile(BytesIO(batch_export(ds, [spec]))) as archive:
            self.assertEqual(len(archive.namelist()), 3)
        self.assertTrue(multipage_pdf(ds, [spec]).startswith(b"%PDF"))

    def test_merged_sorting_and_old_settings_compatibility(self):
        ds, _ = fixture()
        spec = FigureSpec("1", question_merges={"B": "少数回答"}, order="ascending", font=default_font())
        fig = render(ds, spec)
        try:
            self.assertEqual([tick.get_text() for tick in fig.axes[0].get_yticklabels()], ["少数回答", "A"])
        finally:
            plt.close(fig)
        payload = json.loads(save_settings(ds, [spec], {}, {}))
        payload["figures"][0].pop("question_merges")
        payload["figures"][0].pop("group_merges")
        loaded, _, _ = load_settings(json.dumps(payload).encode(), ds)
        self.assertEqual(loaded[0].question_merges, {})
        self.assertEqual(loaded[0].group_merges, {})
        validate_spec(loaded[0], ds)

    def test_render_uses_agg_even_with_base_canvas_backend(self):
        previous = plt.get_backend()
        plt.switch_backend("template")
        try:
            # This backend reproduces the reported missing get_renderer().
            original = plt.figure()
            self.assertFalse(hasattr(original.canvas, "get_renderer"))
            plt.close(original)
            ds, _ = fixture()
            for height in [12, 60]:
                spec = FigureSpec("1", group="3", mode="cross", height=height,
                                  dpi=72, auto_height=False, font=default_font())
                fig = render(ds, spec)
                try:
                    self.assertIsInstance(fig.canvas, FigureCanvasAgg)
                    for fmt in ["png", "svg", "pdf"]:
                        self.assertTrue(export_figure(fig, spec, fmt))
                finally:
                    plt.close(fig)
        finally:
            plt.switch_backend(previous)

    def test_dense_cross_value_labels_do_not_overlap(self):
        ds = dense_fixture()
        spec = FigureSpec("q", group="g", mode="cross", metric="count",
                          height=12, auto_height=True, font=default_font())
        fig = render(ds, spec)
        try:
            self.assertGreater(fig.get_figheight() * 2.54, 12)
            self.assertLessEqual(fig.get_figheight() * 2.54, 60)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            labels = sorted((label.get_window_extent(renderer) for label in fig.axes[0].texts), key=lambda box: box.y0)
            self.assertEqual(len(labels), 60)
            for first, second in zip(labels, labels[1:]):
                self.assertLess(first.y1, second.y0)
            axis = fig.axes[0]
            legend_gap = (axis.xaxis.label.get_window_extent(renderer).y0 - axis.get_legend().get_window_extent(renderer).y1) / fig.dpi
            self.assertGreater(legend_gap, 0)
            self.assertLess(legend_gap, 1)
            for fmt in ["png", "svg", "pdf"]:
                self.assertTrue(export_figure(fig, spec, fmt, dpi=72))
        finally:
            plt.close(fig)
        spec.auto_height = False
        fig = render(ds, spec)
        try:
            self.assertAlmostEqual(fig.get_figheight() * 2.54, 12)
        finally:
            plt.close(fig)

    def test_oversized_png_keeps_preview_and_vector_exports(self):
        ds, _ = fixture()
        spec = FigureSpec("1", width=60, height=60, dpi=600, auto_height=False,
                          font=default_font())
        fig = render(ds, spec)
        try:
            with self.assertRaisesRegex(SurveyError, "PNGのサイズが大きすぎます"):
                export_figure(fig, spec, "png")
            self.assertTrue(export_figure(fig, spec, "png", dpi=72).startswith(b"\x89PNG"))
            for fmt in ["svg", "pdf"]:
                self.assertTrue(export_figure(fig, spec, fmt))
        finally:
            plt.close(fig)

    def test_old_settings_default_layout_controls(self):
        ds, _ = fixture()
        payload = json.loads(save_settings(ds, [FigureSpec("1")], {}, {}))
        payload["figures"][0].pop("auto_height")
        payload["figures"][0].pop("value_font_size")
        specs, _, _ = load_settings(json.dumps(payload).encode(), ds)
        self.assertTrue(specs[0].auto_height)
        self.assertEqual(specs[0].value_font_size, 9)

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
            loaded = next(iter(read_sheets(str(path), "test-password", preserve_blank_rows=True).values()))
            decrypted = build_dataset(ds.questions, loaded)
            self.assertFalse(decrypted.problems)
            self.assertEqual(len(decrypted.index), 5)
            self.assertEqual(aggregate(decrypted, "2").counts.iloc[0].tolist(), [2, 2])
            with self.assertRaises(SurveyError):
                read_sheets(str(path), "wrong")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in directory.iterdir()], ["encrypted.xlsx"])
        finally:
            path.unlink(missing_ok=True)
            directory.rmdir()


class AppTests(unittest.TestCase):
    def test_merge_tables_restore_edit_reset_and_question_switch(self):
        from streamlit.testing.v1 import AppTest
        ds, frame = fixture()
        definitions_frame = pd.DataFrame(
            [["No", "質問内容", "回答方式", "選択肢"]]
            + [[q.id, q.title, {"single": "選択", "multi": "複数選択", "numeric": "整数入力"}[q.kind], "\n".join(q.choices)] for q in ds.questions]
        )
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
        app.session_state.sheets = ({"定義": definitions_frame}, {"回答": frame})
        app.run()
        self.assertFalse(app.exception)
        spec = FigureSpec("1", group="5", mode="cross", font=default_font(),
                          question_merges={"A": "まとめ", "B": "まとめ"},
                          group_merges={"M": "全属性", "N": "全属性"})
        app.session_state.figures = [spec]
        app.run()
        next(b for b in app.button if b.label == "編集").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.question_merges, spec.question_merges)
        self.assertEqual(app.session_state.active_spec.group_merges, spec.group_merges)
        tables = [table.value for table in app.dataframe]
        self.assertTrue(any(list(table.columns) == ["元の選択肢", "集約後の区分名"] and table["集約後の区分名"].tolist() == ["まとめ", "まとめ"] for table in tables))
        next(b for b in app.button if b.label == "回答側を元に戻す").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.question_merges, {})
        self.assertEqual(app.session_state.active_spec.group_merges, spec.group_merges)
        self.assertEqual(app.session_state.figures[0].question_merges, spec.question_merges)
        app.session_state["editor_merges_question_1"] = {
            "edited_rows": {0: {"集約後の区分名": "新区分"}, 1: {"集約後の区分名": "新区分"}},
            "added_rows": [], "deleted_rows": [],
        }
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.question_merges, {"A": "新区分", "B": "新区分"})
        app.run()
        self.assertEqual(app.session_state.active_spec.question_merges, {"A": "新区分", "B": "新区分"})
        next(s for s in app.selectbox if s.label == "表示する設問").select("2").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.question_merges, {})
        self.assertEqual(app.session_state.active_spec.group_merges, spec.group_merges)
        next(b for b in app.button if b.label == "属性側を元に戻す").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_spec.group_merges, {})

    def test_start_and_sample_load(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "読み込む / 再読込").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.dataset.questions), 128)
        self.assertEqual(len(app.session_state.dataset.index), 0)
        self.assertEqual(app.session_state.answer_password, "")
        next(n for n in app.number_input if n.label == "回答の最終行（0で自動）").set_value(1609)
        next(b for b in app.button if b.label == "読み込む / 再読込").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.dataset.index), 1607)
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
