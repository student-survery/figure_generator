from __future__ import annotations

from pathlib import Path
import copy

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from survey.core import (Dataset, SurveyError, definitions, read_sheets, map_questions,
                         build_dataset, header_groups, MISSING)
from survey.charts import (FigureSpec, PALETTES, fonts, default_font, data_for_spec,
                           render, export_figure, batch_export, multipage_pdf,
                           save_settings, load_settings, validate_spec)


st.set_page_config(page_title="アンケート図表室", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent


def clear_results():
    for key in list(st.session_state):
        if key.startswith(("editor_", "mapping_", "alias_", "filter_", "choice_", "settings_", "batch_")) or key in {"dataset", "dataset_revision", "figures", "active_spec", "edit_index", "overrides", "choice_overrides", "aliases", "sheet_signature", "revision"}:
            del st.session_state[key]


def load_files():
    clear_results()
    st.session_state.pop("sheets", None)
    st.session_state.pop("definition_sheet", None)
    st.session_state.pop("answer_sheet", None)
    try:
        q = read_sheets(st.session_state["definition_path"], st.session_state.get("definition_password", ""))
        a = read_sheets(st.session_state["answer_path"], st.session_state.get("answer_password", ""))
        st.session_state.sheets = (q, a)
        st.session_state.pop("load_error", None)
    except SurveyError as exc:
        st.session_state.load_error = str(exc)
    finally:
        st.session_state.definition_password = ""
        st.session_state.answer_password = ""


def clear_files():
    clear_results()
    for key in ["sheets", "load_error", "definition_password", "answer_password", "definition_path", "answer_path", "definition_sheet", "answer_sheet"]:
        st.session_state.pop(key, None)


def get_dataset(def_frame, answer_frame) -> Dataset:
    revision = st.session_state.get("revision", 0)
    if st.session_state.get("dataset_revision") == revision:
        return st.session_state.dataset
    questions = definitions(def_frame)
    map_questions(questions, answer_frame, st.session_state.get("overrides", {}))
    for q in questions:
        correction = st.session_state.get("choice_overrides", {}).get(q.id)
        if correction and q.columns:
            mapped = {int(k): v for k, v in correction.items()}
            if set(mapped) != set(q.columns) or any(c not in q.choices for c in mapped.values()) or len(set(mapped.values())) != len(mapped):
                q.issue = "選択肢の対応付けに重複・未設定・不正な値があります。"
            else:
                q.column_choices = mapped
                q.issue = "" if q.kind == "multi" else q.issue
    ds = build_dataset(questions, answer_frame, st.session_state.get("aliases", {}))
    st.session_state.dataset, st.session_state.dataset_revision = ds, revision
    return ds


def changed():
    st.session_state.revision = st.session_state.get("revision", 0) + 1
    for key in list(st.session_state):
        if key.startswith("batch_"):
            del st.session_state[key]


def apply_settings(raw, ds, def_frame, answer_frame):
    specs, mappings, choice_maps = load_settings(raw, ds)
    # Validate against a temporary dataset; failed imports do not change current state.
    questions = definitions(def_frame)
    groups = map_questions(questions, answer_frame, mappings)
    for qid, number in mappings.items():
        if qid not in {q.id for q in questions} or (number != "" and number not in groups):
            raise SurveyError("設定の設問対応付けが不正です。")
    for qid, choices in choice_maps.items():
        q = next((q for q in questions if q.id == qid), None)
        if not q or q.kind != "multi" or not isinstance(choices, dict):
            raise SurveyError("設定の選択肢対応付けが不正です。")
        try:
            mapped = {int(k): v for k, v in choices.items()}
        except (ValueError, TypeError):
            raise SurveyError("設定の選択肢対応付けが不正です。") from None
        if set(mapped) != set(q.columns) or any(v not in q.choices for v in mapped.values()) or len(set(mapped.values())) != len(mapped):
            raise SurveyError("設定の選択肢対応付けに重複・未設定があります。")
        q.column_choices, q.issue = mapped, ""
    candidate = build_dataset(questions, answer_frame, st.session_state.get("aliases", {}))
    for spec in specs:
        validate_spec(spec, candidate)
    clear_editor()
    st.session_state.pop("edit_index", None)
    st.session_state.overrides, st.session_state.choice_overrides = mappings, choice_maps
    st.session_state.figures = specs
    st.session_state.active_spec = specs[0] if specs else None
    changed()


def clear_editor():
    for key in list(st.session_state):
        if key.startswith(("editor_", "filter_")):
            del st.session_state[key]


def edit_figure(spec, index):
    clear_editor()
    st.session_state.active_spec = copy.deepcopy(spec)
    st.session_state.edit_index = index


def select(label, options, default, key, **kwargs):
    return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key, **kwargs)


def editor(ds: Dataset):
    available = ds.available()
    if not available or len(ds.index) == 0:
        st.info("回答データと対応付けを確認すると、ここで図を作成できます。サンプルには回答行がありません。")
        return
    old = st.session_state.get("active_spec") or FigureSpec(available[0].id, font=default_font())
    ids = [q.id for q in available]
    qlabel = lambda qid: ds.question(qid).label
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("集計条件")
        mode = select("集計方法", ["single", "attribute", "cross"], old.mode, "editor_mode", format_func=lambda m: {"single": "設問単独", "attribute": "属性別", "cross": "クロス集計"}[m])
        eligible = ids if mode == "single" else [q.id for q in available if q.kind in ("single", "multi")]
        if not eligible:
            st.warning("このモードには選択式の設問が必要です。")
            return
        if st.session_state.get("editor_question") not in eligible:
            st.session_state.pop("editor_question", None)
        qid = select("表示する設問", eligible, old.question, "editor_question", format_func=qlabel)
        q = ds.question(qid)
        group_id = None
        if mode != "single":
            group_ids = [qq.id for qq in available if qq.kind in ("single", "multi") and qq.id != qid]
            if not group_ids:
                st.warning("比較するもう1つの選択式設問が必要です。")
                return
            if st.session_state.get("editor_group") not in group_ids:
                st.session_state.pop("editor_group", None)
            group_id = select("属性 / クロス集計の行設問", group_ids, old.group, "editor_group", format_func=qlabel)
        spec = copy.deepcopy(old)
        spec.question, spec.group, spec.mode = qid, group_id, mode
        filter_ids = st.multiselect("絞り込みに使う設問", ids, default=[f["question"] for f in old.filters if f["question"] in ids], format_func=qlabel, key="filter_ids")
        spec.filters = []
        for fid in filter_ids:
            fq = ds.question(fid)
            before = next((f for f in old.filters if f["question"] == fid), {})
            if fq.kind == "numeric":
                lo = st.number_input(f"No.{fq.number} 最小", value=float(before.get("min", 0)), key=f"filter_{fid}_min")
                hi = st.number_input(f"No.{fq.number} 最大", value=float(before.get("max", 168)), key=f"filter_{fid}_max")
                spec.filters.append({"question": fid, "min": lo, "max": hi})
            else:
                choices = st.multiselect(f"No.{fq.number} の対象回答（OR）", fq.choices + [MISSING], default=before.get("choices", []), key=f"filter_{fid}_choices")
                spec.filters.append({"question": fid, "choices": choices})
        st.caption("条件間はAND、同じ設問内の選択肢はOR。空の選択条件では対象者0人になります。")
        if q.kind == "numeric":
            spec.chart, spec.metric, spec.percent_mode = "histogram", "count", "row"
            spec.bins = st.number_input("ヒストグラムの区間数", 1, 100, int(old.bins), key="editor_bins")
        else:
            spec.metric = select("表示値", ["percent", "count"], old.metric, "editor_metric", format_func=lambda m: "割合（％）" if m == "percent" else "人数")
            spec.denominator = select("割合の分母", ["valid", "all"], old.denominator, "editor_denominator", format_func=lambda d: "有効回答者数" if d == "valid" else "対象者全員")
            spec.include_missing = st.checkbox("無回答をカテゴリーに含める", value=old.include_missing, key="editor_missing")
            modes = ["row", "column", "overall"] if mode == "cross" and q.kind == "single" and ds.question(group_id).kind == "single" else ["row"]
            if st.session_state.get("editor_percent_mode") not in modes:
                st.session_state.pop("editor_percent_mode", None)
            spec.percent_mode = select("割合の方向", modes, old.percent_mode, "editor_percent_mode", format_func=lambda m: {"row": "行割合（グループ内）", "column": "列割合（回答選択肢内）", "overall": "全体割合"}[m])
            charts = ["bar"]
            if mode != "single":
                charts.append("heatmap")
            if q.kind == "single" and spec.metric == "percent" and spec.percent_mode == "row" and (spec.denominator == "valid" or spec.include_missing):
                charts.append("stacked")
            if st.session_state.get("editor_chart") not in charts:
                st.session_state.pop("editor_chart", None)
            spec.chart = select("グラフの種類", charts, old.chart, "editor_chart", format_func=lambda c: {"bar": "横棒グラフ", "heatmap": "ヒートマップ", "stacked": "100％積み上げ棒"}[c])
        with st.expander("デザイン", expanded=True):
            spec.title = st.text_input("図のタイトル（空欄で設問名）", old.title, key="editor_title")
            spec.palette = select("配色", list(PALETTES), old.palette, "editor_palette")
            installed = fonts()
            spec.font = select("フォント", installed, old.font or default_font(), "editor_font")
            if default_font() == "DejaVu Sans":
                st.warning("日本語フォントが見つかりません。日本語フォントをインストールしてアプリを再起動してください。")
            spec.font_size = st.slider("文字サイズ（pt）", 6, 30, int(old.font_size), key="editor_font_size")
            a, b = st.columns(2)
            spec.width = a.number_input("幅（cm）", 8.0, 60.0, float(old.width), .5, key="editor_width")
            spec.height = b.number_input("高さ（cm）", 6.0, 60.0, float(old.height), .5, key="editor_height")
            spec.auto_height = st.checkbox("棒グラフの高さを自動調整", old.auto_height, key="editor_auto_height", help="指定した高さを最小値とし、カテゴリー・グループ数と文字の高さに応じて最大60cmまで広げます。")
            spec.dpi = st.number_input("PNG解像度（dpi）", 72, 600, int(old.dpi), 50, key="editor_dpi")
            spec.order = select("選択肢の並び順", ["definition", "descending", "ascending"], old.order, "editor_order", format_func=lambda o: {"definition": "設問定義順", "descending": "人数の多い順", "ascending": "人数の少ない順"}[o])
            spec.show_values = st.checkbox("値ラベル", old.show_values, key="editor_values")
            spec.value_font_size = st.slider("値ラベルの文字サイズ（pt）", 6, 30, int(old.value_font_size), key="editor_value_font_size")
            spec.legend = st.checkbox("凡例", old.legend, key="editor_legend")
            spec.grid = st.checkbox("補助線", old.grid, key="editor_grid")
            spec.transparent = st.checkbox("背景を透明にする", old.transparent, key="editor_transparent")
            spec.background = st.color_picker("背景色", old.background, key="editor_background")
            spec.wrap = st.slider("ラベルの折り返し文字数", 8, 60, int(old.wrap), key="editor_wrap")
            spec.padding = st.slider("図の余白", .2, 4.0, float(old.padding), .1, key="editor_padding")
            spec.svg_text = st.checkbox("SVGの文字をテキストとして残す", old.svg_text, key="editor_svg_text", help="標準はパス化です。テキストを残す場合、閲覧先でも同じフォントが必要です。")
        labels = list(dict.fromkeys(q.choices + ([MISSING] if spec.include_missing else []) + (ds.question(group_id).choices if group_id else []))) if q.kind != "numeric" else [q.title]
        with st.expander("選択肢の色・表示名"):
            rows = []
            palette = PALETTES[spec.palette]
            for i, c in enumerate(labels):
                rows.append({"元のラベル": c, "表示名": old.labels.get(c, c), "色": old.colors.get(c, palette[i % len(palette)])})
            # Recreate on question/palette changes, preserving saved figure defaults.
            revision_key = f"editor_labels_{qid}_{group_id}_{spec.palette}_{spec.include_missing}"
            edited = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["元のラベル"], key=revision_key, width="stretch")
            spec.labels = {r["元のラベル"]: str(r["表示名"]) for _, r in edited.iterrows() if str(r["表示名"]) != r["元のラベル"]}
            spec.colors = {r["元のラベル"]: str(r["色"]) for i, r in edited.iterrows() if str(r["色"]).upper() != palette[i % len(palette)].upper()}
        edit_index = st.session_state.get("edit_index")
        if edit_index is not None and edit_index < len(st.session_state.get("figures", [])):
            st.caption(f"一覧の図 {edit_index + 1} を編集中。更新後は「図の一覧」で確認できます。")
            if st.button("一覧のこの図を更新", width="stretch"):
                try:
                    validate_spec(spec, ds)
                    st.session_state.figures[edit_index] = copy.deepcopy(spec)
                    changed()
                    st.success("一覧の図を更新しました。")
                except SurveyError as exc:
                    st.error(str(exc))
        if st.button("この図を一覧に追加", type="primary", width="stretch"):
            try:
                validate_spec(spec, ds)
                if len(st.session_state.get("figures", [])) >= 200:
                    raise SurveyError("一覧は200図までです。")
                st.session_state.setdefault("figures", []).append(copy.deepcopy(spec))
                changed()
                st.success("図の一覧に追加しました。")
            except (SurveyError, ValueError, TypeError, KeyError) as exc:
                st.error(str(exc) if isinstance(exc, SurveyError) else "図の設定を確認してください。")
    with right:
        st.subheader("プレビューと出力")
        fig = None
        try:
            result, note = data_for_spec(ds, spec)
            fig = render(ds, spec)
            st.image(export_figure(fig, spec, "png", dpi=min(spec.dpi, 120)), width="stretch")
            if fig.get_figheight() * 2.54 > spec.height + .05:
                st.caption(f"ラベルの重なりを抑えるため、高さを {fig.get_figheight() * 2.54:.1f}cm に自動調整しています。")
            st.caption(note)
            buttons = st.columns(3)
            for column, fmt in zip(buttons, ("png", "svg", "pdf")):
                try:
                    output = export_figure(fig, spec, fmt)
                    column.download_button(fmt.upper(), output, f"survey_No{q.number}.{fmt}", mime={"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[fmt], width="stretch")
                except SurveyError as exc:
                    column.warning(str(exc))
            if q.kind == "numeric":
                st.dataframe(result.describe().rename("値"), width="stretch")
            else:
                with st.expander("集計表・分母", expanded=True):
                    st.write("件数")
                    st.dataframe(result.counts, width="stretch")
                    st.write("割合（％）／分母0は —")
                    st.dataframe(result.percents.round(2), width="stretch")
                    st.write("割合の分母（人数）")
                    st.dataframe(result.denominators.astype(int), width="stretch")
            st.session_state.active_spec = spec
        except SurveyError as exc:
            st.error(str(exc))
        except (ValueError, TypeError, KeyError, StopIteration):
            st.error("この条件で図を作成できません。設問・フィルター・色・フォントの設定を確認してください。")
        finally:
            if fig is not None:
                plt.close(fig)


st.title("アンケート図表室")
st.caption("Box上のExcelから、報告書に使える図を作成します。回答データはこのPCのメモリ上で処理します。")
with st.sidebar:
    st.subheader("ファイル読み込み")
    with st.form("load_form"):
        st.text_input("設問定義のパス", value=str(ROOT / "question_explanation.xlsx"), key="definition_path")
        st.text_input("設問定義のパスワード（必要な場合）", type="password", key="definition_password")
        st.text_input("回答ファイルのパス", value=str(ROOT / "sample.xlsx"), key="answer_path")
        st.text_input("回答ファイルのパスワード", type="password", key="answer_password")
        st.form_submit_button("読み込む / 再読込", on_click=load_files, type="primary", width="stretch")
    st.button("データと設定をクリア", on_click=clear_files, width="stretch")
    st.caption("元Excelは読み取り専用。パスワードは読み込み後に消去します。終了時は起動したターミナルで Ctrl+C。")

if st.session_state.get("load_error"):
    st.error(st.session_state.load_error)
if "sheets" not in st.session_state:
    st.info("左のパスとパスワードを指定して「読み込む」を押してください。BoxのURLではなく、Box Drive等のファイルパスを指定します。")
    st.stop()

definition_sheets, answer_sheets = st.session_state.sheets
with st.sidebar:
    q_sheet = st.selectbox("設問定義のシート", list(definition_sheets), key="definition_sheet")
    a_sheet = st.selectbox("回答のシート", list(answer_sheets), key="answer_sheet")
signature = (q_sheet, a_sheet)
if st.session_state.get("sheet_signature") != signature:
    clear_results()
    st.session_state.sheet_signature = signature
def_frame, answer_frame = definition_sheets[q_sheet], answer_sheets[a_sheet]
try:
    ds = get_dataset(def_frame, answer_frame)
except SurveyError as exc:
    st.error(str(exc))
    st.stop()

data_tab, chart_tab, list_tab = st.tabs(["① データ確認", "② 図を作成", "③ 図の一覧・設定"])
with data_tab:
    a, b, c = st.columns(3)
    a.metric("回答者数", len(ds.index))
    b.metric("設問数", len(ds.questions))
    c.metric("確認が必要な設問", sum(bool(q.issue or q.id in ds.problems) for q in ds.questions))
    if len(ds.index) == 0:
        st.info("回答行がありません。sample.xlsx は2行のヘッダーだけの形式見本です。実データのパスを指定してください。")
    rows = [{"回答No.": q.number, "設問": (q.parent + " / " if q.parent else "") + q.title, "形式": {"single": "単一選択", "multi": "複数選択", "numeric": "数値", "text": "自由記述"}[q.kind], "状態": q.issue or ("回答値の確認が必要" if q.id in ds.problems else "対応済み")} for q in ds.questions]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("設問の対応付けを修正"):
        groups = header_groups(answer_frame)
        st.caption("定義の番号と回答No.は別物です。同じ回答No.を複数の定義に指定すると集計を停止します。")
        table = pd.DataFrame([{"定義ID": q.id, "設問": q.title, "回答No.": q.number} for q in ds.questions])
        edited = st.data_editor(table, disabled=["定義ID", "設問"], hide_index=True, column_config={"回答No.": st.column_config.SelectboxColumn(options=[""] + list(groups))}, key=f"mapping_table_{st.session_state.get('revision', 0)}", width="stretch")
        st.dataframe(pd.DataFrame([{"回答No.": n, "設問": g["title"]} for n, g in groups.items()]), hide_index=True, width="stretch")
        if st.button("設問の対応付けを適用"):
            st.session_state.overrides = {str(r["定義ID"]): str(r["回答No."] or "") for _, r in edited.iterrows()}
            st.session_state.choice_overrides = {}
            changed()
            st.rerun()
    multi = [q for q in ds.questions if q.kind == "multi" and q.columns]
    if multi:
        with st.expander("複数選択列の対応付けを修正"):
            mqid = st.selectbox("修正する設問", [q.id for q in multi], format_func=lambda i: ds.question(i).label, key="choice_question")
            mq = ds.question(mqid)
            correction = {}
            for col in mq.columns:
                raw_label = groups[mq.number]["choices"].get(col, "（2行目が空欄）")
                current = mq.column_choices.get(col)
                correction[str(col)] = select(f"列{col + 1}：{raw_label}", mq.choices, current, f"choice_{mqid}_{col}")
            if st.button("選択肢の対応付けを適用"):
                st.session_state.setdefault("choice_overrides", {})[mqid] = correction
                changed()
                st.rerun()
    for qid, unknown in ds.problems.items():
        q = ds.question(qid)
        with st.expander(f"回答値の確認：{q.label}"):
            if q.kind == "numeric":
                st.error(unknown[0])
            else:
                st.caption("下記は読み込み中の回答値です。補正はメモリ上だけで適用し、設定ファイルには保存しません。")
                aliases = {}
                for i, value in enumerate(dict.fromkeys(unknown)):
                    choices = [""] + q.choices if q.kind == "single" else [""] + list(dict.fromkeys(q.column_choices.values()))
                    selected = st.selectbox(f"未認識の値：{value}", choices, key=f"alias_{qid}_{i}")
                    if selected:
                        aliases[value] = selected
                if st.button("回答値の補正を適用", key=f"alias_apply_{qid}"):
                    st.session_state.setdefault("aliases", {}).setdefault(qid, {}).update(aliases)
                    changed()
                    st.rerun()
    st.caption("自由記述はグラフ対象外です。複数選択の全列空欄から、分岐による対象外と未回答は判別しません。")

with chart_tab:
    editor(ds)

with list_tab:
    st.subheader("保存した図")
    figures = st.session_state.setdefault("figures", [])
    if not figures:
        st.info("「図を作成」で図を一覧に追加すると、まとめて出力できます。")
    for i, spec in enumerate(figures):
        cols = st.columns([6, 1, 1, 1, 1])
        cols[0].write(f"{i + 1}. {spec.title or ds.question(spec.question).label}")
        cols[1].button("編集", key=f"list_edit_{i}", on_click=edit_figure, args=(spec, i))
        if cols[2].button("↑", key=f"list_up_{i}", disabled=i == 0):
            figures[i - 1], figures[i] = figures[i], figures[i - 1]
            st.session_state.pop("edit_index", None)
            changed()
            st.rerun()
        if cols[3].button("↓", key=f"list_down_{i}", disabled=i == len(figures) - 1):
            figures[i + 1], figures[i] = figures[i], figures[i + 1]
            st.session_state.pop("edit_index", None)
            changed()
            st.rerun()
        if cols[4].button("削除", key=f"list_delete_{i}"):
            figures.pop(i)
            st.session_state.pop("edit_index", None)
            changed()
            st.rerun()
    if figures:
        if st.button("一括出力を生成", type="primary"):
            try:
                with st.spinner("図を出力しています…"):
                    st.session_state.batch_zip = batch_export(ds, figures)
                    st.session_state.batch_pdf = multipage_pdf(ds, figures)
            except (SurveyError, ValueError, TypeError, KeyError):
                st.error("一括出力できません。各図の設定と現在のデータを確認してください。")
        if "batch_zip" in st.session_state:
            st.download_button("PNG・SVG・個別PDFのZIP", st.session_state.batch_zip, "survey_figures.zip", "application/zip")
        if "batch_pdf" in st.session_state:
            st.download_button("複数ページPDF", st.session_state.batch_pdf, "survey_figures.pdf", "application/pdf")
    st.subheader("設定ファイル")
    # Save the current editor too when no figures have yet been added.
    saved = figures or ([st.session_state.active_spec] if st.session_state.get("active_spec") else [])
    st.download_button("設定をJSONで保存", save_settings(ds, saved, st.session_state.get("overrides", {}), st.session_state.get("choice_overrides", {})), "survey_settings.json", "application/json")
    st.caption("保存するのは図の条件・デザイン・対応付けだけです。回答、実データのパス、パスワード、回答値の補正は含みません。")
    upload = st.file_uploader("設定JSONを読み込む", type=["json"], key="settings_upload")
    if upload and st.button("設定を適用"):
        try:
            apply_settings(upload.getvalue(), ds, def_frame, answer_frame)
            st.rerun()
        except (SurveyError, ValueError, TypeError, KeyError):
            st.error("設定を読み込めません。設問構造・フォント・図の条件・JSON形式を確認してください。")
