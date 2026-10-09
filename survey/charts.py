from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
import math
import re
import textwrap
from threading import RLock
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from .core import Dataset, SurveyError, aggregate, filter_index, MISSING, validate_merges


_DRAW_LOCK = RLock()
MAX_PNG_PIXELS = 40_000_000
MAX_PNG_SIDE = 16_000


PALETTES = {
    "標準": ["#2563EB", "#16A085", "#F4B942", "#EA7464", "#8B6EC8", "#536779", "#9AB3CA", "#DDA0B4"],
    "色覚に配慮": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#555555"],
    "落ち着いた色": ["#446785", "#769D91", "#BC9A66", "#AA7770", "#8B819B", "#6D7780"],
    "モノクロ": ["#252525", "#525252", "#737373", "#969696", "#BDBDBD", "#D9D9D9"],
}


def fonts() -> list[str]:
    return sorted({f.name for f in font_manager.fontManager.ttflist})


def default_font() -> str:
    installed = fonts()
    for name in ["Meiryo", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "MS Gothic"]:
        if name in installed:
            return name
    return "DejaVu Sans"


@dataclass
class FigureSpec:
    question: str
    group: str | None = None
    mode: str = "single"
    filters: list[dict] = field(default_factory=list)
    chart: str = "bar"
    metric: str = "percent"
    denominator: str = "valid"
    percent_mode: str = "row"
    include_missing: bool = False
    title: str = ""
    palette: str = "標準"
    colors: dict[str, str] = field(default_factory=dict)
    font: str = ""
    font_size: float = 10
    width: float = 18
    height: float = 12
    dpi: int = 300
    labels: dict[str, str] = field(default_factory=dict)
    order: str = "definition"
    show_values: bool = True
    legend: bool = True
    transparent: bool = False
    background: str = "#FFFFFF"
    grid: bool = True
    wrap: int = 24
    padding: float = 1.2
    bins: int = 10
    svg_text: bool = False
    auto_height: bool = True
    value_font_size: float = 9
    question_merges: dict[str, str] = field(default_factory=dict)
    group_merges: dict[str, str] = field(default_factory=dict)


def validate_spec(spec: FigureSpec, ds: Dataset):
    for value in [spec.question, spec.mode, spec.chart, spec.metric, spec.denominator, spec.percent_mode, spec.title, spec.palette, spec.font, spec.order, spec.background]:
        if not isinstance(value, str):
            raise SurveyError("図の設定値の形式が不正です。")
    if spec.group is not None and not isinstance(spec.group, str):
        raise SurveyError("グループ指定の形式が不正です。")
    for mapping in [spec.colors, spec.labels]:
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise SurveyError("色・ラベル設定の形式が不正です。")
    if not all(isinstance(value, bool) for value in [spec.include_missing, spec.show_values, spec.legend, spec.transparent, spec.grid, spec.svg_text, spec.auto_height]):
        raise SurveyError("図の表示設定の形式が不正です。")
    available = {q.id: q for q in ds.available()}
    if spec.question not in available or (spec.group and spec.group not in available):
        raise SurveyError("図に指定された設問は現在のデータでは使用できません。")
    q = available[spec.question]
    validate_merges(q, spec.question_merges)
    if not isinstance(spec.group_merges, dict):
        raise SurveyError("属性側の集約設定の形式が不正です。")
    if spec.group:
        validate_merges(available[spec.group], spec.group_merges)
    elif spec.group_merges:
        raise SurveyError("属性側の集約には属性の設問を指定してください。")
    if spec.mode not in ("single", "attribute", "cross") or spec.chart not in ("bar", "stacked", "heatmap", "histogram"):
        raise SurveyError("図の形式が不正です。")
    if (spec.mode == "single" and spec.group) or (spec.mode != "single" and not spec.group):
        raise SurveyError("集計モードとグループ指定が一致しません。")
    if spec.group and available[spec.group].kind not in ("single", "multi"):
        raise SurveyError("グループには選択式の設問を指定してください。")
    if q.kind == "numeric":
        if spec.chart != "histogram" or spec.mode != "single":
            raise SurveyError("数値回答は単独のヒストグラムで表示します。")
    elif spec.chart == "histogram":
        raise SurveyError("ヒストグラムには数値回答を指定してください。")
    if spec.chart == "stacked" and (q.kind != "single" or spec.metric != "percent" or spec.percent_mode != "row"):
        raise SurveyError("100％積み上げ棒は単一選択の行割合で使用できます。")
    if spec.metric not in ("count", "percent") or spec.denominator not in ("valid", "all") or spec.percent_mode not in ("row", "column", "overall") or spec.order not in ("definition", "ascending", "descending"):
        raise SurveyError("図の集計設定が不正です。")
    if spec.chart == "stacked" and spec.denominator == "all" and not spec.include_missing:
        raise SurveyError("対象者全員を分母にする100％積み上げ棒では、無回答も表示してください。")
    for v, low, high in [(spec.width, 8, 60), (spec.height, 6, 60), (spec.font_size, 6, 30), (spec.value_font_size, 6, 30), (spec.dpi, 72, 600), (spec.wrap, 8, 60), (spec.padding, .2, 4), (spec.bins, 1, 100)]:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise SurveyError("図の寸法・解像度・文字サイズが範囲外です。")
    if spec.palette not in PALETTES or not colors.is_color_like(spec.background) or any(not colors.is_color_like(v) for v in spec.colors.values()):
        raise SurveyError("配色設定が不正です。")
    if spec.font and spec.font not in fonts():
        raise SurveyError("指定フォントがインストールされていません。デザイン画面で選び直してください。")
    # Check filters even when a saved figure is loaded without opening the editor.
    filter_index(ds, spec.filters)


def filter_note(ds, spec):
    parts = []
    for f in spec.filters:
        q = ds.question(f["question"])
        condition = f"{f['min']}～{f['max']}" if q.kind == "numeric" else " / ".join(f.get("choices", []))
        parts.append(f"No.{q.number} {condition}")
    return "絞り込み：" + ("、".join(parts) if parts else "なし")


def data_for_spec(ds: Dataset, spec: FigureSpec):
    validate_spec(spec, ds)
    if ds.question(spec.question).kind == "numeric":
        idx = filter_index(ds, spec.filters)
        series = ds.values[spec.question].loc[idx].dropna()
        return series, f"対象者数 N={len(idx)}、有効回答 n={len(series)}、無回答 {len(idx) - len(series)}。" + filter_note(ds, spec)
    result = aggregate(ds, spec.question, spec.group, spec.filters, spec.denominator, spec.include_missing, spec.percent_mode,
                       question_merges=spec.question_merges, group_merges=spec.group_merges)
    note = result.note + filter_note(ds, spec)
    den = result.denominators
    if spec.percent_mode == "row":
        note += " 分母 n：" + "、".join(f"{g}={int(den.loc[g].iloc[0])}" for g in den.index)
    elif spec.percent_mode == "column":
        note += " 分母 n：" + "、".join(f"{c}={int(den[c].iloc[0])}" for c in den.columns)
    else:
        note += f" 分母 n={int(den.iloc[0, 0])}"
    return result, note


def _fit_bar_height(fig, ax, canvas, spec, bar_pitch, annotations):
    """Reserve physical vertical space for each value label and category tick."""
    for _ in range(5):
        fig.tight_layout(pad=spec.padding)
        canvas.draw()
        renderer = canvas.get_renderer()
        if not spec.auto_height:
            return renderer
        span = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        current = ax.get_window_extent(renderer).height
        value_height = max((a.get_window_extent(renderer).height for a in annotations), default=0)
        tick_height = max((t.get_window_extent(renderer).height for t in ax.get_yticklabels()), default=0)
        required = max(value_height * 1.35 * span / bar_pitch, tick_height * 1.25 * span)
        if required <= current + .5:
            return renderer
        if fig.get_figheight() * 2.54 >= 60 - .01:
            raise SurveyError("棒グラフのラベルを配置するには高さ60cmを超えます。値ラベルの文字サイズを小さくする、値ラベルを非表示にする、またはヒートマップを選んでください。")
        # Titles and legends have fixed physical height. Only increase the
        # plotting area; scaling its fraction would also enlarge those margins.
        new_height = fig.get_figheight() + (required - current) / fig.dpi + .02
        fig.set_size_inches(fig.get_figwidth(), min(new_height, 60 / 2.54), forward=False)
    raise SurveyError("棒グラフの自動高さ調整で十分な余白を確保できません。値ラベルの文字サイズを小さくするか、自動調整を解除して高さを指定してください。")


def render(ds: Dataset, spec: FigureSpec):
    data, note = data_for_spec(ds, spec)
    q = ds.question(spec.question)
    family = spec.font or default_font()
    def label(value):
        return "\n".join(textwrap.wrap(spec.labels.get(str(value), str(value)), width=spec.wrap))
    with _DRAW_LOCK, plt.rc_context({"font.family": family, "font.size": spec.font_size, "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none" if spec.svg_text else "path"}):
        # Avoid pyplot's global backend/figure registry: it can leave a base
        # canvas without get_renderer(), including across Streamlit reruns.
        fig = Figure(figsize=(spec.width / 2.54, spec.height / 2.54), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.subplots()
        legend_transform = ax.transAxes + ScaledTranslation(0, -max(.55, spec.font_size / 72 * 3 + .15), fig.dpi_scale_trans)
        annotations = []
        bar_pitch = 1.0
        fig.set_facecolor(spec.background)
        ax.set_facecolor(spec.background)
        palette = PALETTES[spec.palette]
        color = lambda name, i: spec.colors.get(str(name), palette[i % len(palette)])
        if q.kind == "numeric":
            if len(data):
                ax.hist(data, bins=int(spec.bins), color=color(q.title, 0), edgecolor="white")
                note += f" 平均={data.mean():.2f}、中央値={data.median():.2f}。"
            else:
                ax.text(.5, .5, "有効回答がありません", transform=ax.transAxes, ha="center")
            ax.set_xlabel(label(q.title))
            ax.set_ylabel("人数")
        else:
            table = (data.counts if spec.metric == "count" else data.percents).copy()
            if spec.order != "definition":
                order = data.counts.sum(axis=0).sort_values(ascending=spec.order == "ascending", kind="stable").index
                table = table.loc[:, order]
            if spec.chart == "heatmap":
                numeric = table.to_numpy(dtype=float)
                masked = np.ma.masked_invalid(numeric)
                cmap = colors.LinearSegmentedColormap.from_list("survey", ["#F3F6FA", palette[0]])
                im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
                ax.set_xticks(range(len(table.columns)), [label(c) for c in table.columns], rotation=45, ha="right")
                ax.set_yticks(range(len(table.index)), [label(g) for g in table.index])
                fig.colorbar(im, ax=ax, label="人数" if spec.metric == "count" else "割合（％）")
                if spec.show_values:
                    limit = np.nanmax(numeric) if np.isfinite(numeric).any() else 1
                    for r in range(table.shape[0]):
                        for c in range(table.shape[1]):
                            v = numeric[r, c]
                            ax.text(c, r, "—" if not np.isfinite(v) else f"{v:.0f}" if spec.metric == "count" else f"{v:.1f}%", ha="center", va="center", fontsize=spec.value_font_size, color="white" if np.isfinite(v) and v > limit * .65 else "#172033")
            elif spec.chart == "stacked":
                left = np.zeros(len(table.index))
                for i, c in enumerate(table.columns):
                    values = table[c].fillna(0).to_numpy()
                    ax.barh(range(len(table.index)), values, left=left, color=color(c, i), label=label(c), edgecolor="white", linewidth=.5)
                    if spec.show_values:
                        for j, v in enumerate(values):
                            if v >= 4:
                                rgb = colors.to_rgb(color(c, i))
                                foreground = "#172033" if sum(rgb) / 3 > .65 else "white"
                                ax.text(left[j] + v / 2, j, f"{v:.1f}%", ha="center", va="center", fontsize=spec.value_font_size, color=foreground)
                    left += values
                for j, (_, row) in enumerate(table.iterrows()):
                    if row.isna().all():
                        ax.text(50, j, "—（分母0）", ha="center", va="center")
                ax.set_yticks(range(len(table.index)), [label(g) for g in table.index])
                ax.set_xlim(0, 100)
                ax.set_xlabel("割合（％）")
                if spec.legend:
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, 0), bbox_transform=legend_transform, ncol=min(3, len(table.columns)), frameon=False)
            else:
                height = .8 / max(len(table.index), 1)
                bar_pitch = height if spec.group else 1.0
                for i, g in enumerate(table.index):
                    pos = np.arange(len(table.columns)) + (i - (len(table.index) - 1) / 2) * height
                    values = table.loc[g].fillna(0).to_numpy(dtype=float)
                    bar_colors = [color(c, j) for j, c in enumerate(table.columns)] if not spec.group else color(g, i)
                    bars = ax.barh(pos, values, height=height, color=bar_colors, label=label(g))
                    if spec.show_values:
                        raw = table.loc[g].to_numpy(dtype=float)
                        labels = ["—" if not np.isfinite(v) else f"{v:.0f}" if spec.metric == "count" else f"{v:.1f}%" for v in raw]
                        annotations.extend(ax.bar_label(bars, labels=labels, padding=3, fontsize=spec.value_font_size))
                ax.set_yticks(range(len(table.columns)), [label(c) for c in table.columns])
                ax.set_xlabel("人数" if spec.metric == "count" else "割合（％）")
                ax.margins(x=.2)
                if spec.group and spec.legend:
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, 0), bbox_transform=legend_transform, ncol=min(3, len(table.index)), frameon=False)
            ax.invert_yaxis()
        if spec.chart != "heatmap":
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y" if q.kind == "numeric" else "x", alpha=.2, visible=spec.grid)
        ax.set_title(label(spec.title or q.label), loc="left", pad=14, fontweight="bold")
        if spec.chart == "bar":
            renderer = _fit_bar_height(fig, ax, canvas, spec, bar_pitch, annotations)
        else:
            fig.tight_layout(pad=spec.padding)
            canvas.draw()
            renderer = canvas.get_renderer()
        # Position below the entire axes/legend bounding box; exports use bbox_inches=tight.
        boxes = [ax.get_tightbbox(renderer)]
        if ax.get_legend():
            boxes.append(ax.get_legend().get_window_extent(renderer))
        bottom = min(b.y0 for b in boxes) / fig.bbox.height
        note_width = max(35, int(spec.width * 5))
        fig.text(.02, bottom - .035, "\n".join(textwrap.wrap(note, note_width)), fontsize=max(6, spec.font_size - 2), va="top", family=family)
        # Mark chosen fonts and SVG behavior for export's own rc_context.
        fig._survey_export = (family, spec.svg_text)
        return fig


def export_figure(fig, spec: FigureSpec, fmt: str, *, dpi: int | None = None) -> bytes:
    if fmt not in ("png", "svg", "pdf"):
        raise SurveyError("対応していない出力形式です。")
    output = BytesIO()
    resolution = int(spec.dpi if dpi is None else dpi)
    with _DRAW_LOCK, plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none" if spec.svg_text else "path"}):
        if fmt == "png":
            canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
            canvas.draw()
            bbox = fig.get_tightbbox(canvas.get_renderer())
            width, height = math.ceil((bbox.width + .3) * resolution), math.ceil((bbox.height + .3) * resolution)
            if max(width, height) > MAX_PNG_SIDE or width * height > MAX_PNG_PIXELS:
                raise SurveyError(f"PNGのサイズが大きすぎます（約{width}×{height}px）。dpi・幅・高さを下げるか、SVG/PDFで出力してください。")
        fig.savefig(output, format=fmt, dpi=resolution, bbox_inches="tight", pad_inches=.15, transparent=spec.transparent, facecolor="none" if spec.transparent else spec.background)
    return output.getvalue()


def batch_export(ds: Dataset, specs: list[FigureSpec], formats=("png", "svg", "pdf")) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, spec in enumerate(specs, 1):
            fig = render(ds, spec)
            try:
                name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", spec.title or f"No.{ds.question(spec.question).number}")[:70].strip(". ")
                for fmt in formats:
                    archive.writestr(f"{i:03d}_{name}.{fmt}", export_figure(fig, spec, fmt))
            finally:
                plt.close(fig)
    return output.getvalue()


def multipage_pdf(ds: Dataset, specs: list[FigureSpec]) -> bytes:
    output = BytesIO()
    with _DRAW_LOCK, plt.rc_context({"pdf.fonttype": 42}), PdfPages(output) as pages:
        for spec in specs:
            fig = render(ds, spec)
            try:
                pages.savefig(fig, bbox_inches="tight", pad_inches=.15, transparent=spec.transparent, facecolor="none" if spec.transparent else spec.background)
            finally:
                plt.close(fig)
    return output.getvalue()


def save_settings(ds: Dataset, specs: list[FigureSpec], mappings: dict, choice_mappings: dict) -> bytes:
    # Deliberately exclude Dataset.values, file paths, passwords and runtime value aliases.
    payload = {"version": 1, "fingerprint": ds.fingerprint, "figures": [asdict(s) for s in specs], "mappings": mappings, "choice_mappings": choice_mappings}
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def load_settings(raw: bytes, ds: Dataset) -> tuple[list[FigureSpec], dict, dict]:
    try:
        if len(raw) > 2_000_000:
            raise SurveyError("設定ファイルが大きすぎます。")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("version") != 1:
            raise SurveyError("対応していない設定ファイルのバージョンです。")
        if payload.get("fingerprint") != ds.fingerprint:
            raise SurveyError("設定の設問構造が現在のデータと一致しません。")
        if not isinstance(payload.get("figures"), list) or len(payload["figures"]) > 200:
            raise SurveyError("図の一覧が不正です（上限200図）。")
        specs = [FigureSpec(**v) for v in payload["figures"]]
        mappings = payload.get("mappings", {})
        choice_mappings = payload.get("choice_mappings", {})
        if not isinstance(mappings, dict) or not isinstance(choice_mappings, dict):
            raise SurveyError("設問の対応付け設定が不正です。")
        return specs, mappings, choice_mappings
    except SurveyError:
        raise
    except (TypeError, ValueError, AttributeError, KeyError):
        raise SurveyError("設定ファイルの形式が不正です。") from None
