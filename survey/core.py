from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


class SurveyError(ValueError):
    """A safe message suitable for the UI; never includes raw answers."""


def norm(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def similar(a: str, b: str) -> bool:
    a, b = norm(a).rstrip("*"), norm(b).rstrip("*")
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a)))


def excel_bytes(path: str, password: str = "") -> BytesIO:
    try:
        raw = Path(path.strip().strip('"')).expanduser().read_bytes()
    except (OSError, ValueError):
        raise SurveyError("ファイルを読み込めません。パス、アクセス権、Box Driveの接続・ダウンロード状態を確認してください。") from None
    if zipfile.is_zipfile(BytesIO(raw)):
        return BytesIO(raw)
    try:
        import msoffcrypto
    except ImportError:
        raise SurveyError("暗号化Excelの読み込みには msoffcrypto-tool が必要です。セットアップを実行してください。") from None
    try:
        office = msoffcrypto.OfficeFile(BytesIO(raw))
        if not office.is_encrypted():
            raise SurveyError("対応する .xlsx または .xlsm ファイルを指定してください。")
        if not password:
            raise SurveyError("このExcelは暗号化されています。パスワードを入力してください。")
        office.load_key(password=password, verify_password=True)
        output = BytesIO()
        office.decrypt(output)
        output.seek(0)
        return output
    except SurveyError:
        raise
    except Exception:
        raise SurveyError("復号できません。パスワード、Excel形式、ファイルの破損を確認してください。") from None


def _xml_sheets(buffer: BytesIO, preserve_blank_rows=False, end_row=None) -> dict[str, pd.DataFrame]:
    """Read cached OOXML values when openpyxl is unavailable (sample inspection)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(buffer) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        result = {}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("m:sheets/m:sheet", ns):
            target = rels[sheet.get(f"{{{rns}}}id")]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            cells, max_row, max_col = {}, 0, 0
            worksheet = ET.fromstring(archive.read(target))
            if preserve_blank_rows:
                dimension = worksheet.find("m:dimension", ns)
                if dimension is not None:
                    match = re.search(r"(\d+)$", dimension.get("ref", ""))
                    if match:
                        max_row = int(match[1])
            for row in worksheet.findall("m:sheetData/m:row", ns):
                row_number = int(row.get("r", "0"))
                if preserve_blank_rows:
                    max_row = max(max_row, row_number)
                if end_row is not None and row_number > end_row:
                    continue
                for cell in row.findall("m:c", ns):
                    value = cell.find("m:v", ns)
                    if cell.get("t") == "inlineStr":
                        node = cell.find("m:is", ns)
                        v = "".join(node.itertext()) if node is not None else ""
                    elif value is not None:
                        v = strings[int(value.text)] if cell.get("t") == "s" else value.text
                    else:
                        continue
                    match = re.fullmatch(r"([A-Z]+)(\d+)", cell.get("r"))
                    col = 0
                    for char in match[1]:
                        col = col * 26 + ord(char) - 64
                    index = (int(match[2]) - 1, col - 1)
                    cells[index] = v
                    max_row, max_col = max(max_row, index[0] + 1), max(max_col, col)
            if end_row is not None:
                max_row = end_row
            frame = pd.DataFrame([[cells.get((r, c), None) for c in range(max_col)] for r in range(max_row)], dtype=object)
            result[sheet.get("name")] = frame
        return result


def read_sheets(path: str, password: str = "", *, preserve_blank_rows=False, end_row=None) -> dict[str, pd.DataFrame]:
    if end_row is not None and (isinstance(end_row, bool) or not isinstance(end_row, int) or not 2 <= end_row <= 1_048_576):
        raise SurveyError("回答の最終行は2～1048576の整数で指定してください。")
    buffer = excel_bytes(path, password)
    try:
        try:
            import openpyxl
        except ImportError:
            return _xml_sheets(buffer, preserve_blank_rows, end_row)
        if preserve_blank_rows:
            # pandas.read_excel trims trailing blank rows. Keep the worksheet's
            # saved row extent (or the explicit response end row) for answers.
            workbook = openpyxl.load_workbook(buffer, read_only=True, data_only=True, keep_links=False)
            try:
                result = {}
                for sheet in workbook.worksheets:
                    last_row = end_row if end_row is not None else sheet.max_row
                    frame = pd.DataFrame(
                        [[None if cell.data_type == "e" else cell.value for cell in row]
                         for row in sheet.iter_rows(max_row=last_row)],
                        dtype=object,
                    )
                    # The read-only iterator may stop at the final XML row,
                    # even when the saved/explicit extent continues beyond it.
                    if last_row is not None:
                        frame = frame.reindex(index=range(last_row))
                    result[sheet.title] = frame
                return result
            finally:
                workbook.close()
        return pd.read_excel(buffer, sheet_name=None, header=None, dtype=object, engine="openpyxl", keep_default_na=False)
    except Exception:
        raise SurveyError("Excelの内容を読み込めません。対応する .xlsx または .xlsm ファイルか確認してください。") from None
    finally:
        buffer.close()


@dataclass
class Question:
    id: str
    title: str
    kind: str
    choices: list[str]
    section: str = ""
    parent: str = ""
    number: str = ""
    columns: list[int] = field(default_factory=list)
    column_choices: dict[int, str] = field(default_factory=dict)
    issue: str = ""

    @property
    def label(self):
        return f"No.{self.number or '?'}｜{self.parent + ' / ' if self.parent else ''}{self.title}"


def definitions(frame: pd.DataFrame) -> list[Question]:
    headers = [text(v) for v in frame.iloc[0]] if len(frame) else []
    required = ["No", "質問内容", "回答方式", "選択肢"]
    if not all(h in headers for h in required):
        raise SurveyError("設問定義のヘッダー（No・質問内容・回答方式・選択肢）が見つかりません。")
    def get(row, key):
        positions = [i for i, h in enumerate(headers) if h == key]
        return text(row.iloc[positions[-1]]) if positions else ""
    result, parent = [], ""
    for _, row in frame.iloc[1:].iterrows():
        title, raw_kind = get(row, "質問内容"), get(row, "回答方式")
        if not title or not raw_kind:
            continue
        # The supplied definition has furigana in header names.
        sub = next((text(row.iloc[i]) for i, h in enumerate(headers) if h.startswith("小問番号")), "")
        if "見出し" in raw_kind:
            parent = title if sub == "0" else ""
            continue
        if sub == "0":
            parent = title
            continue
        if not sub:
            parent = ""
        kind = "multi" if "複数選択" in raw_kind else "single" if "選択" in raw_kind or "リスト" in raw_kind else "numeric" if "整数" in raw_kind else "text"
        choices = list(dict.fromkeys(v.strip() for v in get(row, "選択肢").splitlines() if v.strip()))
        result.append(Question(get(row, "No"), title, kind, choices, get(row, "ジャンル"), parent))
    if not result:
        raise SurveyError("集計可能な設問定義がありません。")
    return result


def header_groups(frame: pd.DataFrame):
    if len(frame) < 2:
        raise SurveyError("回答ファイルには2行のヘッダーが必要です。")
    groups = {}
    for col in range(frame.shape[1]):
        header = text(frame.iloc[0, col])
        match = re.match(r"No\.(\d+)\s*[：:]\s*(.*)", header, re.S)
        if not match:
            if header:
                raise SurveyError("回答ファイルの1行目は No.番号：設問文 の形式にしてください。")
            continue
        number, title = match.groups()
        group = groups.setdefault(number, {"title": title, "columns": [], "choices": {}})
        if norm(group["title"]) != norm(title):
            raise SurveyError("同じ設問番号に異なる設問文があります。ヘッダーを確認してください。")
        group["columns"].append(col)
        choice = re.sub(r"^\d+\s*[-－]\s*", "", text(frame.iloc[1, col]))
        if choice:
            group["choices"][col] = choice
    if not groups:
        raise SurveyError("回答ファイルの設問ヘッダーが見つかりません。")
    return groups


def map_questions(questions: list[Question], frame: pd.DataFrame, overrides: dict | None = None):
    groups = header_groups(frame)
    overrides = overrides or {}
    sequence = len(groups) == len(questions)
    used = set()
    for position, q in enumerate(questions):
        q.columns, q.column_choices, q.number, q.issue = [], {}, "", ""
        number = overrides.get(q.id)
        if number is None:
            if sequence:
                candidate = list(groups)[position]
                number = candidate if similar(q.title, groups[candidate]["title"]) else ""
            else:
                candidates = [n for n, g in groups.items() if similar(q.title, g["title"])]
                number = candidates[0] if len(candidates) == 1 else ""
        if number not in groups or number in used:
            q.issue = "設問の対応付けを確認してください。"
            continue
        group = groups[number]
        q.number, q.columns = number, list(group["columns"])
        used.add(number)
        if q.kind in ("single", "multi") and not q.choices:
            q.issue = "設問定義に選択肢がありません。"
            continue
        if q.kind == "multi":
            if len(group["choices"]) != len(q.columns):
                q.issue = "複数選択列の2行目に選択肢が必要です。"
            for col in q.columns:
                label = group["choices"].get(col, "")
                candidates = [c for c in q.choices if norm(c).rstrip("*") == norm(label).rstrip("*")]
                if not candidates:
                    candidates = [c for c in q.choices if similar(c, label)]
                if len(candidates) == 1:
                    q.column_choices[col] = candidates[0]
                elif label:
                    q.column_choices[col] = label
                    q.issue = "定義と選択肢が一致しません。選択肢の対応付けを確認してください。"
            if len(set(q.column_choices.values())) != len(q.columns):
                q.issue = "選択肢の対応付けに重複または未設定があります。"
        elif len(q.columns) != 1:
            q.issue = "単一回答の設問に複数列が対応しています。"
    return groups


@dataclass
class Dataset:
    questions: list[Question]
    values: dict[str, pd.Series | pd.DataFrame]
    index: pd.Index
    fingerprint: str
    problems: dict[str, list[str]]

    def question(self, qid):
        found = next((q for q in self.questions if q.id == qid), None)
        if found is None:
            raise SurveyError("指定された設問が現在のデータにありません。")
        return found

    def available(self, kinds=("single", "multi", "numeric")):
        return [q for q in self.questions if q.kind in kinds and q.id in self.values and q.id not in self.problems and not q.issue]


def fingerprint(questions: list[Question], frame: pd.DataFrame):
    payload = {"questions": [(q.id, q.title, q.parent, q.kind, q.choices) for q in questions], "headers": [[text(v) for v in frame.iloc[r]] for r in range(2)]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_dataset(questions: list[Question], frame: pd.DataFrame, aliases: dict | None = None) -> Dataset:
    answers = frame.iloc[2:].copy().reset_index(drop=True)
    # Every response row counts as a respondent, including wholly blank rows.
    values, problems = {}, {}
    aliases = aliases or {}
    for q in questions:
        if q.issue or not q.columns:
            continue
        if q.kind == "multi":
            value = pd.DataFrame(False, index=answers.index, columns=q.choices, dtype=bool)
            for col, choice in q.column_choices.items():
                if choice not in value.columns:
                    problems.setdefault(q.id, []).append(choice)
                    continue
                raw = answers[col].map(text)
                # Choice names come from the second header row; answer cells
                # normally contain a circle, not the name of that choice.
                # Keep legacy choice-name answers and free text in 'other' columns.
                selection_by_value = {
                    v: bool(v) and (
                        norm(v) in {"〇", "○"}
                        or similar(v, choice)
                        or "その他" in choice
                        or aliases.get(q.id, {}).get(v) == choice
                    )
                    for v in raw.unique()
                }
                selected = raw.map(selection_by_value).astype(bool)
                unknown = [v for v, is_selected in selection_by_value.items() if v and not is_selected]
                if unknown:
                    problems.setdefault(q.id, []).extend(unknown)
                value[choice] |= selected
            values[q.id] = value
        elif q.kind == "numeric":
            raw = answers[q.columns[0]].map(text)
            value = pd.to_numeric(raw.replace("", None), errors="coerce")
            invalid = raw.ne("") & (value.isna() | (value < 0) | (value % 1 != 0) | ~value.fillna(0).map(lambda n: float('-inf') < n < float('inf')))
            if invalid.any():
                problems[q.id] = ["数値回答に非数値・負数・整数以外の値があります。元データを確認してください。"]
            values[q.id] = value
        else:
            raw = answers[q.columns[0]].map(text)
            if q.kind == "single":
                choice_by_norm = {norm(c): c for c in q.choices}
                value = raw.map(lambda v: aliases.get(q.id, {}).get(v, choice_by_norm.get(norm(v), v)))
                unknown = [v for v in value.unique() if v and v not in q.choices]
                if unknown:
                    problems[q.id] = unknown
                values[q.id] = value
            else:
                values[q.id] = raw
    return Dataset(questions, values, answers.index, fingerprint(questions, frame), problems)


MISSING = "（無回答）"


def membership(ds: Dataset, qid: str, include_missing=False) -> pd.DataFrame:
    q, values = ds.question(qid), ds.values[qid]
    if qid in ds.problems or q.issue:
        raise SurveyError("対応付け・回答値に未確認の項目があります。読み込み画面で確認してください。")
    if q.kind == "single":
        result = pd.DataFrame({c: values.eq(c) for c in q.choices}, index=ds.index)
    elif q.kind == "multi":
        result = values.copy()
    else:
        raise SurveyError("この集計は選択式設問に対応しています。")
    if include_missing:
        result[MISSING] = ~result.any(axis=1)
    return result


def validate_merges(question: Question, merges: dict):
    if not isinstance(merges, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in merges.items()):
        raise SurveyError("選択肢の集約設定の形式が不正です。")
    if merges and question.kind not in ("single", "multi"):
        raise SurveyError("選択肢の集約は選択式設問で使用できます。")
    if any(choice not in question.choices for choice in merges):
        raise SurveyError("集約元の選択肢が現在の設問にありません。")
    if any(not name.strip() for name in merges.values()):
        raise SurveyError("集約後の区分名を空欄にすることはできません。")
    if any(norm(name) == norm(MISSING) for name in merges.values()):
        raise SurveyError("無回答の区分名は集約先に指定できません。")


def merged_choices(question: Question, merges: dict) -> list[str]:
    validate_merges(question, merges)
    return list(dict.fromkeys(merges.get(choice, choice).strip() for choice in question.choices))


def merged_membership(ds: Dataset, qid: str, merges: dict, include_missing=False) -> pd.DataFrame:
    question = ds.question(qid)
    validate_merges(question, merges)
    members = membership(ds, qid)
    result = pd.DataFrame(index=members.index)
    for choice in members:
        name = merges.get(choice, choice).strip()
        if name in result:
            # OR, not addition: each respondent counts once in a merged category.
            result[name] |= members[choice]
        else:
            result[name] = members[choice].copy()
    if include_missing:
        result[MISSING] = ~members.any(axis=1)
    return result


def merge_note(question: Question, merges: dict, role: str) -> str:
    if not merges:
        return ""
    groups = {}
    for choice in question.choices:
        groups.setdefault(merges.get(choice, choice).strip(), []).append(choice)
    changes = [f"{name} ← {' / '.join(choices)}" for name, choices in groups.items()
               if len(choices) > 1 or choices[0] != name]
    return f" {role}の集約：{'；'.join(changes)}。" if changes else ""


def filter_index(ds: Dataset, filters: list[dict]) -> pd.Index:
    if not isinstance(filters, list):
        raise SurveyError("絞り込み条件の形式が不正です。")
    mask = pd.Series(True, index=ds.index)
    for f in filters:
        if not isinstance(f, dict) or not isinstance(f.get("question"), str):
            raise SurveyError("絞り込み条件の形式が不正です。")
        q = ds.question(f["question"])
        if q.id not in {question.id for question in ds.available()}:
            raise SurveyError("絞り込みに指定された設問は現在のデータでは使用できません。")
        if q.kind == "numeric":
            lo, hi = f.get("min"), f.get("max")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [lo, hi]) or lo > hi:
                raise SurveyError("数値の絞り込み範囲が不正です。")
            mask &= ds.values[q.id].between(f["min"], f["max"]).fillna(False)
        else:
            members = membership(ds, q.id, True)
            choices = f.get("choices", [])
            if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                raise SurveyError("絞り込みの選択肢の形式が不正です。")
            if any(c not in members for c in choices):
                raise SurveyError("絞り込みの選択肢が現在のデータと一致しません。")
            mask &= members[choices].any(axis=1) if choices else False
    return ds.index[mask]


@dataclass
class Aggregate:
    counts: pd.DataFrame
    percents: pd.DataFrame
    denominators: pd.DataFrame
    note: str
    filtered_n: int


def aggregate(ds: Dataset, qid: str, group_id: str | None = None, filters: list[dict] | None = None, denominator="valid", include_missing=False, percent_mode="row", *, question_merges=None, group_merges=None) -> Aggregate:
    question_merges = {} if question_merges is None else question_merges
    group_merges = {} if group_merges is None else group_merges
    if not isinstance(group_merges, dict):
        raise SurveyError("属性側の集約設定の形式が不正です。")
    if group_id is None and group_merges:
        raise SurveyError("属性側の集約には属性の設問を指定してください。")
    idx = filter_index(ds, filters or [])
    target = merged_membership(ds, qid, question_merges, include_missing).loc[idx]
    group = merged_membership(ds, group_id, group_merges, include_missing).loc[idx] if group_id else pd.DataFrame({"全体": True}, index=idx)
    counts = pd.DataFrame(group.to_numpy(dtype=int).T @ target.to_numpy(dtype=int), index=group.columns, columns=target.columns)
    valid_target, valid_group = target.any(axis=1), group.any(axis=1)
    den = pd.DataFrame(0.0, index=counts.index, columns=counts.columns)
    if denominator not in ("valid", "all") or percent_mode not in ("row", "column", "overall"):
        raise SurveyError("割合の設定が不正です。")
    if percent_mode == "row":
        for g in group:
            den.loc[g, :] = int((group[g] & (valid_target if denominator == "valid" else True)).sum())
        basis = "各グループ内の有効回答者数" if denominator == "valid" else "各グループの対象者数"
    elif percent_mode == "column":
        for c in target:
            den.loc[:, c] = int((target[c] & (valid_group if denominator == "valid" else True)).sum())
        basis = "各回答選択肢内のグループ有効回答者数" if denominator == "valid" else "各回答選択肢の対象者数"
    else:
        den.loc[:, :] = int((valid_target & valid_group).sum()) if denominator == "valid" else len(idx)
        basis = "両設問に回答した人数" if denominator == "valid" else "絞り込み後の対象者数"
    percents = counts.div(den.where(den.ne(0))).mul(100)
    multi = ds.question(qid).kind == "multi" or (group_id and ds.question(group_id).kind == "multi")
    note = f"対象者数 N={len(idx)}。割合の分母：{basis}。"
    if include_missing:
        note += "無回答を1カテゴリーとして含む。"
    else:
        note += "無回答はカテゴリーから除外。"
    if multi:
        note += "複数選択は回答者単位で計数し、合計が100％を超える場合があります。全列空欄は無回答扱い。"
    if group_id and ds.question(group_id).kind == "multi" and (group.sum(axis=1) > 1).any():
        note += "グループ間に回答者の重複があります。"
    note += merge_note(ds.question(qid), question_merges, "回答側")
    if group_id:
        note += merge_note(ds.question(group_id), group_merges, "属性側")
    return Aggregate(counts, percents, den, note, len(idx))
