# app.py
import streamlit as st
import pandas as pd
import csv, io, json, re
from collections import Counter

# ---------- Page setup ----------
st.set_page_config(
    page_title="CSV Cleaner + Rubric Validator", page_icon="🧹", layout="wide"
)

# # Shows a clickable link to the page
# st.page_link("pages/analysis.py", label="➡️ Go to Analysis")
st.title("🧹 CSV Cleaner + Rubric Validator")

if st.button("Take me to Create JSON Page"):
    # Jumps straight to the other page
    st.switch_page("pages/create_json.py")

# if st.button("Take me to Fix JSON Page"):
#     # Jumps straight to the other page
#     st.switch_page("pages/fix_json.py")


# ---------- Helpers ----------
def detect_encoding(raw_bytes, fallbacks=("utf-8", "utf-16", "latin-1", "cp1252")):
    for enc in fallbacks:
        try:
            text = raw_bytes.decode(enc)
            return enc, text
        except UnicodeDecodeError:
            continue
    return "utf-8 (with replacement)", raw_bytes.decode("utf-8", errors="replace")


def sniff_sep(text: str) -> str:
    sample = text[:10000]
    try:
        # csv.Sniffer expects a *string* of possible delimiters, not a list.
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def is_nullish_colname(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return True
    if isinstance(x, str):
        if x.strip() == "" or x.lower().startswith("unnamed:"):
            return True
    return False


def nullish_series(s: pd.Series) -> pd.Series:
    # Treat NaN as null. For object-like columns, also treat empty/whitespace-only strings as null.
    base = s.isna()
    # cover object & pandas 'string' dtype by stringifying
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        empties = s.astype(str).str.strip().eq("")
        return base | empties
    return base


def null_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    data = []
    n_rows = len(df)
    for c in df.columns:
        m = nullish_series(df[c])
        n = int(m.sum())
        data.append(
            {
                "column": str(c),
                "nulls": n,
                "rows": n_rows,
                "null_%": (n / n_rows * 100.0) if n_rows else 0.0,
            }
        )
    out = pd.DataFrame(data)
    return out.sort_values(["nulls", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


# ---------- Rubric validation (single, de-duplicated set) ----------
ALLOWED_WEIGHTS = {"Primary Objective(s)", "Not Primary Objective"}
REQUIRED_CRITERION_FIELDS = [
    "description",
    "sources",
    "justification",
    "weight",
    "human_rating",
    "gemini_as_autorater_rating",
    "gpt_as_autorater_rating",
    "criterion_type",
    "dependent_criteria",
]


def _criterion_key(item):
    """Return (key, payload) if item is a single-key dict, else (None, None)."""
    if isinstance(item, dict) and len(item) == 1:
        k, v = next(iter(item.items()))
        return k, v
    return None, None


def find_smart_quotes(obj, path="$"):
    smart_quotes = {"“", "”", "‘", "’"}
    errs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(c in str(k) for c in smart_quotes):
                errs.append(f"{path}.{k} (key) contains smart quotes.")
            errs.extend(find_smart_quotes(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errs.extend(find_smart_quotes(v, f"{path}[{i}]"))
    # 🔁 Remove string check entirely
    return errs


def validate_single_rubric_object(item, idx_in_list, all_names):
    """
    Validate a single rubric object against schema + cross-refs.
    all_names = set of all criterion keys in the same rubric array.
    """
    errs = []

    if not isinstance(item, dict):
        errs.append(f"[{idx_in_list}] must be an object with a single criterion key.")
        return errs

    if len(item.keys()) != 1:
        errs.append(
            f"[{idx_in_list}] must have exactly one key (e.g., 'criterion 1'). Found {len(item.keys())}."
        )
        return errs

    criterion_name, payload = next(iter(item.items()))
    # Check all keys and values for smart quotes
    errs.extend([f"[{idx_in_list}] {e}" for e in find_smart_quotes(item)])

    if not isinstance(criterion_name, str) or not criterion_name.strip():
        errs.append(f"[{idx_in_list}] criterion key must be a non-empty string.")

    # Optional pattern check
    if not re.fullmatch(r"criterion \d+", criterion_name):
        errs.append(
            f"[{idx_in_list}] Invalid key '{criterion_name}': must match exact format \"criterion <number>\" with lowercase keyword and space separator."
        )

    if not isinstance(payload, dict):
        errs.append(f"[{idx_in_list}] value for '{criterion_name}' must be an object.")
        return errs

    # Required fields present?
    for field in REQUIRED_CRITERION_FIELDS:
        if field not in payload:
            errs.append(
                f"[{idx_in_list}] missing required field '{field}' in '{criterion_name}'."
            )

    # Types + constraints
    if "description" in payload and not isinstance(payload["description"], str):
        errs.append(f"[{idx_in_list}] 'description' must be string.")
    if "sources" in payload:
        sources = payload["sources"]
        if not isinstance(sources, list):
            errs.append(f"[{idx_in_list}] 'sources' must be a list of strings.")
        else:
            if not all(isinstance(s, str) and s.strip() for s in sources):
                errs.append(
                    f"[{idx_in_list}] 'sources' must contain only non-empty strings."
                )

    if "justification" in payload and not isinstance(payload["justification"], str):
        errs.append(f"[{idx_in_list}] 'justification' must be string.")
    if "weight" in payload and payload["weight"] not in ALLOWED_WEIGHTS:
        errs.append(
            f"[{idx_in_list}] 'weight' must be either 'Primary Objective(s)' or 'Not Primary Objective'. Found: {payload['weight']!r}"
        )
    for b in ["human_rating", "gemini_as_autorater_rating", "gpt_as_autorater_rating"]:
        if b in payload and not isinstance(payload[b], bool):
            errs.append(f"[{idx_in_list}] '{b}' must be boolean.")
    if "criterion_type" in payload:
        ct = payload["criterion_type"]
        if not isinstance(ct, list) or not all(isinstance(x, str) for x in ct):
            errs.append(f"[{idx_in_list}] 'criterion_type' must be a list of strings.")

    # dependent_criteria cross-references
    if "dependent_criteria" in payload:
        dc = payload["dependent_criteria"]
        if not isinstance(dc, list):
            errs.append(
                f"[{idx_in_list}] 'dependent_criteria' must be a list of strings."
            )
        else:
            seen = set()
            for j, dep in enumerate(dc):
                if not isinstance(dep, str):
                    errs.append(
                        f"[{idx_in_list}] 'dependent_criteria[{j}]' must be a string."
                    )
                    continue
                if dep == criterion_name:
                    errs.append(
                        f"[{idx_in_list}] '{criterion_name}' cannot depend on itself."
                    )
                if dep in seen:
                    errs.append(f"[{idx_in_list}] duplicate dependency '{dep}'.")
                seen.add(dep)
                if dep not in all_names:
                    errs.append(
                        f"[{idx_in_list}] dependency '{dep}' does not match any criterion key in this rubric."
                    )
    return errs


def validate_rubric_json(value):
    """
    Returns (ok: bool, error_list: list[str]).
    Accepts strings that parse to: list[ { "criterion n": { required fields... } }, ... ]
    Also validates that 'dependent_criteria' entries reference existing criterion keys.
    """
    errs = []
    try:
        obj = value if isinstance(value, (list, dict)) else json.loads(value)
    except Exception as e:
        return False, [f"JSON parse error: {e}"]

    if not isinstance(obj, list):
        return False, ["Top-level must be a JSON array."]

    # Build the set of all criterion names to validate cross-refs
    all_names = set()
    for i, item in enumerate(obj):
        k, _ = _criterion_key(item)
        if k is None:
            # Defer exact error to per-item validator
            continue
        all_names.add(k)

    # Validate each item with access to all_names
    for i, item in enumerate(obj):
        errs.extend(validate_single_rubric_object(item, i, all_names))

    return (len(errs) == 0), errs


def normalized_json(value):
    """Pretty-print JSON string with stable formatting (2-space indent)."""
    obj = value if isinstance(value, (list, dict)) else json.loads(value)
    return json.dumps(obj, ensure_ascii=False, indent=2)


criterion_pattern = re.compile(r"^criterion\s*(\d+)$", re.I)


def check(x, n):
    n = int(n)
    if not x or (isinstance(x, str) and not x.strip()):
        return {"missing": list(range(1, n + 1)), "duplicated": [], "misordered": []}
    try:
        arr = x if isinstance(x, list) else json.loads(x)
    except Exception:
        return "invalid_json"

    nums = []
    for d in arr:
        if isinstance(d, dict):
            for k in d.keys():  # only top-level keys
                m = criterion_pattern.match(k)
                if m:
                    nums.append(int(m.group(1)))
                    break

    w = [i for i in nums if 1 <= i <= n]  # only count 1..n
    if w == list(range(1, n + 1)) and len(w) == len(nums):
        return True

    c = Counter(w)
    return {
        "missing": [i for i in range(1, n + 1) if c[i] == 0],
        "duplicated": sorted(i for i, v in c.items() if v > 1),
        "misordered": [(pos, val) for pos, val in enumerate(w, 1) if val != pos],
    }


def init_state():
    for k, v in {
        "df": None,
        "encoding": None,
        "sep": ",",
        "view_null_column": None,
        "rubric_col": None,
        "edit_row_for_rubric": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ---------- Upload UI ----------
with st.sidebar:
    st.header("⚙️ Import options")
    sep_choice = st.radio(
        "Delimiter",
        ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab (\\t)", "Pipe (|)"],
        index=0,
        horizontal=False,
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    enc, text = detect_encoding(raw_bytes)
    st.session_state["encoding"] = enc
    sep = (
        sniff_sep(text)
        if sep_choice == "Auto-detect"
        else {
            "Comma (,)": ",",
            "Semicolon (;)": ";",
            "Tab (\\t)": "\t",
            "Pipe (|)": "|",
        }[sep_choice]
    )
    st.session_state["sep"] = sep

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as e:
        st.error(f"Couldn't read the file. Details: {e}")
        st.stop()
    st.session_state["df"] = df.copy()
    st.session_state["view_null_column"] = None

df = st.session_state["df"]

if df is None:
    st.info("⬅️ Upload a CSV from the sidebar to begin.")
    st.stop()

st.caption(
    f"Loaded **{len(df)}** rows × **{len(df.columns)}** columns • Encoding: **{st.session_state['encoding']}** • Delimiter: **{repr(st.session_state['sep'])}**"
)

# ---------- 1) Column-name validation & fixing ----------
st.subheader("1) Column name check & fix")
bad_pos = [i for i, c in enumerate(df.columns) if is_nullish_colname(c)]
if bad_pos:
    st.error(f"Found {len(bad_pos)} blank/NaN/`Unnamed:` column name(s).")
    rename_data = []
    for i in range(len(df.columns)):
        curr = df.columns[i]
        rename_data.append(
            {
                "position (0-based)": i,
                "current_name": "" if is_nullish_colname(curr) else str(curr),
                "new_name": f"col_{i}" if is_nullish_colname(curr) else str(curr),
            }
        )
    rename_df = pd.DataFrame(rename_data)
    edited = st.data_editor(
        rename_df,
        key="rename_editor",
        use_container_width=True,
        column_config={
            "position (0-based)": st.column_config.NumberColumn(disabled=True),
            "current_name": st.column_config.TextColumn(disabled=True),
            "new_name": st.column_config.TextColumn(help="Edit to your preferred name"),
        },
    )
    if st.button("Apply column renames"):
        new_cols = list(df.columns)
        for _, row in edited.iterrows():
            idx = int(row["position (0-based)"])
            new_cols[idx] = str(row["new_name"]).strip() or f"col_{idx}"
        df.columns = new_cols
        st.session_state["df"] = df
        st.success("Column names updated.")
else:
    st.success("No null/blank/`Unnamed:` column names detected.")

st.divider()

# ---------- 2) Null counts per column with clickable drilldown & inline fixing ----------
st.subheader("2) Null value scan & inline fixing")

# Compute counts but do NOT render the first (summary) table
counts = null_counts_df(df)

st.write(
    "**Click a count below to open an editable view of rows with null/blank values in that column.**"
)

with st.container(border=True):
    for _, row in counts.iterrows():
        col_name = row["column"]
        n_null = int(row["nulls"])
        left, mid, right = st.columns([3, 1, 8])
        with left:
            st.write(f"**{col_name}**")
        with mid:
            clicked = st.button(str(n_null), key=f"btn_{col_name}")
        with right:
            st.write(f"({row['null_%']:.1f}% of {row['rows']} rows)")
        if clicked and n_null > 0:
            st.session_state["view_null_column"] = col_name

# Editor for selected column's null rows
target_col = st.session_state["view_null_column"]
if target_col:
    st.markdown(f"**Editing rows with null/blank values in:** `{target_col}`")
    mask = nullish_series(df[target_col])
    sub = df.loc[mask].copy()
    if sub.empty:
        st.info("No null/blank rows remaining for this column.")
    else:
        edited_sub = st.data_editor(
            sub,
            key=f"edit_nulls_{target_col}",
            use_container_width=True,
            num_rows="dynamic",
            height=350,
        )
        if st.button("Apply changes to main data", key=f"apply_{target_col}"):
            df.update(edited_sub)  # aligns on index/columns
            st.session_state["df"] = df
            st.success("Changes applied.")
            # Recompute counts (not strictly necessary; the page will rerun)
            counts = null_counts_df(df)

st.divider()

# ---------- 3) Rubric JSON validation by column ----------
st.subheader("3) Rubric JSON validation (column-wide)")

# Build options & pick a sensible default (first col containing 'rubric')
options = list(df.columns)
likely_cols = [c for c in options if isinstance(c, str) and "rubric" in c.lower()]
id_col = [c for c in options if isinstance(c, str) and "id" in c.lower()]
try:
    default_id = int(options.index(id_col[0])) if id_col else 0
except ValueError:
    default_id = 0
prompt_id_col = st.selectbox(
    "Select the column that contains the Prompt ID",
    options=options,
    index=default_id if len(options) else 0,
    help="This value will be used to label each row in the validation output.",
)

try:
    default_idx = int(options.index(likely_cols[0])) if likely_cols else 0
except ValueError:
    default_idx = 0

rubric_col = st.selectbox(
    "Select the column that contains the rubric JSON",
    options=options,
    index=default_idx if len(options) else 0,
)

normalize_checkbox = st.checkbox(
    "Write back pretty-printed JSON to this column when valid",
    value=True,
    help="When a row is valid (either originally or after auto-fix), write normalized JSON with 2-space indent.",
)

auto_fix = st.checkbox(
    "Try to auto-fix invalid JSON cells",
    value=True,
    help="Attempts: normalize quotes, remove trailing commas, convert True/False/None/NaN, parse Python-like literals, and coerce to expected rubric structure.",
)


# --- Auto-fix helpers (with change notes) ---
def _subn_note(pattern, repl, text, label_fmt):
    new_text, n = re.subn(pattern, repl, text)
    note = label_fmt(n) if n else None
    return new_text, n, note


def _clean_json_text(s: str):
    """
    Returns cleaned_text, changes(list[str]).
    Adds concise 'old: X → new Y' notes with counts.
    """
    if not isinstance(s, str):
        s = str(s)
    t = s.strip()
    notes = []

    # Smart quotes → straight
    mapping = [
        ("“", '"'),
        ("”", '"'),
        ("„", '"'),
        ("‟", '"'),
        ("’", "'"),
        ("‘", "'"),
    ]
    for old, new in mapping:
        if old in t:
            count = t.count(old)
            t = t.replace(old, new)
            notes.append(f"old: {old} → new {new} ({count})")

    # Remove trailing commas before } or ]
    t, n_tc, note = _subn_note(
        r",\s*([}\]])", r"\1", t, lambda n: f"removed trailing commas ({n})"
    )
    if note:
        notes.append(note)

    # Python/NumPy-ish tokens → JSON
    convs = [
        (r"\bTrue\b", "true", "old: True → new true"),
        (r"\bFalse\b", "false", "old: False → new false"),
        (r"\bNone\b", "null", "old: None → new null"),
        (r"\bNaN\b", "null", "old: NaN → new null"),
    ]
    for pat, rep, label in convs:
        t2, n, _ = _subn_note(pat, rep, t, lambda n: f"{label} ({n})")
        if n:
            notes.append(f"{label} ({n})")
            t = t2

    # Backticks → double quotes
    if "`" in t:
        cnt = t.count("`")
        t = t.replace("`", '"')
        notes.append(f'old: ` → new " ({cnt})')

    return t, notes


def _coerce_to_rubric_list(obj):
    """
    Accepts list[ {k: {...}} ] or dict {k: {...}} and coerces to list of single-key dicts.
    """
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and len(item) == 1:
                out.append(item)
            elif isinstance(item, dict) and len(item) > 1:
                out.extend([{k: v} for k, v in item.items()])
            else:
                raise ValueError("List contains non-dict items; cannot coerce.")
        return out
    if isinstance(obj, dict):
        return [{k: v} for k, v in obj.items()]
    raise ValueError("Top-level is neither list nor dict; cannot coerce.")


def _postprocess_payload_shapes(lst):
    """
    Fix common schema-shape issues inside payloads:
    - dependent_criteria: string -> [string]
    - criterion_type: string -> [string]
    """
    fixed = []
    for item in lst:
        if not (isinstance(item, dict) and len(item) == 1):
            fixed.append(item)
            continue
        k, v = next(iter(item.items()))
        if isinstance(v, dict):
            if "dependent_criteria" in v and isinstance(v["dependent_criteria"], str):
                v["dependent_criteria"] = [v["dependent_criteria"]]
            if "criterion_type" in v and isinstance(v["criterion_type"], str):
                v["criterion_type"] = [v["criterion_type"]]
        fixed.append({k: v})
    return fixed


def _try_parse_and_fix(cell_value):
    """
    Returns (obj_or_none, fix_notes: list[str], error_msg_or_none)
    """
    notes = []

    # 1) direct JSON
    try:
        return json.loads(cell_value), notes, None
    except Exception as e_json:
        err = str(e_json)

    # 2) cleaned JSON text (with detailed notes)
    cleaned, clean_notes = _clean_json_text(cell_value)
    notes.extend(clean_notes)
    try:
        return json.loads(cleaned), notes, None
    except Exception as e_clean:
        err = f"{err} | {e_clean}"

    # 3) Python literal -> JSON-ish
    try:
        import ast

        py_obj = ast.literal_eval(cleaned)
        # Flag the common single-quote to double-quote conversion
        if "'" in str(cell_value) and '"' in json.dumps(py_obj):
            notes.append("converted single-quoted strings to JSON (old: ' → new \")")
        notes.append("parsed Python-like literal (via ast.literal_eval)")
        return py_obj, notes, None
    except Exception as e_ast:
        err = f"{err} | literal_eval: {e_ast}"

    return None, notes, err


def _normalize_and_validate(obj):
    """
    Coerce obj to rubric list form, fix inner shapes, then validate.
    Returns (ok, normalized_json_str, errors_list, maybe_obj)
    """
    try:
        coerced = _coerce_to_rubric_list(obj)
        coerced = _postprocess_payload_shapes(coerced)
        ok, errs = validate_rubric_json(coerced)
        if ok:
            return True, json.dumps(coerced, ensure_ascii=False, indent=2), [], coerced
        else:
            return False, None, errs, coerced
    except Exception as e:
        return False, None, [f"Coercion error: {e}"], None


def _shorten_for_note(s: str, max_len: int = 200) -> str:
    """Single-line preview for notes; trims long blobs."""
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _error_fix_note(before: str, after: str) -> str:
    """Format: Error: "<before>" → Fix: "<after>""" ""
    b = _shorten_for_note(before)
    a = _shorten_for_note(after)
    # keep quotes readable in notes
    b = b.replace('"', '\\"')
    a = a.replace('"', '\\"')
    return f'Error: "{b}" \u2192 Fix: "{a}"'


import json, re
import pandas as pd

# put near your other helpers
_crit_pat = re.compile(r"^criterion\s*\d+$", re.I)


def count_criteria(cell):
    """Return the number of top-level criterion objects in a rubric JSON cell."""
    if pd.isna(cell) or (isinstance(cell, str) and not cell.strip()):
        return 0
    try:
        obj = cell if isinstance(cell, (list, dict)) else json.loads(cell)
    except Exception:
        return pd.NA  # invalid JSON → missing

    # coerce dict → list of single-key dicts
    if isinstance(obj, dict):
        obj = [{k: v} for k, v in obj.items()]
    if not isinstance(obj, list):
        return pd.NA

    cnt = 0
    for item in obj:
        if isinstance(item, dict) and len(item) == 1:
            k = next(iter(item.keys()))
            if isinstance(k, str) and _crit_pat.fullmatch(k):
                cnt += 1
    return cnt


# compute/update the column (use your selected rubric_col)
df["n_criteria"] = df[rubric_col].apply(count_criteria).astype("Int64")


def _safe_int(n):
    try:
        return 0 if pd.isna(n) else int(n)
    except Exception:
        return 0


df["order_check"] = df.apply(
    lambda r: check(r[rubric_col], _safe_int(r["n_criteria"])), axis=1
)
asdict = lambda v: (
    {"missing": [], "duplicated": [], "misordered": []}
    if v is True
    else (v if isinstance(v, dict) else None)
)
fmt = lambda xs: [f"criterion {i}" for i in xs] if isinstance(xs, list) else pd.NA

df["missing_list"] = df["order_check"].map(
    lambda v: fmt(asdict(v)["missing"]) if asdict(v) is not None else pd.NA
)
df["duplicated_list"] = df["order_check"].map(
    lambda v: fmt(asdict(v)["duplicated"]) if asdict(v) is not None else pd.NA
)
df["misordered_list"] = df["order_check"].map(
    lambda v: asdict(v)["misordered"] if asdict(v) is not None else pd.NA
)

if st.button("Validate all rows"):
    results = []
    normalized_values = {}
    fixes_applied = 0

    for idx, cell in df[rubric_col].items():
        prompt_id = df.at[idx, prompt_id_col] if prompt_id_col in df.columns else idx
        n_crit = df.at[idx, "n_criteria"]
        order_info = df.at[idx, "order_check"]

        if pd.isna(cell) or (isinstance(cell, str) and cell.strip() == ""):
            results.append(
                {
                    "Prompt ID": prompt_id,
                    "n_criteria": n_crit,
                    "order_check": order_info,
                    "valid": False,
                    "fixed": False,
                    "error_count": 1,
                    "errors": "Cell is empty / missing rubric JSON.",
                    "notes": "",
                }
            )
            continue

        ok_direct, errs_direct = validate_rubric_json(cell)
        if ok_direct:
            if normalize_checkbox:
                normalized_values[idx] = normalized_json(cell)
            results.append(
                {
                    "Prompt ID": prompt_id,
                    "n_criteria": n_crit,
                    "order_check": order_info,
                    "valid": True,
                    "fixed": False,
                    "error_count": 0,
                    "errors": "",
                    "notes": "already valid",
                }
            )
            continue

        if auto_fix:
            parsed, notes, parse_err = _try_parse_and_fix(cell)
            if parsed is not None:
                ok_norm, norm_text, errs_after, _ = _normalize_and_validate(parsed)
                if ok_norm:
                    fixes_applied += 1
                    explicit_note = _error_fix_note(cell, norm_text)
                    if normalize_checkbox:
                        normalized_values[idx] = norm_text
                    combined_notes = [*notes] if notes else []
                    combined_notes.append(explicit_note)
                    results.append(
                        {
                            "Prompt ID": prompt_id,
                            "n_criteria": n_crit,
                            "order_check": order_info,
                            "valid": True,
                            "fixed": True,
                            "error_count": 0,
                            "errors": "",
                            "notes": "; ".join(combined_notes),
                        }
                    )
                else:
                    fail_hint = _shorten_for_note("Normalization/validation failed")
                    combined_notes = [*notes] if notes else []
                    combined_notes.append(fail_hint)
                    results.append(
                        {
                            "Prompt ID": prompt_id,
                            "n_criteria": n_crit,
                            "order_check": order_info,
                            "valid": False,
                            "fixed": True,
                            "error_count": len(errs_after),
                            "errors": " | ".join(errs_after),
                            "notes": "; ".join(combined_notes),
                        }
                    )
            else:
                results.append(
                    {
                        "Prompt ID": prompt_id,
                        "n_criteria": n_crit,
                        "order_check": order_info,
                        "valid": False,
                        "fixed": False,
                        "error_count": 1,
                        "errors": f"Parse error after fixes: {parse_err}",
                        "notes": "; ".join(notes) if notes else "",
                    }
                )
        else:
            results.append(
                {
                    "Prompt ID": prompt_id,
                    "n_criteria": n_crit,
                    "order_check": order_info,
                    "valid": False,
                    "fixed": False,
                    "error_count": len(errs_direct),
                    "errors": " | ".join(errs_direct),
                    "notes": "auto-fix disabled",
                }
            )

    results_df = pd.DataFrame(results).sort_values(
        ["valid", "fixed", "error_count", "Prompt ID"],
        ascending=[True, False, True, True],
    )

    # Summary
    n_rows = len(results_df)
    n_valid = int(results_df["valid"].sum())
    n_invalid = n_rows - n_valid
    st.markdown(
        f"**Validation complete:** {n_valid} / {n_rows} rows valid, {n_invalid} invalid."
    )
    if auto_fix:
        st.caption(f"Auto-fixes applied to {fixes_applied} row(s).")

    # --- Build combined "Criterion Order Check + Errors" table in one expander ---


def _collect_errors(cell):
    # Empty cell
    if pd.isna(cell) or (isinstance(cell, str) and not cell.strip()):
        return False, ["Cell is empty / missing rubric JSON."]
    # Validate against your schema
    try:
        ok, errs = validate_rubric_json(
            cell if isinstance(cell, (list, dict)) else cell
        )
        return ok, errs if not ok else []
    except Exception as e:
        return False, [f"Validation crashed: {e}"]


# run validation (no auto-fix) to get error lists/counts per row
_ok_errs = df[rubric_col].apply(_collect_errors)
error_count = _ok_errs.apply(lambda t: 0 if t[0] else len(t[1]))
errors_joined = _ok_errs.apply(lambda t: "" if t[0] else " | ".join(t[1]))

# build the combined view
combined_view = pd.DataFrame(
    {
        "Prompt ID": df[prompt_id_col],
        "rubric": df[rubric_col].astype(str),
        "n_criteria": df["n_criteria"],
        "error_count": error_count,
        "errors": errors_joined,
        "missing": df["missing_list"],
        "duplicated": df["duplicated_list"],
        "misordered": df["misordered_list"],
    }
)

cols = [
    "Prompt ID",
    "rubric",
    "n_criteria",
    "error_count",
    "errors",
    "missing",
    "duplicated",
    "misordered",
]
st.dataframe(combined_view[cols], use_container_width=True, hide_index=True)


# optional: sort to show problem rows first
combined_view = combined_view.sort_values(
    ["error_count", "Prompt ID"], ascending=[False, True]
)

st.subheader("🔍 Criterion Order Check & Errors")
with st.expander("Show order check & validation errors", expanded=False):
    st.dataframe(
        combined_view[["Prompt ID", "rubric", "n_criteria", "error_count", "errors"]],
        use_container_width=True,
        hide_index=True,
    )

    # Write back normalized JSON (from already-valid or fixed rows)
    if normalize_checkbox and normalized_values:
        if "results_df" in locals() and "normalized_values" in locals():
            df.loc[list(normalized_values.keys()), rubric_col] = pd.Series(
                normalized_values
            )
            st.session_state["df"] = df
            st.success(
                f"Wrote normalized JSON back to '{rubric_col}' for {len(normalized_values)} row(s)."
            )

        # Downloadable error report (CSV)
        csv_buf = io.StringIO()
        results_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download validation report (CSV)",
            data=csv_buf.getvalue(),
            file_name="rubric_validation_report.csv",
            mime="text/csv",
        )
