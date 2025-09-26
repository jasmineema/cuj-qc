# --- Dynamic "Create Rubric JSON" UI ---
import streamlit as st
import json
from copy import deepcopy
from uuid import uuid4

ALLOWED_WEIGHTS = ["Primary objective(s)", "Not primary objective"]
CRITERION_TYPE_OPTIONS = [
    "Reasoning",
    "Extraction (recall)",
    "Style",
    "Compliance",
    "Safety",
    "Information Architecture",
    "Persona & Tone",
    "Multimodal Output",
    "Pedagogy",
]

# one-time state init
if "criteria" not in st.session_state:
    st.session_state.criteria = []  # list of payload dicts, one per criterion


def _empty_payload():
    return {
        "_id": f"crit_{uuid4().hex}",  # stable id for widget keys
        "description": "",
        "sources": ["Prompt"],
        "justification": "",
        "weight": ALLOWED_WEIGHTS[0],
        "human_rating": False,
        "criterion_type": [],
        "dependent_criteria": [],
        "gemini_as_autorater_rating": False,
        "gpt_as_autorater_rating": False,
    }


def _clear_keys_for(prefix: str):
    """Remove any session_state keys that start with the given prefix."""
    for k in list(st.session_state.keys()):
        if k.startswith(prefix):
            del st.session_state[k]


st.subheader("➕ Build Rubric JSON")
help_cols = st.columns([1, 1, 2])
with help_cols[0]:
    if st.button("Add new criterion"):
        st.session_state.criteria.append(_empty_payload())
with help_cols[1]:
    if st.button("Clear all"):
        # wipe widget state too
        for c in st.session_state.criteria:
            _clear_keys_for(c["_id"])
        st.session_state.criteria = []

# Render an editor for each criterion
to_delete = []
for idx, payload in enumerate(st.session_state.criteria, start=1):
    crit_key = f"criterion {idx}"
    crit_id = payload["_id"]  # stable id for keys

    # ---- seed per-row UI state ONCE (no defaults afterward) ----
    ctype_key = f"{crit_id}:ctype"
    if ctype_key not in st.session_state:
        # keep only allowed legacy values
        st.session_state[ctype_key] = [
            t for t in payload.get("criterion_type", []) if t in CRITERION_TYPE_OPTIONS
        ]

    deps_key = f"{crit_id}:deps"
    # dep options depend on current rows; seed below after computing dep_opts

    with st.expander(f"{crit_key}", expanded=True):
        c1, c2 = st.columns([2, 1])

        # Left column — text fields
        with c1:
            payload["description"] = st.text_area(
                f"{crit_key} — description",
                value=payload.get("description", ""),
                key=f"{crit_id}:desc",
                height=90,
            )
            sources_csv = st.text_input(
                f"{crit_key} — sources (comma-separated)",
                value=", ".join(payload.get("sources", [])),
                key=f"{crit_id}:sources",
                help="Example: Prompt, Paper, Docs",
            )
            payload["sources"] = [
                s.strip() for s in sources_csv.split(",") if s.strip()
            ]
            payload["justification"] = st.text_area(
                f"{crit_key} — justification",
                value=payload.get("justification", ""),
                key=f"{crit_id}:just",
                height=90,
            )

        # Right column — structured fields
        with c2:
            payload["weight"] = st.selectbox(
                f"{crit_key} — weight",
                ALLOWED_WEIGHTS,
                index=ALLOWED_WEIGHTS.index(payload.get("weight", ALLOWED_WEIGHTS[0])),
                key=f"{crit_id}:weight",
            )
            payload["human_rating"] = st.checkbox(
                f"{crit_key} — human_rating",
                value=bool(payload.get("human_rating", False)),
                key=f"{crit_id}:hr",
            )
            payload["gemini_as_autorater_rating"] = st.checkbox(
                f"{crit_key} — gemini_as_autorater_rating",
                value=bool(payload.get("gemini_as_autorater_rating", False)),
                key=f"{crit_id}:gem",
            )
            payload["gpt_as_autorater_rating"] = st.checkbox(
                f"{crit_key} — gpt_as_autorater_rating",
                value=bool(payload.get("gpt_as_autorater_rating", False)),
                key=f"{crit_id}:gpt",
            )

            # --- multiselects ---
            # criterion_type: allow multiple values — BIND TO STATE, NO default=
            st.multiselect(
                f"{crit_key} — criterion_type",
                options=CRITERION_TYPE_OPTIONS,
                key=ctype_key,
                help="Choose one or more allowed types.",
            )
            # write back from state to payload
            payload["criterion_type"] = [
                t for t in st.session_state[ctype_key] if t in CRITERION_TYPE_OPTIONS
            ]

            # dependent_criteria options are all other criteria keys (by current index label)
            dep_opts = [
                f"criterion {j}"
                for j in range(1, len(st.session_state.criteria) + 1)
                if j != idx
            ]

            # seed deps state once, then always filter to valid options when rows change
            if deps_key not in st.session_state:
                st.session_state[deps_key] = [
                    d for d in payload.get("dependent_criteria", []) if d in dep_opts
                ]
            else:
                # drop any selections that are no longer valid (e.g., after deletions)
                st.session_state[deps_key] = [
                    d for d in st.session_state[deps_key] if d in dep_opts
                ]

            st.multiselect(
                f"{crit_key} — dependent_criteria",
                options=dep_opts,
                key=deps_key,
            )
            payload["dependent_criteria"] = [
                d for d in st.session_state[deps_key] if d in dep_opts
            ]

        # remove button for this criterion
        if st.button(f"Remove {crit_key}", key=f"{crit_id}:rm"):
            to_delete.append((idx - 1, crit_id))

# apply removals (from end to start to keep indices valid) and clean widget state
for idx, crit_id in sorted(to_delete, key=lambda x: x[0], reverse=True):
    del st.session_state.criteria[idx]
    _clear_keys_for(crit_id)

# Build JSON (list of { "criterion n": {payload} } )
rubric_list = []
for i, payload in enumerate(st.session_state.criteria, start=1):
    key = f"criterion {i}"
    # don't emit internal _id
    out = deepcopy(payload)
    out.pop("_id", None)
    rubric_list.append({key: out})

# Preview + download
st.markdown("#### Preview JSON")
st.json(rubric_list)

json_str = json.dumps(rubric_list, ensure_ascii=False, indent=2)
st.download_button(
    "Download rubric.json",
    data=json_str,
    file_name="rubric.json",
    mime="application/json",
)
