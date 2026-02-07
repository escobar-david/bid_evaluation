"""
Staged Bid Evaluation Demo

A Streamlit app for multi-stage bid evaluation with filtering between stages.
Upload data, configure stages and criteria, evaluate, and explore results.

Run with: streamlit run demos/streamlit_staged_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
from io import BytesIO
from typing import Callable, Dict

from simpleeval import EvalWithCompoundTypes

from bid_evaluation import StagedEvaluator

# === Formula support ===

def create_formula_function(formula: str, variables: Dict[str, float]) -> Callable:
    """Create a scoring function from a formula expression using simpleeval."""
    def formula_func(values: pd.Series, stats: dict) -> pd.Series:
        results = []
        evaluator = EvalWithCompoundTypes()
        evaluator.functions = {
            "abs": abs, "min": min, "max": max,
            "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
            "exp": np.exp,
            "clip": lambda x, lo, hi: max(lo, min(hi, x)),
        }
        for val in values:
            evaluator.names = {
                "value": val,
                "min": stats.get("min", values.min()),
                "max": stats.get("max", values.max()),
                "mean": stats.get("mean", values.mean()),
                "median": stats.get("median", values.median()),
                "std": stats.get("std", values.std()),
                **variables,
            }
            try:
                results.append(float(evaluator.eval(formula)))
            except Exception:
                results.append(0.0)
        return pd.Series(results, index=values.index).clip(0, 100)
    return formula_func


# === Color palette (Tableau 10 — colorblind-friendly) ===

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def build_color_map(stages):
    """Assign a consistent color to each criterion name across all stages."""
    seen = []
    for stage in stages:
        for crit in stage["criteria"]:
            cname = crit.get("name", crit["column"])
            if cname not in seen:
                seen.append(cname)
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(seen)}


# === Display mappings ===

FILTER_OPTIONS = ["None", "Score Threshold", "Top N"]
FILTER_TO_INTERNAL = {"None": None, "Score Threshold": "score_threshold", "Top N": "top_n"}
FILTER_TO_DISPLAY = {v: k for k, v in FILTER_TO_INTERNAL.items()}

CRITERION_OPTIONS = ["Linear", "Direct Score", "Min Ratio", "Geometric Mean", "Inverse", "Threshold", "Formula"]
CTYPE_TO_INTERNAL = {
    "Linear": "linear",
    "Direct Score": "direct",
    "Min Ratio": "min_ratio",
    "Geometric Mean": "geometric_mean",
    "Inverse": "inverse",
    "Threshold": "threshold",
    "Formula": "formula",
}
CTYPE_TO_DISPLAY = {v: k for k, v in CTYPE_TO_INTERNAL.items()}

CRITERION_DESCRIPTIONS = {
    "Linear": "Normalizes values linearly to 0\u2013100. Best for quantities like years of experience.",
    "Direct Score": "Uses values as-is (already scored). For committee/expert ratings.",
    "Min Ratio": "Score = (min value / value) \u00d7 100. Lowest value gets 100. Ideal for price.",
    "Geometric Mean": "Scores relative to the geometric mean. Values at or below the mean get 100.",
    "Inverse": "Inversely proportional: lower values get higher scores. For costs, durations.",
    "Threshold": "Maps value ranges to fixed scores. For categorical/tiered evaluation.",
    "Formula": "Write a math expression. Available: value, min, max, mean, median, std, plus custom variables. Functions: abs, sqrt, log, exp, clip.",
}


# === Sample data ===

def get_sample_data():
    return pd.DataFrame({
        "vendor": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
        "experience_years": [15, 3, 10, 7, 12, 2, 8, 5],
        "quality_score": [88, 45, 92, 65, 78, 40, 85, 58],
        "certifications": [4, 1, 5, 2, 3, 1, 4, 2],
        "bid_amount": [120_000, 85_000, 145_000, 95_000, 110_000, 78_000, 130_000, 92_000],
        "delivery_days": [30, 60, 20, 45, 35, 70, 25, 50],
    })


def get_example_config():
    return {
        "final_score_mode": "last_stage",
        "stages": [
            {
                "name": "Technical",
                "weight": 1.0,
                "filter": {"type": "score_threshold", "threshold": 55.0},
                "criteria": {
                    "experience_years": {"type": "linear", "weight": 0.3, "higher_is_better": True},
                    "quality_score": {"type": "direct", "weight": 0.5},
                    "certifications": {"type": "linear", "weight": 0.2, "higher_is_better": True},
                },
            },
            {
                "name": "Economic",
                "weight": 1.0,
                "criteria": {
                    "bid_amount": {"type": "min_ratio", "weight": 0.6},
                    "delivery_days": {"type": "linear", "weight": 0.4, "higher_is_better": False},
                },
            },
        ],
    }


# === Session state helpers ===

def init_session_state():
    for key, default in {
        "stages": [],
        "next_stage_id": 0,
        "df": None,
        "results_df": None,
        "staged_evaluator": None,
        "stage_results_list": None,
        "final_score_mode": "last_stage",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def add_stage_to_state(name="", filter_type=None, threshold=60.0,
                       top_n=3, on_tie="include", weight=1.0):
    sid = st.session_state.next_stage_id
    if not name:
        name = f"Stage {len(st.session_state.stages) + 1}"
    st.session_state.stages.append({
        "id": sid,
        "name": name,
        "filter_type": filter_type,
        "threshold": threshold,
        "top_n": top_n,
        "on_tie": on_tie,
        "weight": weight,
        "criteria": [],
        "next_crit_id": 0,
    })
    st.session_state.next_stage_id += 1


def remove_stage_from_state(stage_id):
    st.session_state.stages = [s for s in st.session_state.stages if s["id"] != stage_id]
    st.session_state.results_df = None


def add_criterion_to_stage(stage, column, ctype, weight, name, params):
    cid = stage["next_crit_id"]
    stage["criteria"].append({
        "id": cid,
        "column": column,
        "type": ctype,
        "weight": weight,
        "name": name or column,
        **params,
    })
    stage["next_crit_id"] += 1
    st.session_state.results_df = None


def remove_criterion_from_stage(stage, crit_id):
    stage["criteria"] = [c for c in stage["criteria"] if c["id"] != crit_id]
    st.session_state.results_df = None


def sync_from_widgets():
    """Read current widget values back into the stages list.

    If any value changed since the last evaluation, invalidate results
    so stale charts are never shown.
    """
    changed = False
    for stage in st.session_state.stages:
        sid = stage["id"]
        for field, key_suffix in [("name", "name"), ("weight", "weight")]:
            k = f"s{sid}_{key_suffix}"
            if k in st.session_state and stage[field] != st.session_state[k]:
                stage[field] = st.session_state[k]
                changed = True
        # Filter type
        k = f"s{sid}_filter"
        if k in st.session_state:
            new_filter = FILTER_TO_INTERNAL.get(st.session_state[k])
            if stage["filter_type"] != new_filter:
                stage["filter_type"] = new_filter
                changed = True
        for field in ("threshold", "top_n", "on_tie"):
            k = f"s{sid}_{field}"
            if k in st.session_state and stage.get(field) != st.session_state[k]:
                stage[field] = st.session_state[k]
                changed = True
    if changed and st.session_state.results_df is not None:
        st.session_state.results_df = None
        st.session_state.stage_results_list = None
        st.session_state.staged_evaluator = None


def load_config_into_state(config):
    """Populate session state stages from a config dict."""
    st.session_state.final_score_mode = config.get("final_score_mode", "last_stage")
    st.session_state.stages = []
    st.session_state.next_stage_id = 0
    st.session_state.results_df = None

    for stage_cfg in config.get("stages", []):
        ft = None
        threshold = 60.0
        top_n = 3
        on_tie = "include"
        if "filter" in stage_cfg:
            f = stage_cfg["filter"]
            ft = f["type"]
            threshold = f.get("threshold", 60.0)
            top_n = f.get("top_n", 3)
            on_tie = f.get("on_tie", "include")

        add_stage_to_state(
            name=stage_cfg["name"],
            filter_type=ft,
            threshold=threshold,
            top_n=top_n,
            on_tie=on_tie,
            weight=stage_cfg.get("weight", 1.0),
        )
        stage = st.session_state.stages[-1]

        for col, crit_cfg in stage_cfg.get("criteria", {}).items():
            params = {}
            ct = crit_cfg["type"]
            if ct == "linear":
                params["higher_is_better"] = crit_cfg.get("higher_is_better", True)
            elif ct == "direct":
                params["input_scale"] = crit_cfg.get("input_scale", 100.0)
            elif ct == "threshold":
                params["thresholds"] = crit_cfg.get("thresholds", [])
            elif ct == "formula":
                params["formula"] = crit_cfg.get("formula", "value")
                params["formula_variables"] = crit_cfg.get("formula_variables", {})
            add_criterion_to_stage(
                stage, col, ct, crit_cfg["weight"], crit_cfg.get("name", col), params
            )


def build_config_dict():
    """Export current stages to a config dict."""
    sync_from_widgets()
    config = {"final_score_mode": st.session_state.final_score_mode, "stages": []}
    for stage in st.session_state.stages:
        sc = {"name": stage["name"], "weight": stage["weight"], "criteria": {}}
        if stage["filter_type"]:
            sc["filter"] = {"type": stage["filter_type"]}
            if stage["filter_type"] == "score_threshold":
                sc["filter"]["threshold"] = stage["threshold"]
            elif stage["filter_type"] == "top_n":
                sc["filter"]["top_n"] = stage["top_n"]
                sc["filter"]["on_tie"] = stage["on_tie"]
        for crit in stage["criteria"]:
            cc = {"type": crit["type"], "weight": crit["weight"]}
            if crit.get("name") and crit["name"] != crit["column"]:
                cc["name"] = crit["name"]
            if crit["type"] == "linear":
                cc["higher_is_better"] = crit.get("higher_is_better", True)
            elif crit["type"] == "direct":
                cc["input_scale"] = crit.get("input_scale", 100.0)
            elif crit["type"] == "threshold":
                cc["thresholds"] = crit.get("thresholds", [])
            elif crit["type"] == "formula":
                cc["formula"] = crit.get("formula", "value")
                cc["formula_variables"] = crit.get("formula_variables", {})
            sc["criteria"][crit["column"]] = cc
        config["stages"].append(sc)
    return config


# === Charting helpers ===

def _safe_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def _find_id_column(df):
    for candidate in ("vendor", "name", "bidder", "company", "supplier"):
        if candidate in df.columns:
            return candidate
    return None


def make_stacked_bar_chart(chart_df, id_col, value_col, color_col, color_map,
                           title="", sort_ascending=True):
    """Build a horizontal stacked bar chart with Altair.

    Args:
        chart_df: Long-format DataFrame with columns [id_col, color_col, value_col].
        id_col: Column used for the y-axis (bid identifier).
        value_col: Column with numeric values (contributions).
        color_col: Column with category labels (criterion names).
        color_map: Dict mapping category labels to hex colors.
        title: Chart title.
        sort_ascending: Sort order for the y-axis (by total score).
    """
    # Compute totals for sort order
    totals = chart_df.groupby(id_col)[value_col].sum().reset_index()
    totals.columns = [id_col, "_total"]
    sort_order = (
        totals.sort_values("_total", ascending=sort_ascending)[id_col].tolist()
    )

    # Build color scale from the color_map
    domain = list(color_map.keys())
    range_ = [color_map[k] for k in domain]

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_col}:Q", title="Score", stack="zero"),
            y=alt.Y(f"{id_col}:N", sort=sort_order, title=None),
            color=alt.Color(
                f"{color_col}:N",
                scale=alt.Scale(domain=domain, range=range_),
                legend=alt.Legend(title="Criterion", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(f"{id_col}:N", title="Bid"),
                alt.Tooltip(f"{color_col}:N", title="Criterion"),
                alt.Tooltip(f"{value_col}:Q", title="Contribution", format=".1f"),
            ],
        )
        .properties(title=title, height=alt.Step(45))
    )
    return chart


def build_criterion_breakdown(result, stage, id_col, color_map):
    """Build a long-format DataFrame of per-criterion contributions for a stage.

    Each criterion's contribution is proportional to its weight-adjusted score
    so that all contributions sum to the stage's total score per bid.
    """
    safe = _safe_name(stage["name"])
    score_col = f"stage_{safe}_score"

    # Find per-criterion score columns for this stage
    crit_cols = {}
    for crit in stage["criteria"]:
        cname = crit.get("name", crit["column"])
        col = f"stage_{safe}_score_{cname}"
        if col in result.columns:
            crit_cols[cname] = col

    if not crit_cols or score_col not in result.columns:
        return None

    # Only non-eliminated rows that have a stage score
    mask = result[score_col].notna()
    if not mask.any():
        return None

    rows = []
    for idx in result.index[mask]:
        bid_id = result.loc[idx, id_col] if id_col else str(idx)
        stage_score = result.loc[idx, score_col]
        raw_sum = sum(
            result.loc[idx, c] for c in crit_cols.values()
            if pd.notna(result.loc[idx, c])
        )
        for cname, col in crit_cols.items():
            raw = result.loc[idx, col]
            if pd.isna(raw) or raw_sum == 0:
                contribution = 0.0
            else:
                contribution = (raw / raw_sum) * stage_score
            rows.append({
                "bid": str(bid_id),
                "criterion": cname,
                "contribution": contribution,
            })

    return pd.DataFrame(rows)


def build_stage_level_breakdown(result, stages, id_col, color_map):
    """Build a long-format DataFrame of per-stage contributions for weighted_combination mode."""
    total_weight = sum(s["weight"] for s in stages)
    if total_weight == 0:
        return None

    rows = []
    non_elim = result[result["eliminated_at_stage"].isna()]
    for idx in non_elim.index:
        bid_id = non_elim.loc[idx, id_col] if id_col else str(idx)
        for stage in stages:
            safe = _safe_name(stage["name"])
            score_col = f"stage_{safe}_score"
            if score_col in result.columns and pd.notna(result.loc[idx, score_col]):
                contribution = result.loc[idx, score_col] * (stage["weight"] / total_weight)
            else:
                contribution = 0.0
            rows.append({
                "bid": str(bid_id),
                "stage": stage["name"],
                "contribution": contribution,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


# === Sidebar ===

def render_sidebar():
    st.sidebar.header("Data Input")

    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    if uploaded:
        try:
            xl = pd.ExcelFile(uploaded)
            sheets = xl.sheet_names
            sheet = st.sidebar.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
            st.session_state.df = pd.read_excel(uploaded, sheet_name=sheet)
            st.sidebar.success(f"Loaded {len(st.session_state.df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Sample Data"):
            st.session_state.df = get_sample_data()
            st.rerun()
    with col_b:
        if st.button("Try Example"):
            st.session_state.df = get_sample_data()
            load_config_into_state(get_example_config())
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Settings")

    mode_label = "Last Stage" if st.session_state.final_score_mode == "last_stage" else "Weighted Combination"
    mode = st.sidebar.radio(
        "Final Score Mode",
        ["Last Stage", "Weighted Combination"],
        index=0 if mode_label == "Last Stage" else 1,
        help="**Last Stage**: final ranking uses only the last stage's score. "
             "**Weighted Combination**: weighted average across all stage scores.",
    )
    new_mode = "last_stage" if mode == "Last Stage" else "weighted_combination"
    if st.session_state.final_score_mode != new_mode:
        st.session_state.final_score_mode = new_mode
        if st.session_state.results_df is not None:
            st.session_state.results_df = None
            st.session_state.stage_results_list = None
            st.session_state.staged_evaluator = None

    st.sidebar.divider()
    st.sidebar.header("Configuration")

    config_file = st.sidebar.file_uploader("Load Config (JSON)", type=["json"], key="config_upload")
    if config_file:
        try:
            config = json.load(config_file)
            load_config_into_state(config)
            st.sidebar.success("Config loaded")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    if st.session_state.stages:
        config_json = json.dumps(build_config_dict(), indent=2)
        st.sidebar.download_button(
            "Save Config (JSON)",
            data=config_json,
            file_name="staged_config.json",
            mime="application/json",
        )

    st.sidebar.divider()
    with st.sidebar.expander("How to Use"):
        st.markdown(
            "1. **Load Data** \u2014 upload Excel or click *Sample Data*\n"
            "2. **Add Stages** \u2014 click *+ Add Stage*, set name and filter\n"
            "3. **Add Criteria** \u2014 expand *Add Criterion* within each stage\n"
            "4. **Evaluate** \u2014 click *Run Evaluation*\n"
            "5. **Explore** \u2014 review pipeline, charts, per-stage tabs, and export\n\n"
            "Or click **Try Example** for a ready-made demo with sample data and configuration."
        )


# === Data preview ===

def render_data_preview():
    with st.expander("Data Preview", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Rows:** {len(st.session_state.df)}")
        with c2:
            numeric = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"**Numeric columns:** {', '.join(numeric)}")


# === Stage builder ===

def render_stage_builder():
    st.header("Stage Configuration")

    # Pipeline summary bar
    if st.session_state.stages:
        sync_from_widgets()
        parts = []
        for s in st.session_state.stages:
            label = f"**{s['name']}**"
            ncrit = len(s["criteria"])
            crit_note = f"{ncrit} criteria" if ncrit != 1 else "1 criterion"
            if s["filter_type"] == "score_threshold":
                label += f" [{crit_note}, filter: score \u2265 {s['threshold']}]"
            elif s["filter_type"] == "top_n":
                label += f" [{crit_note}, filter: top {s['top_n']}]"
            else:
                label += f" [{crit_note}]"
            parts.append(label)
        st.markdown("Pipeline: " + " \u2192 ".join(parts))

    if st.button("+ Add Stage", type="primary"):
        add_stage_to_state()
        st.rerun()

    color_map = build_color_map(st.session_state.stages)
    num_stages = len(st.session_state.stages)
    for idx, stage in enumerate(st.session_state.stages):
        is_last = (idx == num_stages - 1)
        _render_stage(stage, idx, color_map, is_last)


def _render_stage(stage, idx, color_map, is_last):
    sid = stage["id"]
    # Build a non-redundant expander title
    default_name = f"Stage {idx + 1}"
    if stage["name"] == default_name:
        expander_label = default_name
    else:
        expander_label = f"{idx + 1}. {stage['name']}"

    with st.expander(expander_label, expanded=True):
        # Top row: name, weight, filter, remove
        c1, c2, c3, c4 = st.columns([3, 1.5, 2.5, 0.8])
        with c1:
            st.text_input("Name", value=stage["name"], key=f"s{sid}_name",
                          label_visibility="collapsed", placeholder="Stage name")
        with c2:
            st.number_input("Weight", value=stage["weight"], min_value=0.0,
                            max_value=100.0, step=0.1, format="%.1f",
                            key=f"s{sid}_weight")
        with c3:
            filter_display = FILTER_TO_DISPLAY.get(stage["filter_type"], "None")
            st.selectbox("Filter", FILTER_OPTIONS,
                         index=FILTER_OPTIONS.index(filter_display),
                         key=f"s{sid}_filter")
        with c4:
            st.write("")  # vertical spacer
            if st.button("\U0001f5d1", key=f"s{sid}_del", help="Remove stage"):
                remove_stage_from_state(sid)
                st.rerun()

        # Filter details (conditional)
        current_filter = st.session_state.get(f"s{sid}_filter", "None")
        if current_filter == "Score Threshold":
            st.number_input("Minimum score to advance", value=stage["threshold"],
                            min_value=0.0, step=1.0, key=f"s{sid}_threshold")
        elif current_filter == "Top N":
            tc1, tc2 = st.columns(2)
            with tc1:
                st.number_input("N (top bids to advance)", value=stage["top_n"],
                                min_value=1, step=1, key=f"s{sid}_top_n")
            with tc2:
                on_tie_idx = 0 if stage["on_tie"] == "include" else 1
                st.selectbox("On tie", ["include", "exclude"], index=on_tie_idx,
                             key=f"s{sid}_on_tie",
                             help="include = advance all tied bids; exclude = strict cutoff")
        elif not is_last:
            # Hint: no filter on a non-final stage
            st.caption("Tip: Set a filter to eliminate bids before the next stage.")

        # Configured criteria
        if stage["criteria"]:
            for crit in stage["criteria"]:
                _render_criterion_row(stage, crit, color_map)
        else:
            st.caption("No criteria yet. Add one below.")

        # Add criterion form
        _render_add_criterion(stage)


def _render_criterion_row(stage, crit, color_map):
    sid = stage["id"]
    cid = crit["id"]
    ctype_display = CTYPE_TO_DISPLAY.get(crit["type"], crit["type"])
    cname = crit.get("name", crit["column"])
    color = color_map.get(cname, "#888888")

    extra = ""
    if crit["type"] == "linear":
        direction = "higher is better" if crit.get("higher_is_better", True) else "lower is better"
        extra = f" ({direction})"
    elif crit["type"] == "formula":
        formula = crit.get("formula", "")
        if len(formula) > 30:
            formula = formula[:27] + "..."
        extra = f" `{formula}`"

    # Show display name alongside column if they differ
    if cname != crit["column"]:
        name_label = f"{cname} (<code>{crit['column']}</code>)"
    else:
        name_label = f"<code>{crit['column']}</code>"

    cols = st.columns([3, 2.5, 1.5, 0.5])
    with cols[0]:
        st.markdown(
            f'<span style="color:{color}; font-size:1.1em;">\u25cf</span> '
            f'{name_label}',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(f"{ctype_display}{extra}")
    with cols[2]:
        st.markdown(f"w = {crit['weight']:.1f}")
    with cols[3]:
        if st.button("\u2715", key=f"s{sid}_c{cid}_del"):
            remove_criterion_from_stage(stage, cid)
            st.rerun()


def _sync_display_name(sid):
    """Reset the display name to match the newly selected column."""
    st.session_state[f"s{sid}_ac_name"] = st.session_state[f"s{sid}_ac_col"]


def _render_add_criterion(stage):
    sid = stage["id"]
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns in the data.")
        return

    with st.expander("Add Criterion"):
        ac1, ac2, ac3, ac4 = st.columns([2.5, 2, 2.5, 1])
        with ac1:
            column = st.selectbox("Column", numeric_cols, key=f"s{sid}_ac_col",
                                  on_change=_sync_display_name, args=(sid,))
        with ac2:
            ctype_display = st.selectbox("Type", CRITERION_OPTIONS, key=f"s{sid}_ac_type")
        with ac3:
            # Default to column name; auto-updated when column changes
            if f"s{sid}_ac_name" not in st.session_state:
                st.session_state[f"s{sid}_ac_name"] = column
            display_name = st.text_input("Display Name",
                                         key=f"s{sid}_ac_name",
                                         placeholder="Display name")
        with ac4:
            weight = st.number_input("Weight", value=1.0, min_value=0.0,
                                     step=0.1, key=f"s{sid}_ac_w")

        # Show description for the selected criterion type
        st.caption(CRITERION_DESCRIPTIONS.get(ctype_display, ""))

        ctype = CTYPE_TO_INTERNAL[ctype_display]
        params = {}

        if ctype == "linear":
            params["higher_is_better"] = st.checkbox("Higher is better", value=True,
                                                     key=f"s{sid}_ac_hib")
        elif ctype == "direct":
            params["input_scale"] = st.number_input("Input scale", value=100.0,
                                                    key=f"s{sid}_ac_iscale")
        elif ctype == "threshold":
            num_t = st.number_input("Number of ranges", min_value=1, max_value=10,
                                    value=3, key=f"s{sid}_ac_nt")
            thresholds = []
            for i in range(int(num_t)):
                tc = st.columns(3)
                with tc[0]:
                    lo = st.number_input(f"Lower {i+1}", value=float(i * 30),
                                         key=f"s{sid}_ac_tlo{i}")
                with tc[1]:
                    hi = st.number_input(f"Upper {i+1}", value=float((i+1) * 30),
                                         key=f"s{sid}_ac_thi{i}")
                with tc[2]:
                    sc = st.number_input(f"Score {i+1}", value=float((i+1) * 25),
                                         key=f"s{sid}_ac_tsc{i}")
                thresholds.append([lo, hi, sc])
            params["thresholds"] = thresholds
        elif ctype == "formula":
            params["formula"] = st.text_input(
                "Formula",
                value="100 - abs(value - target) / target * 100",
                key=f"s{sid}_ac_formula",
                help="Use: value, min, max, mean, median, std, plus custom variables",
            )
            num_vars = st.number_input("Custom variables", min_value=0,
                                       max_value=5, value=1, key=f"s{sid}_ac_nvars")
            variables = {}
            for i in range(int(num_vars)):
                vc = st.columns(2)
                with vc[0]:
                    var_name = st.text_input(f"Name {i+1}", value="target",
                                             key=f"s{sid}_ac_vn{i}")
                with vc[1]:
                    var_value = st.number_input(f"Value {i+1}", value=100000.0,
                                                key=f"s{sid}_ac_vv{i}")
                if var_name:
                    variables[var_name] = var_value
            params["formula_variables"] = variables

        if st.button("Add", key=f"s{sid}_ac_btn", type="primary"):
            add_criterion_to_stage(stage, column, ctype, weight,
                                   display_name or column, params)
            st.rerun()


# === Build evaluator ===

def build_staged_evaluator():
    sync_from_widgets()
    staged = StagedEvaluator(final_score_mode=st.session_state.final_score_mode)

    for stage in st.session_state.stages:
        if not stage["criteria"]:
            raise ValueError(f"Stage '{stage['name']}' has no criteria.")

        ft = stage["filter_type"]
        staged.add_stage(
            name=stage["name"],
            filter_type=ft,
            threshold=stage["threshold"] if ft == "score_threshold" else None,
            top_n=int(stage["top_n"]) if ft == "top_n" else None,
            on_tie=stage["on_tie"],
            weight=stage["weight"],
        )

        for crit in stage["criteria"]:
            ct = crit["type"]
            col, w, name = crit["column"], crit["weight"], crit.get("name", crit["column"])

            if ct == "linear":
                staged.linear(col, w, name=name,
                              higher_is_better=crit.get("higher_is_better", True))
            elif ct == "direct":
                staged.direct(col, w, name=name,
                              input_scale=crit.get("input_scale", 100.0))
            elif ct == "min_ratio":
                staged.min_ratio(col, w, name=name)
            elif ct == "geometric_mean":
                staged.geometric_mean(col, w, name=name)
            elif ct == "inverse":
                staged.inverse(col, w, name=name)
            elif ct == "threshold":
                staged.threshold(col, w, thresholds=crit.get("thresholds", []), name=name)
            elif ct == "formula":
                func = create_formula_function(
                    crit.get("formula", "value"),
                    crit.get("formula_variables", {}),
                )
                staged.custom(col, w, func, name=name)

    return staged


# === Evaluation & results ===

def render_evaluation():
    if not st.session_state.stages:
        return

    all_have_criteria = all(len(s["criteria"]) > 0 for s in st.session_state.stages)
    if not all_have_criteria:
        st.info("Add at least one criterion to every stage before evaluating.")
        return

    st.header("Evaluation")

    if st.button("Run Evaluation", type="primary"):
        try:
            staged = build_staged_evaluator()
            result = staged.evaluate(st.session_state.df)
            st.session_state.results_df = result
            st.session_state.staged_evaluator = staged
            st.session_state.stage_results_list = staged.get_stage_results()
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if st.session_state.results_df is not None:
        render_results()


def render_results():
    result = st.session_state.results_df
    stage_results = st.session_state.stage_results_list
    df = st.session_state.df
    stages = st.session_state.stages
    id_col = _find_id_column(result)
    color_map = build_color_map(stages)

    # --- Winner callout ---
    non_elim = result[result["eliminated_at_stage"].isna()]
    if not non_elim.empty and "final_score" in result.columns:
        winner = non_elim.loc[non_elim["final_score"].idxmax()]
        winner_name = winner[id_col] if id_col else f"Row {winner.name}"
        winner_score = winner["final_score"]
        st.success(f"**Winner: {winner_name}** with a final score of **{winner_score:.1f}**")

    # --- Pipeline metrics ---
    st.subheader("Pipeline")
    num_stages = len(stage_results)
    metric_cols = st.columns(num_stages + 1)

    with metric_cols[0]:
        st.metric("Input", f"{len(df)} bids")

    for i, sr in enumerate(stage_results):
        with metric_cols[i + 1]:
            adv = len(sr.advanced_indices)
            elim = len(sr.eliminated_indices)
            stage_cfg = stages[i]
            # Build filter description
            if stage_cfg["filter_type"] == "score_threshold":
                filter_note = f"Filter: score \u2265 {stage_cfg['threshold']}"
            elif stage_cfg["filter_type"] == "top_n":
                filter_note = f"Filter: top {stage_cfg['top_n']}"
            else:
                filter_note = "No filter"

            if elim > 0:
                st.metric(sr.name, f"{adv} passed", delta=f"-{elim} eliminated",
                          delta_color="inverse")
            else:
                st.metric(sr.name, f"{adv} passed", delta="all passed", delta_color="off")
            st.caption(filter_note)

    # --- Tabbed results ---
    sync_from_widgets()
    tab_names = ["Combined"] + [s["name"] for s in stages]
    tabs = st.tabs(tab_names)

    # -- Combined tab --
    with tabs[0]:
        _render_combined_tab(result, stages, stage_results, id_col, color_map)

    # -- Per-stage tabs --
    for i, stage in enumerate(stages):
        with tabs[i + 1]:
            _render_stage_tab(result, stage, stage_results[i], id_col, color_map)

    # --- Export ---
    st.subheader("Export")
    ec1, ec2 = st.columns(2)
    with ec1:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            result.to_excel(writer, sheet_name="Results", index=False)
            if st.session_state.staged_evaluator:
                st.session_state.staged_evaluator.summary().to_excel(
                    writer, sheet_name="Configuration", index=False
                )
        st.download_button("Download Excel", data=output.getvalue(),
                           file_name="staged_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with ec2:
        st.download_button("Download Config (JSON)",
                           data=json.dumps(build_config_dict(), indent=2),
                           file_name="staged_config.json",
                           mime="application/json")


def _render_combined_tab(result, stages, stage_results, id_col, color_map):
    """Render the Combined results tab with table and score breakdown chart."""
    # Pick display columns
    display_cols = []
    if id_col:
        display_cols.append(id_col)
    for s in stages:
        safe = _safe_name(s["name"])
        for suffix in (f"stage_{safe}_score", f"stage_{safe}_ranking"):
            if suffix in result.columns:
                display_cols.append(suffix)
    display_cols.extend(["eliminated_at_stage", "final_score", "ranking"])
    display_cols = [c for c in display_cols if c in result.columns]

    st.dataframe(
        result[display_cols].style.format(
            {c: "{:.1f}" for c in display_cols
             if c in result.columns and pd.api.types.is_float_dtype(result[c])},
            na_rep="\u2014",
        ),
        use_container_width=True,
    )

    # --- Score breakdown chart ---
    mode = st.session_state.final_score_mode

    if mode == "weighted_combination" and len(stages) > 1:
        # Show stage-level contributions (one color per stage)
        stage_df = build_stage_level_breakdown(result, stages, id_col, color_map)
        if stage_df is not None and not stage_df.empty:
            st.markdown("##### Final Score Breakdown by Stage")
            stage_colors = {s["name"]: PALETTE[i % len(PALETTE)] for i, s in enumerate(stages)}
            chart = make_stacked_bar_chart(
                stage_df, "bid", "contribution", "stage", stage_colors,
                title="Weighted stage contributions to final score",
            )
            st.altair_chart(chart, use_container_width=True)

    # Always show the criterion-level breakdown of the deciding stage
    if mode == "last_stage":
        deciding_stage = stages[-1]
        label = f"Final Score Breakdown ({deciding_stage['name']} stage criteria)"
    else:
        deciding_stage = stages[-1]
        label = f"Last Stage Criteria Breakdown ({deciding_stage['name']})"

    breakdown_df = build_criterion_breakdown(result, deciding_stage, id_col, color_map)
    if breakdown_df is not None and not breakdown_df.empty:
        st.markdown(f"##### {label}")
        chart = make_stacked_bar_chart(
            breakdown_df, "bid", "contribution", "criterion", color_map,
            title="Per-criterion contributions",
        )
        st.altair_chart(chart, use_container_width=True)


def _render_stage_tab(result, stage, sr, id_col, color_map):
    """Render a single stage's results tab with table and criterion breakdown chart."""
    if sr.result_df.empty:
        st.info("No bids reached this stage.")
        return

    safe = _safe_name(stage["name"])
    score_col = f"stage_{safe}_score"
    rank_col = f"stage_{safe}_ranking"

    # Show which bids were eliminated here
    elim_here = result[result["eliminated_at_stage"] == stage["name"]]
    if not elim_here.empty and id_col:
        st.warning(
            f"Eliminated at this stage: "
            f"{', '.join(elim_here[id_col].astype(str))}"
        )

    # Stage detail columns
    detail_prefix = f"stage_{safe}_score_"
    detail_cols = [c for c in result.columns if c.startswith(detail_prefix)]
    stage_cols = []
    if id_col:
        stage_cols.append(id_col)
    stage_cols.extend(detail_cols)
    if score_col in result.columns:
        stage_cols.append(score_col)
    if rank_col in result.columns:
        stage_cols.append(rank_col)
    stage_cols = [c for c in stage_cols if c in result.columns]

    # Only show rows that participated in this stage
    participated = sr.result_df.index
    stage_data = result.loc[participated, stage_cols]

    st.dataframe(
        stage_data.style.format(
            {c: "{:.1f}" for c in stage_cols
             if c in result.columns and pd.api.types.is_float_dtype(result[c])},
            na_rep="\u2014",
        ),
        use_container_width=True,
    )

    # --- Criterion breakdown chart ---
    breakdown_df = build_criterion_breakdown(result, stage, id_col, color_map)
    if breakdown_df is not None and not breakdown_df.empty:
        st.markdown(f"##### {stage['name']} \u2014 Criterion Contributions")
        chart = make_stacked_bar_chart(
            breakdown_df, "bid", "contribution", "criterion", color_map,
            title=f"How each criterion contributes to {stage['name']} score",
        )
        st.altair_chart(chart, use_container_width=True)


# === Main ===

def main():
    st.set_page_config(page_title="Staged Bid Evaluation", page_icon="\U0001f4ca", layout="wide")
    init_session_state()

    st.title("Staged Bid Evaluation")
    st.caption("Multi-stage evaluation with filtering between stages")

    render_sidebar()

    if st.session_state.df is not None:
        render_data_preview()
        render_stage_builder()
        render_evaluation()
    else:
        st.info(
            "Upload an Excel file or click **Sample Data** in the sidebar to get started. "
            "Or click **Try Example** for a ready-made demo with sample data and configuration."
        )


if __name__ == "__main__":
    main()
