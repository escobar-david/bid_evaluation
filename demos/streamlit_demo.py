"""
Bid Evaluation Web Interface

A Streamlit application for evaluating competitive bids using multiple weighted criteria.
Upload an Excel file, configure evaluation criteria, and export results.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from bid_evaluation import Evaluator, CustomCriterion
from bid_evaluation import custom_templates

# Safe expression evaluation
try:
    from simpleeval import simple_eval, EvalWithCompoundTypes
except ImportError:
    simple_eval = None
    EvalWithCompoundTypes = None

# Configuration
CONFIGS_DIR = Path(__file__).parent / "configs"
CUSTOM_FUNCTIONS_DIR = Path(__file__).parent / "custom_functions"

# Ensure directories exist
CONFIGS_DIR.mkdir(exist_ok=True)
CUSTOM_FUNCTIONS_DIR.mkdir(exist_ok=True)


def get_sample_data():
    """Returns sample bid data"""
    return pd.DataFrame({
        'vendor': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
        'bid_amount': [50_000_000, 45_000_000, 52_000_000, 48_000_000, 55_000_000],
        'experience': [8, 10, 6, 12, 5],
        'quality_score': [85, 90, 75, 88, 82],
        'team_size': [4, 5, 3, 6, 2],
        'certifications': [2, 4, 1, 3, 2]
    })


def init_session_state():
    """Initialize session state variables."""
    if 'criteria_list' not in st.session_state:
        st.session_state.criteria_list = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None


def load_custom_functions() -> Dict[str, Dict[str, Any]]:
    """Load custom functions from the custom_functions directory."""
    functions = {}

    for py_file in CUSTOM_FUNCTIONS_DIR.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find callable functions (exclude private and imports)
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if callable(obj) and hasattr(obj, "__code__"):
                    # Check if it has the right signature (values, stats)
                    params = obj.__code__.co_varnames[:obj.__code__.co_argcount]
                    if len(params) >= 2:
                        functions[f"{py_file.stem}.{name}"] = {
                            'function': obj,
                            'file': py_file.name,
                            'doc': obj.__doc__ or "No description available"
                        }
        except Exception as e:
            st.warning(f"Error loading {py_file.name}: {e}")

    return functions


def create_formula_function(formula: str, variables: Dict[str, float]) -> Callable:
    """Create a scoring function from a formula expression."""
    if simple_eval is None:
        raise ImportError("simpleeval library required for formula expressions. Install with: pip install simpleeval")

    def formula_func(values: pd.Series, stats: dict) -> pd.Series:
        results = []
        evaluator = EvalWithCompoundTypes()

        # Add safe math functions
        evaluator.functions = {
            'abs': abs,
            'min': min,
            'max': max,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'exp': np.exp,
            'clip': lambda x, lo, hi: max(lo, min(hi, x)),
        }

        for val in values:
            # Build context with value and stats
            context = {
                'value': val,
                'min': stats.get('min', values.min()),
                'max': stats.get('max', values.max()),
                'mean': stats.get('mean', values.mean()),
                'median': stats.get('median', values.median()),
                'std': stats.get('std', values.std()),
                **variables
            }
            evaluator.names = context

            try:
                result = evaluator.eval(formula)
                results.append(float(result))
            except Exception:
                results.append(0.0)

        return pd.Series(results, index=values.index).clip(0, 100)

    return formula_func


def render_sidebar():
    """Render the sidebar with file upload and config management."""
    st.sidebar.header("Data Input")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing bid data"
    )

    if uploaded_file is not None:
        try:
            # Check for multiple sheets
            xl = pd.ExcelFile(uploaded_file)
            sheets = xl.sheet_names

            if len(sheets) > 1:
                selected_sheet = st.sidebar.selectbox("Select Sheet", sheets)
            else:
                selected_sheet = sheets[0]

            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name

            st.sidebar.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    if st.button("📊 Try with Sample Data"):
        df = get_sample_data()
        st.session_state['df'] = df
        st.success("Sample data loaded!")

    st.sidebar.divider()

    # Configuration management
    st.sidebar.header("Configuration")

    # Load config
    config_files = list(CONFIGS_DIR.glob("*.json"))
    if config_files:
        config_names = ["-- Select --"] + [f.stem for f in config_files]
        selected_config = st.sidebar.selectbox("Load Configuration", config_names)

        if selected_config != "-- Select --":
            if st.sidebar.button("Load"):
                load_config(selected_config)
                st.rerun()

    # Save config
    with st.sidebar.expander("Save Configuration"):
        config_name = st.text_input("Configuration Name", key="save_config_name")
        if st.button("Save", key="save_config_btn"):
            if config_name:
                save_config(config_name)
                st.success(f"Saved as {config_name}.json")
            else:
                st.warning("Please enter a name")


def load_config(name: str):
    """Load a configuration from file."""
    filepath = CONFIGS_DIR / f"{name}.json"
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        st.session_state.criteria_list = config.get('criteria', [])
        st.success(f"Loaded configuration: {name}")
    except Exception as e:
        st.error(f"Error loading config: {e}")


def save_config(name: str):
    """Save current configuration to file."""
    filepath = CONFIGS_DIR / f"{name}.json"
    config = {
        'criteria': st.session_state.criteria_list
    }
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def render_data_preview():
    """Render data preview section."""
    df = st.session_state.df

    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Rows:** {len(df)}")
        with col2:
            st.write(f"**Columns:** {', '.join(df.columns)}")


def render_criterion_type_options(criterion_type: str, key_prefix: str, df: pd.DataFrame) -> dict:
    """Render type-specific options and return parameters."""
    params = {}

    if criterion_type == "linear":
        params['higher_is_better'] = st.checkbox(
            "Higher is Better",
            value=True,
            key=f"{key_prefix}_higher",
            help="If checked, higher values get higher scores"
        )

    elif criterion_type == "threshold":
        st.write("**Threshold Ranges** (value >= lower and < upper)")
        num_thresholds = st.number_input(
            "Number of thresholds",
            min_value=1,
            max_value=10,
            value=3,
            key=f"{key_prefix}_num_thresh"
        )

        thresholds = []
        for i in range(int(num_thresholds)):
            cols = st.columns(3)
            with cols[0]:
                lower = st.number_input(f"Lower {i+1}", value=float(i * 10), key=f"{key_prefix}_lower_{i}")
            with cols[1]:
                upper = st.number_input(f"Upper {i+1}", value=float((i + 1) * 10), key=f"{key_prefix}_upper_{i}")
            with cols[2]:
                score = st.number_input(f"Score {i+1}", value=float((i + 1) * 25), key=f"{key_prefix}_score_{i}")
            thresholds.append([lower, upper, score])

        params['thresholds'] = thresholds

    elif criterion_type == "direct":
        col1, col2 = st.columns(2)
        with col1:
            params['input_scale'] = st.number_input(
                "Input Scale",
                value=100.0,
                key=f"{key_prefix}_input_scale",
                help="Original score scale"
            )
        with col2:
            params['output_scale'] = st.number_input(
                "Output Scale",
                value=100.0,
                key=f"{key_prefix}_output_scale",
                help="Desired output scale"
            )

    elif criterion_type == "template":
        template_names = custom_templates.get_template_names()
        selected_template = st.selectbox(
            "Template",
            template_names,
            key=f"{key_prefix}_template"
        )
        params['template_name'] = selected_template

        # Show template-specific parameters
        template_info = custom_templates.get_template_info(selected_template)
        if template_info:
            st.write(f"*{template_info['description']}*")
            template_params = {}
            for param_name, param_info in template_info.get('parameters', {}).items():
                if param_info['type'] == 'float':
                    template_params[param_name] = st.number_input(
                        param_info['label'],
                        value=float(param_info['default']),
                        key=f"{key_prefix}_tpl_{param_name}"
                    )
                elif param_info['type'] == 'bool':
                    template_params[param_name] = st.checkbox(
                        param_info['label'],
                        value=param_info['default'],
                        key=f"{key_prefix}_tpl_{param_name}"
                    )
            params['template_params'] = template_params

    elif criterion_type == "formula":
        params['formula'] = st.text_input(
            "Formula",
            value="100 - abs(value - target) / target * 100",
            key=f"{key_prefix}_formula",
            help="Use: value, min, max, mean, median, std, and custom variables"
        )

        st.write("**Variables:**")
        num_vars = st.number_input("Number of variables", 0, 5, 1, key=f"{key_prefix}_num_vars")
        variables = {}
        for i in range(int(num_vars)):
            cols = st.columns(2)
            with cols[0]:
                var_name = st.text_input(f"Name {i+1}", value="target", key=f"{key_prefix}_var_name_{i}")
            with cols[1]:
                var_value = st.number_input(f"Value {i+1}", value=100000.0, key=f"{key_prefix}_var_val_{i}")
            if var_name:
                variables[var_name] = var_value
        params['formula_variables'] = variables

    elif criterion_type == "custom_python":
        custom_funcs = load_custom_functions()
        if custom_funcs:
            func_names = list(custom_funcs.keys())
            selected_func = st.selectbox(
                "Function",
                func_names,
                key=f"{key_prefix}_custom_func"
            )
            params['custom_function'] = selected_func

            if selected_func:
                func_info = custom_funcs[selected_func]
                st.write(f"**File:** {func_info['file']}")
                with st.expander("Documentation"):
                    st.code(func_info['doc'])
        else:
            st.warning("No custom functions found in custom_functions/ directory")

    return params


def render_criteria_builder():
    """Render the criteria builder section."""
    df = st.session_state.df

    st.header("Evaluation Criteria")

    # Display existing criteria
    if st.session_state.criteria_list:
        st.subheader("Configured Criteria")

        # Calculate total weight
        total_weight = sum(c.get('weight', 0) for c in st.session_state.criteria_list)

        for i, criterion in enumerate(st.session_state.criteria_list):
            with st.container():
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    st.write(f"**{criterion.get('name', criterion.get('column'))}**")
                    st.caption(f"Column: {criterion.get('column')} | Type: {criterion.get('type')}")
                with cols[1]:
                    weight = criterion.get('weight', 0)
                    pct = (weight / total_weight * 100) if total_weight > 0 else 0
                    st.write(f"Weight: {weight} ({pct:.1f}%)")
                with cols[2]:
                    st.progress(pct / 100)
                with cols[3]:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.criteria_list.pop(i)
                        st.rerun()

        st.divider()

    # Add new criterion
    with st.expander("Add New Criterion", expanded=len(st.session_state.criteria_list) == 0):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox(
                "Column",
                numeric_cols,
                key="new_column",
                help="Select the data column to evaluate"
            )

        with col2:
            criterion_type = st.selectbox(
                "Type",
                ["linear", "threshold", "direct", "min_ratio", "geometric_mean", "inverse",
                 "template", "formula", "custom_python"],
                key="new_type",
                help="Select the evaluation method"
            )

        col3, col4 = st.columns(2)
        with col3:
            weight = st.number_input(
                "Weight",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="new_weight"
            )
        with col4:
            name = st.text_input(
                "Display Name",
                value=column if column else "",
                key="new_name"
            )

        # Type-specific options
        type_params = render_criterion_type_options(criterion_type, "new", df)

        if st.button("Add Criterion", type="primary"):
            criterion_config = {
                'column': column,
                'type': criterion_type,
                'weight': weight,
                'name': name or column,
                **type_params
            }
            st.session_state.criteria_list.append(criterion_config)
            st.success(f"Added criterion: {name or column}")
            st.rerun()


def build_evaluator() -> Evaluator:
    """Build an Evaluator from the session state criteria list."""
    evaluator = Evaluator(normalize_weights=True)
    df = st.session_state.df

    for criterion in st.session_state.criteria_list:
        column = criterion['column']
        ctype = criterion['type']
        weight = criterion['weight']
        name = criterion.get('name', column)

        if ctype in ['linear', 'threshold', 'direct', 'min_ratio', 'geometric_mean', 'inverse']:
            # Standard criteria - build config for from_config
            config = {'type': ctype, 'weight': weight, 'name': name}

            if ctype == 'linear':
                config['higher_is_better'] = criterion.get('higher_is_better', True)
            elif ctype == 'threshold':
                config['thresholds'] = criterion.get('thresholds', [])
            elif ctype == 'direct':
                config['input_scale'] = criterion.get('input_scale', 100)
                config['output_scale'] = criterion.get('output_scale', 100)

            # Use from_config for standard types
            temp_eval = Evaluator.from_config({column: config}, normalize_weights=False)
            for col, crit in temp_eval.criteria.items():
                evaluator.add_criterion(col, crit)

        elif ctype == 'template':
            template_name = criterion.get('template_name')
            template_params = criterion.get('template_params', {})

            def make_template_func(tname, tparams):
                def func(values, stats):
                    return custom_templates.apply_template(tname, values, stats, **tparams)
                return func

            evaluator.custom(column, weight, make_template_func(template_name, template_params))

        elif ctype == 'formula':
            formula = criterion.get('formula', 'value')
            variables = criterion.get('formula_variables', {})
            func = create_formula_function(formula, variables)
            evaluator.custom(column, weight, func)

        elif ctype == 'custom_python':
            func_name = criterion.get('custom_function')
            custom_funcs = load_custom_functions()
            if func_name and func_name in custom_funcs:
                func = custom_funcs[func_name]['function']
                evaluator.custom(column, weight, func)

    return evaluator


def render_evaluation():
    """Render evaluation results."""
    if not st.session_state.criteria_list:
        return

    st.header("Evaluation")

    if st.button("Run Evaluation", type="primary"):
        try:
            evaluator = build_evaluator()
            results = evaluator.evaluate(st.session_state.df)
            st.session_state.results_df = results

            # Get statistics
            st.session_state.stats = evaluator.get_statistics()
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if st.session_state.results_df is not None:
        results = st.session_state.results_df

        st.subheader("Results")

        # Highlight winner
        winner_idx = results['ranking'].idxmin()

        # Display results table
        st.dataframe(
            results.style.highlight_min(subset=['ranking'], color='lightgreen'),
            width='stretch'
        )

        # Score breakdown chart (only show actual criterion scores, not raw data or final_score)
        score_cols = [col for col in results.columns if col.startswith('score_')]
        if score_cols:
            st.subheader("Score Breakdown")

            chart_data = results[score_cols].copy()
            chart_data.columns = [col.replace('score_', '') for col in score_cols]

            # Add identifier column if available
            if 'vendor' in results.columns:
                chart_data.index = results['vendor']
            elif 'name' in results.columns:
                chart_data.index = results['name']

            st.bar_chart(chart_data)

        # Statistics
        if hasattr(st.session_state, 'stats') and st.session_state.stats:
            with st.expander("Statistics"):
                stats_df = pd.DataFrame(st.session_state.stats).T
                st.dataframe(stats_df, width='stretch')


def create_excel_output() -> bytes:
    """Create Excel file with results, statistics, and configuration."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Results sheet
        if st.session_state.results_df is not None:
            st.session_state.results_df.to_excel(writer, sheet_name='Results', index=False)

        # Statistics sheet
        if hasattr(st.session_state, 'stats') and st.session_state.stats:
            stats_df = pd.DataFrame(st.session_state.stats).T
            stats_df.to_excel(writer, sheet_name='Statistics')

        # Configuration sheet
        config_data = []
        for criterion in st.session_state.criteria_list:
            config_data.append({
                'Column': criterion.get('column'),
                'Name': criterion.get('name'),
                'Type': criterion.get('type'),
                'Weight': criterion.get('weight'),
                'Parameters': json.dumps({k: v for k, v in criterion.items()
                                          if k not in ['column', 'name', 'type', 'weight']})
            })
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Configuration', index=False)

    return output.getvalue()


def render_export():
    """Render export section."""
    if st.session_state.results_df is None:
        return

    st.header("Export")

    col1, col2 = st.columns(2)

    with col1:
        excel_data = create_excel_output()
        filename = st.session_state.filename or "bids"
        output_name = f"{Path(filename).stem}_results.xlsx"

        st.download_button(
            label="Download Excel Results",
            data=excel_data,
            file_name=output_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        # Also offer JSON config download
        config_json = json.dumps({'criteria': st.session_state.criteria_list}, indent=2)
        st.download_button(
            label="Download Configuration",
            data=config_json,
            file_name="evaluation_config.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown("""
       ### 💼 Need More?

       This is a free demo of the open-source library. 

       For production use with:
       - ☁️ Cloud storage & history
       - 👥 Team collaboration  
       - 📊 Advanced reports
       - 🔗 System integrations
       - 📧 Priority support

       📧 **Get in touch:** davesc78@gmail.com
       """)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Bid Evaluation",
        page_icon="📊",
        layout="wide"
    )

    init_session_state()

    st.info("""
    ⚠️ **Free Demo** - This is a demonstration of the open-source library github.com/escobar-david/bid_evaluation. 
    Evaluations are not saved. For production use with cloud storage,
    history, and team features, contact davesc78@gmail.com
    """)

    st.title("Bid Evaluation System")
    st.caption("Upload bid data, configure evaluation criteria, and analyze results")

    # Sidebar
    render_sidebar()

    # Main content
    if st.session_state.df is not None:
        render_data_preview()
        render_criteria_builder()
        render_evaluation()
        render_export()
    else:
        st.info("Upload an Excel file to get started. Use the file uploader in the sidebar.")

    # Show sample usage (always visible)
    with st.expander("How to Use"):
        st.markdown("""
        1. **Upload Data**: Use the sidebar to upload an Excel file with your bid data
        2. **Configure Criteria**: Add evaluation criteria with different types:
           - **Linear**: Normalize values (higher or lower is better)
           - **Threshold**: Score ranges based on value bands
           - **Direct**: Use pre-scored values directly
           - **Min Ratio**: Score relative to minimum value (common for prices)
           - **Geometric Mean**: Score relative to geometric mean
           - **Inverse**: Inversely proportional scoring
           - **Template**: Pre-built custom patterns (budget proximity, sweet spot, etc.)
           - **Formula**: Write your own scoring formula
           - **Custom Python**: Use custom Python functions
        3. **Run Evaluation**: Click "Run Evaluation" to calculate scores
        4. **Export**: Download results as Excel or save your configuration for reuse
        """)


if __name__ == "__main__":
    main()
