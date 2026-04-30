import json
import hmac
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import nextmv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nextmv import cloud


APP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = APP_DIR / "default_input.json"


st.set_page_config(
    page_title="Steel Inventory Optimization",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    h1 {
        letter-spacing: 0;
        color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        background: #f1f5f9;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e293b;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 1.45rem;
        white-space: nowrap;
    }
    .status-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        color: #475569;
        margin: 12px 0 18px 0;
    }
    .intro-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        color: #334155;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def option_key(*parts):
    return "__".join(str(part).replace(" ", "_").replace("/", "_") for part in parts)


@st.cache_data
def load_default_parameters():
    with open(DEFAULT_INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["parameters"]


def sku_rows_to_dataframe(parameters):
    rows = []
    for row in parameters["skus"]:
        item = deepcopy(row)
        channels = set(item.pop("channels", []))
        item["base_channel"] = "base" in channels
        item["discount_channel"] = "discount" in channels
        item["scrap_channel"] = "scrap" in channels
        rows.append(item)
    return pd.DataFrame(rows)


def dataframe_to_sku_rows(dataframe):
    rows = []
    numeric_columns = [
        "forecast",
        "stock",
        "dead_stock",
        "scrap_stock",
        "purchase_cost",
        "selling_price",
        "share_lower",
        "share_upper",
        "alpha_base",
        "alpha_discount",
        "alpha_scrap",
        "short_penalty",
        "hold_penalty",
        "rec_base",
        "rec_promo",
        "rec_scrap",
    ]

    for raw in dataframe.to_dict("records"):
        category = "" if pd.isna(raw.get("category")) else str(raw.get("category", "")).strip()
        sku = "" if pd.isna(raw.get("sku")) else str(raw.get("sku", "")).strip()
        if not category and not sku:
            continue

        row = {
            "category": category,
            "sku": sku,
        }
        for column in numeric_columns:
            value = raw.get(column, 0)
            row[column] = 0.0 if pd.isna(value) else float(value)

        channels = []
        if raw.get("base_channel"):
            channels.append("base")
        if raw.get("discount_channel"):
            channels.append("discount")
        if raw.get("scrap_channel"):
            channels.append("scrap")
        row["channels"] = channels
        rows.append(row)
    return rows


def derive_categories(sku_rows):
    categories = []
    for row in sku_rows:
        category = row["category"]
        if category and category not in categories:
            categories.append(category)
    return categories


def normalize_theta_rows(dataframe):
    rows = []
    for raw in dataframe.to_dict("records"):
        row = {}
        for column in ("target_category", "target_sku", "driver_category", "driver_sku"):
            value = raw.get(column, "")
            row[column] = "" if pd.isna(value) else str(value).strip()

        value = raw.get("value", 0)
        row["value"] = 0.0 if pd.isna(value) else float(value)

        if any(row[column] for column in ("target_category", "target_sku", "driver_category", "driver_sku")):
            rows.append(row)
    return rows


def validate_parameters(parameters):
    errors = []
    sku_pairs = set()

    for row in parameters["skus"]:
        label = f"{row['category']} / {row['sku']}"
        if not row["category"]:
            errors.append("SKU table has a row with a blank category.")
        if not row["sku"]:
            errors.append(f"{row['category'] or 'Blank category'}: SKU cannot be blank.")
        pair = (row["category"], row["sku"])
        if pair in sku_pairs:
            errors.append(f"{label}: duplicate category/SKU pair.")
        sku_pairs.add(pair)

        for column in ("forecast", "stock", "dead_stock", "scrap_stock", "purchase_cost", "selling_price"):
            if row[column] < 0:
                errors.append(f"{label}: {column} cannot be negative.")
        if row["share_lower"] < 0 or row["share_upper"] < 0:
            errors.append(f"{label}: share bounds cannot be negative.")
        if row["share_lower"] > row["share_upper"]:
            errors.append(f"{label}: share_lower cannot be greater than share_upper.")
        if not row["channels"]:
            errors.append(f"{label}: select at least one channel.")

    for category in parameters["categories"]:
        category_rows = [row for row in parameters["skus"] if row["category"] == category]
        lower_sum = sum(row["share_lower"] for row in category_rows)
        upper_sum = sum(row["share_upper"] for row in category_rows)
        if lower_sum > 1 + 1e-9:
            errors.append(f"{category}: sum of share_lower is {lower_sum:.3f}, which is greater than 1.")
        if upper_sum < 1 - 1e-9:
            errors.append(f"{category}: sum of share_upper is {upper_sum:.3f}, which is less than 1.")

    for row in parameters.get("theta", []):
        target = (row["target_category"], row["target_sku"])
        driver = (row["driver_category"], row["driver_sku"])
        if target not in sku_pairs:
            errors.append(f"Theta target {target[0]} / {target[1]} is not in the SKU table.")
        if driver not in sku_pairs:
            errors.append(f"Theta driver {driver[0]} / {driver[1]} is not in the SKU table.")
        if row["value"] < 0:
            errors.append(
                f"Theta {row['target_category']} / {row['target_sku']}: value cannot be negative."
            )

    return errors


def render_parameter_editor(parameters):
    edited = deepcopy(parameters)
    st.subheader("Input Data")
    st.caption("Edit SKU-level inputs here. Forecast and budget are controlled by the sidebar sliders.")

    with st.expander("SKU data", expanded=False):
        sku_df = sku_rows_to_dataframe(edited)
        edited_sku_df = st.data_editor(
            sku_df,
            key="sku_input_editor",
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            disabled=["forecast"],
            column_config={
                "category": st.column_config.TextColumn("Category"),
                "sku": st.column_config.TextColumn("SKU"),
                "forecast": st.column_config.NumberColumn("Forecast", min_value=0.0, step=1.0),
                "stock": st.column_config.NumberColumn("Stock", min_value=0.0, step=1.0),
                "dead_stock": st.column_config.NumberColumn("Dead stock", min_value=0.0, step=1.0),
                "scrap_stock": st.column_config.NumberColumn("Scrap stock", min_value=0.0, step=1.0),
                "purchase_cost": st.column_config.NumberColumn("Purchase cost", min_value=0.0, step=500.0),
                "selling_price": st.column_config.NumberColumn("Selling price", min_value=0.0, step=500.0),
                "share_lower": st.column_config.NumberColumn("Share lower", min_value=0.0, max_value=1.0, step=0.01),
                "share_upper": st.column_config.NumberColumn("Share upper", min_value=0.0, max_value=1.0, step=0.01),
                "alpha_base": st.column_config.NumberColumn("Base usage", min_value=0.0, step=0.01),
                "alpha_discount": st.column_config.NumberColumn("Discount usage", min_value=0.0, step=0.01),
                "alpha_scrap": st.column_config.NumberColumn("Scrap usage", min_value=0.0, step=0.01),
                "short_penalty": st.column_config.NumberColumn("Short penalty", min_value=0.0, step=500.0),
                "hold_penalty": st.column_config.NumberColumn("Hold penalty", min_value=0.0, step=100.0),
                "rec_base": st.column_config.NumberColumn("Base recovery", min_value=0.0, step=500.0),
                "rec_promo": st.column_config.NumberColumn("Promo recovery", min_value=0.0, step=500.0),
                "rec_scrap": st.column_config.NumberColumn("Scrap recovery", min_value=0.0, step=500.0),
                "base_channel": st.column_config.CheckboxColumn("Base"),
                "discount_channel": st.column_config.CheckboxColumn("Promo"),
                "scrap_channel": st.column_config.CheckboxColumn("Scrap"),
            },
        )
        edited["skus"] = dataframe_to_sku_rows(edited_sku_df)
        edited["categories"] = derive_categories(edited["skus"])

    with st.expander("Cross-SKU interaction theta", expanded=False):
        theta_df = pd.DataFrame(edited.get("theta", []))
        if theta_df.empty:
            theta_df = pd.DataFrame(
                columns=["target_category", "target_sku", "driver_category", "driver_sku", "value"]
            )
        edited_theta_df = st.data_editor(
            theta_df,
            key="theta_input_editor",
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "target_category": st.column_config.TextColumn("Target category"),
                "target_sku": st.column_config.TextColumn("Target SKU"),
                "driver_category": st.column_config.TextColumn("Driver category"),
                "driver_sku": st.column_config.TextColumn("Driver SKU"),
                "value": st.column_config.NumberColumn("Value", min_value=0.0, step=0.01),
            },
        )
        edited["theta"] = normalize_theta_rows(edited_theta_df)

    errors = validate_parameters(edited)
    if errors:
        with st.expander("Input validation issues", expanded=True):
            for error in errors:
                st.error(error)

    return edited, errors


def load_multi_file_output(output_dir):
    output_root = Path(output_dir)
    solution_path = output_root / "outputs" / "solutions" / "solution.json"
    assets_path = output_root / "outputs" / "assets" / "assets.json"

    if not solution_path.exists():
        matches = list(output_root.rglob("solution.json"))
        if matches:
            solution_path = matches[0]

    if not assets_path.exists():
        matches = list(output_root.rglob("assets.json"))
        if matches:
            assets_path = matches[0]

    if not solution_path.exists():
        raise RuntimeError(f"Cloud run completed, but solution file was not found at {solution_path}.")

    with open(solution_path, "r", encoding="utf-8") as file:
        output = json.load(file)

    if "solution" not in output:
        output = {"solution": output}

    if assets_path.exists():
        with open(assets_path, "r", encoding="utf-8") as file:
            assets_output = json.load(file)
        output["assets"] = assets_output.get("assets", [])

    return output


def normalize_nextmv_output(output):
    if hasattr(output, "to_dict"):
        output = output.to_dict()
    if isinstance(output, str):
        output = json.loads(output)
    if output is None:
        return {}
    if "solution" in output:
        return output
    if "status" in output and "objective" in output:
        return {"solution": output, "assets": []}
    return output


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


def check_login():
    expected_username = get_secret("APP_USERNAME", "")
    expected_password = get_secret("APP_PASSWORD", "")

    if not expected_password:
        st.warning("APP_PASSWORD is not configured. The app is currently open without a login gate.")
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Steel Inventory Optimization")
    st.caption("Sign in to continue.")

    with st.form("login_form"):
        username = st.text_input("Username") if expected_username else ""
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        username_ok = not expected_username or hmac.compare_digest(username, expected_username)
        password_ok = hmac.compare_digest(password, expected_password)
        if username_ok and password_ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


def run_model_nextmv(parameters, options, cloud_config):
    api_key = cloud_config["api_key"]
    app_id = cloud_config["app_id"]
    instance_id = cloud_config["instance_id"]

    if not api_key:
        return None, [], None, "NEXTMV_API_KEY is not configured in Streamlit secrets."

    try:
        client = cloud.Client(api_key=api_key)
        app = cloud.Application(client=client, id=app_id)
        run_configuration = nextmv.RunConfiguration(
            format=nextmv.Format(
                format_input=nextmv.FormatInput(input_type=nextmv.InputFormat.MULTI_FILE),
                format_output=nextmv.FormatOutput(output_type=nextmv.OutputFormat.MULTI_FILE),
            )
        )
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            input_path = Path(input_dir) / "input.json"
            with open(input_path, "w", encoding="utf-8") as file:
                json.dump({"parameters": parameters}, file)

            result = app.new_run_with_result(
                input_dir_path=input_dir,
                configuration=run_configuration,
                instance_id=instance_id,
                run_options={key: str(value) for key, value in options.items()},
                output_dir_path=output_dir,
            )

            if result.error_log and result.error_log.error:
                raise RuntimeError(result.error_log.error)

            try:
                output = load_multi_file_output(output_dir)
            except RuntimeError:
                output = normalize_nextmv_output(result.output)

            solution = output.get("solution", {})
            assets = output.get("assets", [])
            if "status" not in solution:
                raise RuntimeError(f"Cloud output did not contain a solution object. Received keys: {list(output)}")
        return solution, assets, result, None
    except Exception as exc:
        return None, [], None, str(exc)


def build_options_ui(parameters):
    st.sidebar.header("Scenario Controls")

    with st.sidebar.form("scenario_form"):
        st.caption("Backend: Nextmv Cloud")
        app_id = st.text_input(
            "Nextmv App ID",
            value=get_secret("NEXTMV_APP_ID", "steel-center"),
            help="Use the exact app id shown in Nextmv Cloud. A 404 means this value is wrong.",
        )
        instance_id = st.text_input("Nextmv Instance ID", value=get_secret("NEXTMV_INSTANCE_ID", "devint"))

        st.divider()
        budget = st.slider(
            "Budget",
            min_value=0.0,
            max_value=200_000_000.0,
            value=float(parameters["budget"]),
            step=1_000_000.0,
        )

        options = {"budget": budget}

        st.subheader("Category Forecasts")
        for category in parameters["categories"]:
            forecast = next(row["forecast"] for row in parameters["skus"] if row["category"] == category)
            options[option_key("forecast", category)] = st.slider(
                f"{category}",
                min_value=0.0,
                max_value=200.0,
                value=float(forecast),
                step=1.0,
            )

        st.subheader("Recovery & Limits")
        options["base_usage_multiplier"] = st.slider("Base usage multiplier", 0.0, 2.0, 1.0, 0.05)
        options["discount_usage_multiplier"] = st.slider("Discount usage multiplier", 0.0, 2.0, 1.0, 0.05)
        options["scrap_usage_multiplier"] = st.slider("Scrap usage multiplier", 0.0, 2.0, 1.0, 0.05)
        options["short_penalty_multiplier"] = st.slider("Short penalty multiplier", 0.0, 5.0, 1.0, 0.1)
        options["holding_penalty_multiplier"] = st.slider("Holding penalty multiplier", 0.0, 5.0, 1.0, 0.1)
        options["recovery_multiplier"] = st.slider("Recovery multiplier", 0.0, 2.0, 1.0, 0.05)

        submitted = st.form_submit_button("Run optimization", use_container_width=True)

    cloud_config = {
        "api_key": get_secret("NEXTMV_API_KEY", ""),
        "app_id": app_id.strip(),
        "instance_id": instance_id.strip() or "devint",
    }
    return options, cloud_config, submitted


def format_money(value):
    value = float(value)
    if abs(value) >= 10_000_000:
        return f"{value / 10_000_000:,.2f} Cr"
    if abs(value) >= 100_000:
        return f"{value / 100_000:,.2f} L"
    return f"{value:,.2f}"


def friendly_cloud_error(error):
    if not error:
        return None
    if "application-id with identifier" in error or "status code 404" in error:
        return (
            "Nextmv could not find the App ID. Check the App ID field in the sidebar, "
            "confirm the app was pushed, and make sure the API key belongs to the same Nextmv account."
        )
    if "NEXTMV_API_KEY" in error:
        return "NEXTMV_API_KEY is not configured. Add it to Streamlit Cloud secrets."
    return error


def metric_row(solution):
    summary = solution["spend_summary"]
    recovery = solution["earnings_from_dead_stock_recovery"]
    stuck = solution["dead_stock_remaining_value"]
    scrap = solution["scrap_stock_value"]

    cols = st.columns(5)
    cols[0].metric("Purchase Spend", format_money(summary["total_purchase_spend"]))
    cols[1].metric("Remaining Budget", format_money(summary["remaining_budget"]))
    cols[2].metric("Total Recovery", format_money(recovery["total_recovery"]))
    cols[3].metric("Dead Stock Stuck", format_money(stuck["total_stuck_dead_stock_amount"]))
    cols[4].metric("Scrap Remaining", format_money(scrap["remaining_scrap_amount"]))


def render_charts(assets):
    if not assets:
        st.info("No chart assets returned.")
        return
    st.subheader("Optimization Graphs")
    graph_asset = assets[0]
    content = graph_asset.get("content", graph_asset)
    figure = go.Figure(content)
    st.plotly_chart(figure, use_container_width=True)


def render_tables(solution):
    tabs = st.tabs(["Purchases", "Demand", "Dead Stock", "Options", "JSON"])

    with tabs[0]:
        st.dataframe(pd.DataFrame(solution["purchase_recommendations"]), use_container_width=True)

    with tabs[1]:
        st.dataframe(pd.DataFrame(solution["demand_allocation"]), use_container_width=True)

    with tabs[2]:
        st.dataframe(pd.DataFrame(solution["dead_stock_actions"]), use_container_width=True)

    with tabs[3]:
        options_df = pd.DataFrame(
            [{"option": key, "value": value} for key, value in solution["options_used"].items()]
        )
        st.dataframe(options_df, use_container_width=True)

    with tabs[4]:
        st.json(solution)
        st.download_button(
            "Download solution JSON",
            data=json.dumps({"solution": solution}, indent=2),
            file_name="steel_solution.json",
            mime="application/json",
        )


def main():
    if not check_login():
        return

    st.title("Steel Inventory Optimization")
    st.caption("Scenario planning for purchasing, dead-stock recovery, and scrap liquidation.")

    parameters = load_default_parameters()
    edited_parameters, input_errors = render_parameter_editor(parameters)
    options, cloud_config, submitted = build_options_ui(edited_parameters)

    if st.sidebar.button("Reset scenario"):
        st.session_state.pop("run_output", None)
        st.session_state.pop("sku_input_editor", None)
        st.session_state.pop("theta_input_editor", None)
        st.rerun()

    if submitted:
        if input_errors:
            st.error("Fix the input validation issues before running the optimization.")
        else:
            with st.spinner("Solving optimization model..."):
                st.session_state.run_output = run_model_nextmv(edited_parameters, options, cloud_config)

    if "run_output" not in st.session_state:
        st.info("Set scenario controls in the sidebar, then click **Run optimization**.")
        st.markdown(
            """
            <div class="intro-box">
            This app does not solve automatically when the page opens. Change the scenario controls,
            choose the backend, and submit the form when you are ready to run.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **Backend notes**

            - `Nextmv Cloud` requires a valid API key and the exact Nextmv App ID.
            - A `404 application-id not found` error means the App ID does not exist in your Nextmv account or has not been pushed.
            - This public frontend does not contain the optimization model code.
            """
        )
        return

    solution, assets, run_result, cloud_error = st.session_state.run_output

    if run_result is not None:
        st.success(f"Nextmv Cloud run completed: {run_result.id}")
        if run_result.console_url:
            st.link_button("Open run in Nextmv", run_result.console_url)
    else:
        if cloud_error:
            st.error(f"Nextmv Cloud run failed. {friendly_cloud_error(cloud_error)}")
            return
        else:
            st.caption("Nextmv Cloud run did not return run metadata.")

    st.markdown(
        f"""
        <div class="status-box">
        Stage 1: <b>{solution['status']['stage_1']}</b> |
        Stage 2: <b>{solution['status']['stage_2']}</b> |
        Purchase objective: <b>{format_money(solution['objective']['stage_2_purchase_cost_value'])}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_row(solution)
    render_charts(assets)
    render_tables(solution)


if __name__ == "__main__":
    main()
