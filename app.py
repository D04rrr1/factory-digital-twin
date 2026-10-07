from __future__ import annotations

import pandas as pd
import streamlit as st

from factory_data import (
    downtime_events,
    factory_flow,
    production_lines,
    production_plan,
    quality_metrics,
    targets,
)
from failurdesicion import MachineValidationError, analyze_machine


st.set_page_config(
    page_title="Factory Digital Twin",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Factory Digital Twin")
st.caption(
    "Production KPI monitoring + predictive maintenance + AI recommendations"
)

lines_df = pd.DataFrame(production_lines)
downtime_df = pd.DataFrame(downtime_events)
quality_df = pd.DataFrame(quality_metrics)
plan_df = pd.DataFrame(production_plan)

all_dates = sorted(lines_df["date"].unique())
selected_date = st.sidebar.selectbox(
    "Factory date",
    all_dates,
    index=len(all_dates) - 1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("AI target")
target_risk = st.sidebar.slider(
    "Maximum acceptable failure risk",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.05,
)

day_lines = lines_df[lines_df["date"] == selected_date].copy()
day_quality = quality_df[quality_df["date"] == selected_date].copy()
day_downtime = downtime_df[downtime_df["date"] == selected_date].copy()


def risk_icon(level: str) -> str:
    return {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
    }.get(level, "⚪")


def status_label(status: str) -> str:
    return {
        "OPTIMAL": "Optimal",
        "ACTION_REQUIRED": "Action required",
        "CRITICAL": "Critical",
    }.get(status, status)


overview_tab, twin_tab, ai_tab, data_tab = st.tabs(
    [
        "📊 Factory Overview",
        "🏭 Digital Twin",
        "🤖 AI Maintenance",
        "📋 Factory Data",
    ]
)


with overview_tab:
    st.subheader(f"Factory overview — {selected_date}")

    total_plan = int(day_lines["plan"].sum())
    total_fact = int(day_lines["fact"].sum())
    avg_load = float(day_lines["load_percent"].mean())

    total_produced = int(day_quality["produced"].sum())
    total_defects = int(day_quality["defects"].sum())
    weighted_defect_rate = (
        (total_defects / total_produced) * 100 if total_produced else 0.0
    )

    total_downtime = int(day_downtime["duration_min"].sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Production",
        f"{total_fact} / {total_plan}",
        f"{total_fact - total_plan:+d} vs plan",
    )
    c2.metric("Average load", f"{avg_load:.1f}%")
    c3.metric(
        "Defect rate",
        f"{weighted_defect_rate:.1f}%",
        f"limit ≤ {targets['defect_max_percent']}%",
        delta_color="inverse",
    )
    c4.metric(
        "Downtime",
        f"{total_downtime} min",
        f"limit ≤ {targets['critical_downtime_max_min_per_day']} min/day",
        delta_color="inverse",
    )

    st.markdown("### Production lines")

    for _, row in day_lines.iterrows():
        quality_row = day_quality[day_quality["section"] == row["section"]]
        defect = (
            float(quality_row.iloc[0]["defect_percent"])
            if not quality_row.empty
            else 0.0
        )

        if defect > targets["defect_max_percent"]:
            icon = "🔴"
            state = "Quality alert"
        elif row["load_percent"] >= 100:
            icon = "🟡"
            state = "Full capacity"
        else:
            icon = "🟢"
            state = "Operating"

        a, b, c, d, e = st.columns([1.6, 1, 1, 1, 1.3])
        a.markdown(f"**{icon} {row['line']}**")
        b.metric("Plan", int(row["plan"]))
        c.metric("Fact", int(row["fact"]))
        d.metric("Load", f"{row['load_percent']:.0f}%")
        e.metric("Status", state)

    st.markdown("### Automatic alerts")

    alerts = []

    for _, row in day_quality.iterrows():
        if row["defect_percent"] > targets["defect_max_percent"]:
            alerts.append(
                f"🔴 {row['section']}: defect rate {row['defect_percent']:.1f}% "
                f"is above the {targets['defect_max_percent']}% limit."
            )

    if total_downtime > targets["critical_downtime_max_min_per_day"]:
        alerts.append(
            f"🟡 Total downtime today is {total_downtime} min, above the "
            f"{targets['critical_downtime_max_min_per_day']} min/day target."
        )

    for _, row in day_lines.iterrows():
        if row["fact"] < row["plan"]:
            alerts.append(
                f"🟡 {row['line']}: actual output {int(row['fact'])} "
                f"is below plan {int(row['plan'])}."
            )

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No KPI deviations detected for the selected date.")

    st.markdown("### Plan vs actual")
    st.bar_chart(day_lines.set_index("line")[["plan", "fact"]])

    st.markdown("### Quality by section")
    st.bar_chart(day_quality.set_index("section")[["defect_percent"]])


with twin_tab:
    st.subheader("Production flow")

    st.info(
        "This screen uses the factory test data from the case. "
        "The AI sensor model is shown separately in the AI Maintenance tab."
    )

    flow_cols = st.columns(len(factory_flow))
    for i, stage in enumerate(factory_flow):
        if stage == "Окраска":
            flow_cols[i].error(f"🎨\n\n**{stage}**")
        elif stage in ("Сварка", "Сборка"):
            flow_cols[i].success(f"⚙️\n\n**{stage}**")
        else:
            flow_cols[i].info(f"📦\n\n**{stage}**")

    st.markdown("### Live line state")

    for _, row in day_lines.iterrows():
        section_quality = day_quality[day_quality["section"] == row["section"]]
        defect_rate = (
            float(section_quality.iloc[0]["defect_percent"])
            if not section_quality.empty
            else 0.0
        )

        if defect_rate > targets["defect_max_percent"]:
            icon = "🔴"
            message = f"Quality deviation: {defect_rate:.1f}% defects"
        elif row["load_percent"] >= 100:
            icon = "🟡"
            message = "Running at full capacity"
        else:
            icon = "🟢"
            message = "Normal operation"

        with st.container(border=True):
            left, middle, right = st.columns([2, 2, 3])
            left.markdown(f"### {icon} {row['line']}")
            middle.write(f"**Load:** {row['load_percent']:.0f}%")
            middle.progress(min(float(row["load_percent"]) / 100, 1.0))
            right.write(
                f"**Plan / Fact:** {int(row['plan'])} / {int(row['fact'])}"
            )
            right.write(f"**State:** {message}")

    st.markdown("### Downtime events")

    if day_downtime.empty:
        st.success("No downtime events.")
    else:
        st.dataframe(
            day_downtime.rename(
                columns={
                    "date": "Date",
                    "section": "Section",
                    "equipment": "Equipment",
                    "reason": "Reason",
                    "duration_min": "Duration, min",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with ai_tab:
    st.subheader("Predict → Diagnose → Optimize")

    st.write(
        "Enter machine sensor values. AI #1 estimates failure risk, "
        "AI #2 identifies likely causes, and AI #3 searches for safer settings."
    )

    with st.form("machine_analysis_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        machine_type = r1c1.selectbox(
            "Machine type",
            ["L", "M", "H"],
            help="AI4I equipment type used during model training.",
        )
        air_k = r1c2.number_input(
            "Air temperature [K]",
            min_value=290.0,
            max_value=310.0,
            value=298.5,
            step=0.1,
        )
        proc_k = r1c3.number_input(
            "Process temperature [K]",
            min_value=300.0,
            max_value=320.0,
            value=308.7,
            step=0.1,
        )

        r2c1, r2c2, r2c3 = st.columns(3)
        rpm = r2c1.number_input(
            "Rotational speed [rpm]",
            min_value=500.0,
            max_value=4000.0,
            value=1380.0,
            step=10.0,
        )
        torque = r2c2.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=62.0,
            step=0.5,
        )
        wear = r2c3.number_input(
            "Tool wear [min]",
            min_value=0.0,
            max_value=300.0,
            value=210.0,
            step=1.0,
        )

        run_ai = st.form_submit_button(
            "Run AI analysis",
            type="primary",
            use_container_width=True,
        )

    if run_ai:
        machine = {
            "type": machine_type,
            "air_k": air_k,
            "proc_k": proc_k,
            "rpm": rpm,
            "torque": torque,
            "wear": wear,
        }

        try:
            with st.spinner("Running AI models..."):
                result = analyze_machine(
                    machine,
                    target_max_risk=target_risk,
                )

            risk = float(result["failure_risk"])
            optimized_risk = float(result["optimized_risk"])
            level = result["risk_level"]
            status = result["status"]

            st.markdown("### AI result")

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Failure risk", f"{risk * 100:.1f}%")
            x2.metric("Risk level", f"{risk_icon(level)} {level.upper()}")
            x3.metric("AI status", status_label(status))
            x4.metric(
                "Optimized risk",
                f"{optimized_risk * 100:.1f}%",
                f"{(optimized_risk - risk) * 100:+.1f} p.p.",
                delta_color="inverse",
            )

            causes = result.get("causes", [])
            st.markdown("### Diagnosis")

            if causes:
                for cause in causes:
                    st.error(f"{cause['code']} — {cause['label']}")
            else:
                st.success("No failure cause exceeded the alarm threshold.")

            if result.get("extra_action") == "replace_tool":
                st.error(
                    "🛠 Tool replacement recommended. Tool wear cannot be fixed "
                    "only by changing RPM, torque or process temperature."
                )

            st.markdown("### Prescriptive recommendation")
            recommended = result.get("recommended")

            if recommended:
                before_after = pd.DataFrame(
                    {
                        "Parameter": [
                            "RPM",
                            "Torque [Nm]",
                            "Process temperature [K]",
                        ],
                        "Current": [rpm, torque, proc_k],
                        "Recommended": [
                            recommended["rpm"],
                            recommended["torque"],
                            recommended["proc_k"],
                        ],
                    }
                )
                st.dataframe(
                    before_after,
                    use_container_width=True,
                    hide_index=True,
                )

                if result.get("achieved_target"):
                    st.success(
                        f"Target achieved: predicted risk falls to "
                        f"{optimized_risk * 100:.1f}%."
                    )
                else:
                    st.warning(
                        "The optimizer reduced the predicted risk, but the "
                        "selected target could not be reached."
                    )

            elif status == "OPTIMAL":
                st.success(
                    "Current machine mode is already below the selected risk target. "
                    "No parameter change is required."
                )
            else:
                st.error(
                    "No meaningful safe-setting improvement was found. "
                    "Inspection, maintenance or shutdown may be required."
                )

            st.caption(
                "Prototype decision-support system: recommendations require "
                "human/operator approval and are not autonomous PLC commands."
            )

        except MachineValidationError as exc:
            st.error(f"Input validation error: {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info(
                "Put failure_binary_model.joblib and failure_type_model.joblib "
                "in the same folder as app.py."
            )
        except Exception as exc:
            st.exception(exc)


with data_tab:
    st.subheader("Factory test data")

    st.markdown("### Production lines")
    st.dataframe(lines_df, use_container_width=True, hide_index=True)

    st.markdown("### Downtime")
    st.dataframe(downtime_df, use_container_width=True, hide_index=True)

    st.markdown("### Quality")
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.markdown("### Monthly production plan")
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    current_plan = int(plan_df["monthly_plan"].sum())
    st.metric(
        "Provided monthly plan",
        f"{current_plan:,} cars",
        f"Target ≥ {targets['monthly_output_min']:,}",
    )

    if current_plan < targets["monthly_output_min"]:
        st.warning(
            f"The provided model plan totals {current_plan:,} cars, "
            f"which is below the stated target of "
            f"{targets['monthly_output_min']:,} cars/month."
        )

    st.markdown("### Case KPI targets")
    st.write(
        {
            "OEE target": f"≥ {targets['oee_min_percent']}%",
            "Defect rate": f"≤ {targets['defect_max_percent']}%",
            "Critical downtime": (
                f"≤ {targets['critical_downtime_max_min_per_day']} min/day"
            ),
            "Monthly output": f"≥ {targets['monthly_output_min']} cars",
            "Shift": (
                f"{targets['shifts_per_day']} × "
                f"{targets['shift_hours']} hours"
            ),
        }
    )
