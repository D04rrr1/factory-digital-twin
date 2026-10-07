from __future__ import annotations

import altair as alt
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
    initial_sidebar_state="expanded",
)

st.title("🏭 Factory Digital Twin")
st.caption("Production monitoring • Digital Twin • Predictive AI")

lines_df = pd.DataFrame(production_lines)
downtime_df = pd.DataFrame(downtime_events)
quality_df = pd.DataFrame(quality_metrics)
plan_df = pd.DataFrame(production_plan)

all_dates = sorted(lines_df["date"].unique())

with st.sidebar:
    st.markdown("## Factory")
    selected_date = st.selectbox(
        "Snapshot date",
        all_dates,
        index=len(all_dates) - 1,
    )

    with st.expander("Engineer settings"):
        target_risk = st.slider(
            "Maximum acceptable failure risk",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Used by the optimization engine. 0.15 = 15% target risk.",
        )

    st.markdown("---")
    st.caption("Prototype decision-support system")

day_lines = lines_df[lines_df["date"] == selected_date].copy()
day_quality = quality_df[quality_df["date"] == selected_date].copy()
day_downtime = downtime_df[downtime_df["date"] == selected_date].copy()


def get_section_quality(section: str) -> float:
    row = day_quality[day_quality["section"] == section]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["defect_percent"])


def line_status(section: str, load: float) -> tuple[str, str]:
    defect = get_section_quality(section)
    if defect > targets["defect_max_percent"]:
        return "🔴", "Quality issue"
    if load >= 100:
        return "🟡", "Full capacity"
    return "🟢", "Normal"


def risk_icon(level: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "⚪")


def status_label(status: str) -> str:
    return {
        "OPTIMAL": "Optimal",
        "ACTION_REQUIRED": "Action required",
        "CRITICAL": "Critical",
    }.get(status, status)


def build_alerts() -> list[dict]:
    alerts = []

    for _, row in day_quality.iterrows():
        if row["defect_percent"] > targets["defect_max_percent"]:
            severity = "critical" if row["defect_percent"] >= 4 else "warning"
            alerts.append({
                "severity": severity,
                "text": (
                    f"{row['section']} defect rate: "
                    f"{row['defect_percent']:.1f}% "
                    f"(limit {targets['defect_max_percent']}%)"
                ),
            })

    total_downtime = int(day_downtime["duration_min"].sum())
    if total_downtime > targets["critical_downtime_max_min_per_day"]:
        alerts.append({
            "severity": "warning",
            "text": (
                f"Total downtime: {total_downtime} min "
                f"(target ≤ {targets['critical_downtime_max_min_per_day']} min/day)"
            ),
        })

    if not day_lines.empty:
        tmp = day_lines.copy()
        tmp["gap"] = tmp["fact"] - tmp["plan"]
        worst = tmp.sort_values("gap").iloc[0]
        if worst["gap"] < 0:
            alerts.append({
                "severity": "warning",
                "text": (
                    f"{worst['line']} output: {int(worst['fact'])} / "
                    f"{int(worst['plan'])} ({int(worst['gap'])} vs plan)"
                ),
            })

    return alerts


alerts = build_alerts()

overview_tab, twin_tab, ai_tab = st.tabs(
    ["📊 Overview", "🏭 Digital Twin", "🤖 AI Control Center"]
)


with overview_tab:
    st.subheader(f"Factory Overview — {selected_date}")

    total_plan = int(day_lines["plan"].sum())
    total_fact = int(day_lines["fact"].sum())
    avg_load = float(day_lines["load_percent"].mean())

    total_produced = int(day_quality["produced"].sum())
    total_defects = int(day_quality["defects"].sum())
    defect_rate = (
        total_defects / total_produced * 100
        if total_produced
        else 0.0
    )

    total_downtime = int(day_downtime["duration_min"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Production",
        f"{total_fact} / {total_plan}",
        f"{total_fact - total_plan:+d} vs plan",
    )
    k2.metric("Average load", f"{avg_load:.1f}%")
    k3.metric(
        "Defect rate",
        f"{defect_rate:.1f}%",
        f"limit ≤ {targets['defect_max_percent']}%",
        delta_color="inverse",
    )
    k4.metric(
        "Downtime",
        f"{total_downtime} min",
        f"target ≤ {targets['critical_downtime_max_min_per_day']} min/day",
        delta_color="inverse",
    )
    k5.metric("Active alerts", len(alerts))

    st.caption(
        f"OEE target from the case: ≥ {targets['oee_min_percent']}%. "
        "The provided test file does not contain enough data to calculate true OEE, "
        "so this dashboard does not invent an OEE value."
    )

    left, right = st.columns([1.8, 1])

    with left:
        st.markdown("### Production lines")
        table_rows = []
        for _, row in day_lines.iterrows():
            defect = get_section_quality(row["section"])
            icon, status = line_status(row["section"], float(row["load_percent"]))
            table_rows.append({
                "Line": f"{icon} {row['line']}",
                "Load": f"{row['load_percent']:.0f}%",
                "Plan / Fact": f"{int(row['plan'])} / {int(row['fact'])}",
                "Defect rate": f"{defect:.1f}%",
                "Status": status,
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("### Top alerts")
        if not alerts:
            st.success("No active KPI alerts.")
        else:
            for alert in alerts[:3]:
                if alert["severity"] == "critical":
                    st.error(f"🔴 {alert['text']}")
                else:
                    st.warning(f"🟡 {alert['text']}")

            if len(alerts) > 3:
                with st.expander(f"Show all alerts ({len(alerts)})"):
                    for alert in alerts[3:]:
                        st.write(f"• {alert['text']}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Plan vs Actual Output")

        long_output = day_lines.melt(
            id_vars=["line"],
            value_vars=["plan", "fact"],
            var_name="Series",
            value_name="Cars",
        )
        long_output["Series"] = long_output["Series"].map({
            "plan": "Plan",
            "fact": "Actual",
        })

        output_chart = (
            alt.Chart(long_output)
            .mark_bar()
            .encode(
                x=alt.X("line:N", title=None),
                xOffset="Series:N",
                y=alt.Y("Cars:Q", title="Cars"),
                color=alt.Color("Series:N", title=None),
                tooltip=["line:N", "Series:N", "Cars:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(output_chart, use_container_width=True)

    with c2:
        st.markdown("### Defect Rate by Section")

        bars = (
            alt.Chart(day_quality)
            .mark_bar()
            .encode(
                x=alt.X("section:N", title=None),
                y=alt.Y("defect_percent:Q", title="Defect rate (%)"),
                tooltip=[
                    "section:N",
                    alt.Tooltip("defect_percent:Q", format=".1f"),
                ],
            )
        )

        limit_df = pd.DataFrame({"limit": [targets["defect_max_percent"]]})

        limit_line = (
            alt.Chart(limit_df)
            .mark_rule(strokeDash=[6, 4])
            .encode(
                y="limit:Q",
                tooltip=[
                    alt.Tooltip(
                        "limit:Q",
                        title="Allowed limit",
                        format=".1f",
                    )
                ],
            )
        )

        st.altair_chart(
            (bars + limit_line).properties(height=320),
            use_container_width=True,
        )


with twin_tab:
    st.subheader("Digital Twin — Production Flow")
    st.caption("Current production snapshot based on the factory test data.")

    flow_cols = st.columns(len(factory_flow))

    section_to_line = {
        row["section"]: row
        for _, row in day_lines.iterrows()
    }

    for i, stage in enumerate(factory_flow):
        with flow_cols[i]:
            if stage in section_to_line:
                row = section_to_line[stage]
                defect = get_section_quality(stage)
                icon, status = line_status(stage, float(row["load_percent"]))

                st.markdown(f"### {icon} {stage}")
                st.write(f"**Load:** {row['load_percent']:.0f}%")
                st.write(
                    f"**Plan / Fact:** "
                    f"{int(row['plan'])} / {int(row['fact'])}"
                )
                st.write(f"**Defects:** {defect:.1f}%")
                st.caption(status)

            elif stage == "Контроль качества":
                st.markdown("### 🛡️ Quality Control")
                st.write("Final inspection")
                st.caption("Monitoring stage")

            elif stage == "Склад комплектующих":
                st.markdown("### 📦 Components")
                st.write("Input inventory")
                st.caption("Material supply")

            elif stage == "Склад готовой продукции":
                st.markdown("### 📦 Finished Goods")
                st.write("Ready vehicles")
                st.caption("Output warehouse")

    st.markdown("---")
    st.markdown("### Line details")

    detail_cols = st.columns(len(day_lines))

    for i, (_, row) in enumerate(day_lines.iterrows()):
        defect = get_section_quality(row["section"])
        icon, status = line_status(row["section"], float(row["load_percent"]))

        with detail_cols[i]:
            with st.container(border=True):
                st.markdown(f"#### {icon} {row['line']}")
                st.write(f"Load: **{row['load_percent']:.0f}%**")
                st.progress(min(float(row["load_percent"]) / 100, 1.0))
                st.write(
                    f"Output: **{int(row['fact'])} / {int(row['plan'])}**"
                )
                st.write(f"Defects: **{defect:.1f}%**")
                st.caption(status)

    st.markdown("### Downtime events")

    if day_downtime.empty:
        st.success("No downtime events for this date.")
    else:
        show = day_downtime.rename(
            columns={
                "date": "Date",
                "section": "Section",
                "equipment": "Equipment",
                "reason": "Reason",
                "duration_min": "Duration, min",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


with ai_tab:
    st.subheader("AI Control Center")
    st.caption("Predict → Diagnose → Optimize")

    input_col, result_col = st.columns([1, 1.25])

    with input_col:
        st.markdown("### 1. Input parameters")

        with st.form("machine_analysis_form"):
            machine_type = st.selectbox(
                "Machine type",
                ["L", "M", "H"],
                help="AI4I machine type used during model training.",
            )

            c1, c2 = st.columns(2)
            air_k = c1.number_input(
                "Air temperature [K]",
                min_value=290.0,
                max_value=310.0,
                value=298.5,
                step=0.1,
            )
            proc_k = c2.number_input(
                "Process temperature [K]",
                min_value=300.0,
                max_value=320.0,
                value=308.7,
                step=0.1,
            )

            c3, c4 = st.columns(2)
            rpm = c3.number_input(
                "Rotational speed [rpm]",
                min_value=500.0,
                max_value=4000.0,
                value=1380.0,
                step=10.0,
            )
            torque = c4.number_input(
                "Torque [Nm]",
                min_value=0.0,
                max_value=100.0,
                value=62.0,
                step=0.5,
            )

            wear = st.number_input(
                "Tool wear [min]",
                min_value=0.0,
                max_value=300.0,
                value=210.0,
                step=1.0,
            )

            run_ai = st.form_submit_button(
                "▶ Run AI analysis",
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

            st.session_state["ai_result"] = result
            st.session_state["ai_machine"] = machine
            st.session_state["approved"] = False

        except MachineValidationError as exc:
            st.error(f"Input validation error: {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info(
                "Place both .joblib model files in the same folder as app.py."
            )
        except Exception as exc:
            st.exception(exc)

    with result_col:
        st.markdown("### 2. AI result")

        if "ai_result" not in st.session_state:
            st.info("Run AI analysis to see prediction and recommendations.")

        else:
            result = st.session_state["ai_result"]
            machine = st.session_state["ai_machine"]

            risk = float(result["failure_risk"])
            optimized_risk = float(result["optimized_risk"])
            status = result["status"]
            level = result["risk_level"]

            r1, r2, r3 = st.columns(3)
            r1.metric("Failure risk", f"{risk * 100:.1f}%")
            r2.metric("Risk level", f"{risk_icon(level)} {level.upper()}")
            r3.metric(
                "Optimized risk",
                f"{optimized_risk * 100:.1f}%",
                f"{(optimized_risk - risk) * 100:+.1f} p.p.",
                delta_color="inverse",
            )

            st.write(f"**AI status:** {status_label(status)}")

            st.markdown("### 3. Diagnosis")

            causes = result.get("causes", [])

            if causes:
                for cause in causes:
                    st.error(f"**{cause['code']} — {cause['label']}**")
            else:
                st.success("No failure cause exceeded the alarm threshold.")

            if result.get("extra_action") == "replace_tool":
                st.warning(
                    f"**Additional maintenance issue**\n\n"
                    f"Tool wear: **{machine['wear']:.0f} min**\n\n"
                    f"Recommended action: **Replace tool**"
                )

            st.markdown("### 4. Recommended settings")

            recommended = result.get("recommended")

            if recommended:
                before_after = pd.DataFrame(
                    {
                        "Parameter": [
                            "RPM",
                            "Torque [Nm]",
                            "Process temperature [K]",
                        ],
                        "Current": [
                            machine["rpm"],
                            machine["torque"],
                            machine["proc_k"],
                        ],
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
                        f"Predicted risk: "
                        f"**{risk * 100:.1f}% → {optimized_risk * 100:.1f}%**"
                    )
                else:
                    st.warning(
                        "Risk was reduced, but the selected target was not reached."
                    )

                approve = st.button(
                    "✅ Approve recommendation",
                    type="primary",
                    use_container_width=True,
                )

                if approve:
                    st.session_state["approved"] = True

                if st.session_state.get("approved"):
                    st.success(
                        "Recommendation approved. "
                        "In a production environment, approved settings could be "
                        "sent to PLC/MES after safety validation."
                    )

            elif status == "OPTIMAL":
                st.success(
                    "Current operating mode is already below the target risk. "
                    "No parameter adjustment is required."
                )
            else:
                st.error(
                    "No meaningful parameter adjustment reached the target. "
                    "Inspection, maintenance or shutdown may be required."
                )

            st.caption(
                "Decision-support only. The prototype does not autonomously control industrial equipment."
            )


st.markdown("---")

with st.expander("⚙️ Technical / Raw Data"):
    st.caption(
        "Source tables used by the dashboard. Hidden by default because "
        "this information is mainly useful for technical review."
    )

    raw1, raw2 = st.columns(2)

    with raw1:
        st.markdown("#### Production lines")
        st.dataframe(lines_df, use_container_width=True, hide_index=True)

        st.markdown("#### Quality")
        st.dataframe(quality_df, use_container_width=True, hide_index=True)

    with raw2:
        st.markdown("#### Downtime")
        st.dataframe(downtime_df, use_container_width=True, hide_index=True)

        st.markdown("#### Monthly production plan")
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

    supplied_plan = int(plan_df["monthly_plan"].sum())

    st.write(
        f"**Provided model plan total:** {supplied_plan:,} cars/month"
    )
    st.write(
        f"**Case production target:** ≥ {targets['monthly_output_min']:,} cars/month"
    )

    if supplied_plan < targets["monthly_output_min"]:
        st.warning(
            "The sum of the provided model-level plan is below the stated "
            "factory monthly target. This is kept exactly as supplied by the case data."
        )
