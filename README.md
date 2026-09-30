# AI-Driven-Predictive-Thermal-Battery-Management-Protocol

"""
AI-Driven Predictive Thermal and Battery Management Protocol
for Snapdragon-Powered Devices

Run:
    pip install streamlit plotly scikit-learn pandas numpy
    streamlit run app.py

This is a simulation/prototype. It does NOT directly control Snapdragon
CPU/GPU frequencies or voltages. The policy layer only simulates what
an operating-system/firmware thermal-management layer could recommend.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Thermal & Battery Management",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }

        .metric-card {
            background: linear-gradient(135deg, #161b22, #21262d);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid #30363d;
            margin-bottom: 10px;
        }

        .metric-title {
            font-size: 14px;
            color: #8b949e;
        }

        .metric-value {
            font-size: 30px;
            font-weight: 700;
            margin-top: 5px;
        }

        .metric-sub {
            font-size: 13px;
            color: #8b949e;
        }

        .protocol-box {
            padding: 18px;
            border-radius: 14px;
            margin: 10px 0 20px 0;
            border: 1px solid;
        }

        .safe {
            background-color: rgba(46, 160, 67, 0.12);
            border-color: #2ea043;
        }

        .warning {
            background-color: rgba(210, 153, 34, 0.12);
            border-color: #d29922;
        }

        .danger {
            background-color: rgba(248, 81, 73, 0.12);
            border-color: #f85149;
        }

        .small-text {
            color: #8b949e;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model():
    """
    Generate synthetic telemetry data and train a Random Forest model.

    Features:
        CPU load
        GPU load
        Ambient temperature
        Current temperature

    Targets:
        Future temperature
        Battery drain rate
    """

    rng = np.random.default_rng(42)

    samples = 6000

    cpu_load = rng.uniform(0, 100, samples)
    gpu_load = rng.uniform(0, 100, samples)
    ambient_temp = rng.uniform(15, 45, samples)
    current_temp = rng.uniform(30, 85, samples)

    # Simulated physical relationship.
    workload_factor = (
        0.055 * cpu_load
        + 0.075 * gpu_load
    )

    thermal_factor = (
        0.20 * (current_temp - ambient_temp)
        + 0.08 * (ambient_temp - 25)
    )

    future_temp = (
        current_temp
        + workload_factor
        + thermal_factor
        + rng.normal(0, 1.8, samples)
    )

    # Battery drain is expressed as % battery/hour.
    battery_drain = (
        1.2
        + 0.018 * cpu_load
        + 0.026 * gpu_load
        + 0.035 * np.maximum(ambient_temp - 25, 0)
        + 0.015 * np.maximum(current_temp - 45, 0)
        + rng.normal(0, 0.25, samples)
    )

    X = pd.DataFrame(
        {
            "cpu_load": cpu_load,
            "gpu_load": gpu_load,
            "ambient_temp": ambient_temp,
            "current_temp": current_temp,
        }
    )

    y = pd.DataFrame(
        {
            "future_temp": future_temp,
            "battery_drain": battery_drain,
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    temp_error = mean_absolute_error(
        y_test["future_temp"],
        predictions[:, 0],
    )

    battery_error = mean_absolute_error(
        y_test["battery_drain"],
        predictions[:, 1],
    )

    return model, temp_error, battery_error


model, temp_error, battery_error = train_model()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "time",
            "cpu",
            "gpu",
            "ambient",
            "current_temp",
            "predicted_temp",
            "battery_drain",
            "battery_level",
        ]
    )

if "current_temp" not in st.session_state:
    st.session_state.current_temp = 38.0

if "battery_level" not in st.session_state:
    st.session_state.battery_level = 100.0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Controls")

st.sidebar.markdown(
    """
    Adjust the simulated hardware telemetry.

    **CPU/GPU Load:** simulated workload intensity  
    **Ambient Temperature:** surrounding temperature
    """
)

cpu_load = st.sidebar.slider(
    "CPU Load (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=1,
)

gpu_load = st.sidebar.slider(
    "GPU Load (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
)

ambient_temp = st.sidebar.slider(
    "Ambient Temperature (°C)",
    min_value=10,
    max_value=50,
    value=25,
    step=1,
)

st.sidebar.divider()

st.sidebar.subheader("Simulation")

update_step = st.sidebar.button(
    "🔄 Run AI Prediction",
    use_container_width=True,
)

continuous_mode = st.sidebar.checkbox(
    "Enable Continuous Simulation",
    value=False,
)

if st.sidebar.button(
    "🗑️ Reset Simulation",
    use_container_width=True,
):
    st.session_state.history = pd.DataFrame(
        columns=[
            "time",
            "cpu",
            "gpu",
            "ambient",
            "current_temp",
            "predicted_temp",
            "battery_drain",
            "battery_level",
        ]
    )

    st.session_state.current_temp = 38.0
    st.session_state.battery_level = 100.0

    st.rerun()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_telemetry(
    cpu,
    gpu,
    ambient,
    current,
):
    """
    Send current telemetry to the Random Forest model.
    """

    input_data = pd.DataFrame(
        {
            "cpu_load": [cpu],
            "gpu_load": [gpu],
            "ambient_temp": [ambient],
            "current_temp": [current],
        }
    )

    prediction = model.predict(input_data)[0]

    predicted_temperature = float(prediction[0])
    predicted_battery_drain = max(0.1, float(prediction[1]))

    return predicted_temperature, predicted_battery_drain


# ============================================================
# POLICY ENGINE
# ============================================================

def thermal_policy(predicted_temp):
    """
    Simulated AI mitigation policy.

    NOTE:
    This does not actually change Snapdragon frequency/voltage.
    It only recommends a simulated action.
    """

    if predicted_temp >= 85:
        return {
            "state": "CRITICAL",
            "class": "danger",
            "action": "Emergency thermal mitigation",
            "frequency": "50%",
            "voltage": "-15%",
            "message": (
                "Critical thermal condition predicted. "
                "Simulated aggressive throttling activated."
            ),
        }

    elif predicted_temp >= 75:
        return {
            "state": "WARNING",
            "class": "warning",
            "action": "Predictive thermal mitigation",
            "frequency": "70%",
            "voltage": "-8%",
            "message": (
                "Thermal spike predicted. "
                "AI recommends proactive frequency scaling "
                "and simulated voltage reduction."
            ),
        }

    elif predicted_temp >= 65:
        return {
            "state": "ELEVATED",
            "class": "warning",
            "action": "Light performance optimization",
            "frequency": "85%",
            "voltage": "-3%",
            "message": (
                "Temperature is elevated. "
                "AI recommends light workload optimization."
            ),
        }

    else:
        return {
            "state": "NORMAL",
            "class": "safe",
            "action": "No mitigation required",
            "frequency": "100%",
            "voltage": "0%",
            "message": (
                "Thermal conditions are within the normal "
                "operating simulation range."
            ),
        }


# ============================================================
# SIMULATION STEP
# ============================================================

def run_simulation_step():
    """
    Generate one telemetry point and update rolling history.
    """

    current = st.session_state.current_temp

    predicted_temp, battery_drain = predict_telemetry(
        cpu_load,
        gpu_load,
        ambient_temp,
        current,
    )

    policy = thermal_policy(predicted_temp)

    # Simulate thermal movement toward predicted temperature.
    # This creates a smoother time series.
    thermal_change = (
        predicted_temp - current
    ) * 0.25

    random_variation = np.random.normal(0, 0.35)

    new_current_temp = (
        current
        + thermal_change
        + random_variation
    )

    # Apply a simulated cooling effect after mitigation.
    if policy["state"] == "WARNING":
        new_current_temp -= 0.15

    elif policy["state"] == "CRITICAL":
        new_current_temp -= 0.35

    new_current_temp = float(
        np.clip(new_current_temp, 20, 100)
    )

    # Battery drain for one simulated minute.
    battery_drop = battery_drain / 60

    new_battery_level = max(
        0,
        st.session_state.battery_level - battery_drop,
    )

    st.session_state.current_temp = new_current_temp
    st.session_state.battery_level = new_battery_level

    new_row = pd.DataFrame(
        [
            {
                "time": datetime.now(),
                "cpu": cpu_load,
                "gpu": gpu_load,
                "ambient": ambient_temp,
                "current_temp": new_current_temp,
                "predicted_temp": predicted_temp,
                "battery_drain": battery_drain,
                "battery_level": new_battery_level,
            }
        ]
    )

    st.session_state.history = pd.concat(
        [
            st.session_state.history,
            new_row,
        ],
        ignore_index=True,
    )

    # Keep rolling history to last 100 points.
    st.session_state.history = (
        st.session_state.history.tail(100)
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 AI-Driven Predictive Thermal & Battery Management"
)

st.markdown(
    """
    ### Snapdragon-Powered Device Simulation

    An ML-based software-layer prototype that predicts future
    thermal conditions and battery drain from simulated hardware
    telemetry, then applies a proactive thermal-management policy.
    """
)

st.divider()


# ============================================================
# RUN SIMULATION
# ============================================================

if update_step or continuous_mode:
    run_simulation_step()


# ============================================================
# CURRENT PREDICTION
# ============================================================

predicted_temp, predicted_drain = predict_telemetry(
    cpu_load,
    gpu_load,
    ambient_temp,
    st.session_state.current_temp,
)

policy = thermal_policy(predicted_temp)


# ============================================================
# POLICY BANNER
# ============================================================

st.markdown(
    f"""
    <div class="protocol-box {policy['class']}">
        <h3>🛡️ AI Thermal Policy: {policy['state']}</h3>
        <p>{policy['message']}</p>
        <b>Recommended simulated frequency:</b>
        {policy['frequency']}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Simulated voltage adjustment:</b>
        {policy['voltage']}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRIC CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Current Temperature</div>
            <div class="metric-value">
                {st.session_state.current_temp:.1f} °C
            </div>
            <div class="metric-sub">
                Simulated device core
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">AI Predicted Temperature</div>
            <div class="metric-value">
                {predicted_temp:.1f} °C
            </div>
            <div class="metric-sub">
                Next thermal state
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Battery Level</div>
            <div class="metric-value">
                {st.session_state.battery_level:.1f}%
            </div>
            <div class="metric-sub">
                Simulated battery
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Battery Drain</div>
            <div class="metric-value">
                {predicted_drain:.2f}%/hr
            </div>
            <div class="metric-sub">
                AI prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Policy State</div>
            <div class="metric-value">
                {policy['state']}
            </div>
            <div class="metric-sub">
                {policy['action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LIVE TELEMETRY
# ============================================================

st.subheader("📡 Current Telemetry")

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.metric(
        "CPU Load",
        f"{cpu_load}%",
    )

with t2:
    st.metric(
        "GPU Load",
        f"{gpu_load}%",
    )

with t3:
    st.metric(
        "Ambient Temperature",
        f"{ambient_temp} °C",
    )

with t4:
    st.metric(
        "Current Core Temperature",
        f"{st.session_state.current_temp:.1f} °C",
    )


# ============================================================
# PLOTLY THERMAL CHART
# ============================================================

st.subheader("🌡️ Predictive Thermal Forecast")

history = st.session_state.history.copy()

if not history.empty:

    fig_temp = go.Figure()

    fig_temp.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["current_temp"],
            mode="lines+markers",
            name="Current Temperature",
            line=dict(width=3),
        )
    )

    fig_temp.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["predicted_temp"],
            mode="lines+markers",
            name="AI Predicted Temperature",
            line=dict(
                width=2,
                dash="dash",
            ),
        )
    )

    # Thermal threshold.
    fig_temp.add_hline(
        y=75,
        line_dash="dot",
        annotation_text="AI Mitigation Threshold: 75°C",
    )

    # Critical threshold.
    fig_temp.add_hline(
        y=85,
        line_dash="dot",
        annotation_text="Critical Threshold: 85°C",
    )

    fig_temp.update_layout(
        height=450,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True,
    )

else:

    st.info(
        "Click **Run AI Prediction** to generate telemetry "
        "and start the thermal simulation."
    )


# ============================================================
# BATTERY CHART
# ============================================================

st.subheader("🔋 Battery Simulation")

if not history.empty:

    fig_battery = go.Figure()

    fig_battery.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["battery_level"],
            mode="lines+markers",
            name="Battery Level",
            line=dict(width=3),
        )
    )

    fig_battery.update_layout(
        height=350,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Battery (%)",
        yaxis=dict(
            range=[0, 100]
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_battery,
        use_container_width=True,
    )


# ============================================================
# WORKLOAD CHART
# ============================================================

st.subheader("⚙️ Workload Intensity")

if not history.empty:

    fig_load = go.Figure()

    fig_load.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["cpu"],
            mode="lines",
            name="CPU Load",
        )
    )

    fig_load.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["gpu"],
            mode="lines",
            name="GPU Load",
        )
    )

    fig_load.update_layout(
        height=350,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Load (%)",
        yaxis=dict(
            range=[0, 100]
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_load,
        use_container_width=True,
    )


# ============================================================
# ML MODEL INFORMATION
# ============================================================

st.subheader("🧠 ML Prediction Engine")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Model",
        "Random Forest",
    )

with m2:
    st.metric(
        "Temperature MAE",
        f"{temp_error:.2f} °C",
    )

with m3:
    st.metric(
        "Battery MAE",
        f"{battery_error:.2f}%/hr",
    )

st.markdown(
    """
    **Model inputs**

    `CPU Load + GPU Load + Ambient Temperature + Current Temperature`

    **Model outputs**

    `Future Temperature + Battery Drain Rate`

    The training dataset in this prototype is synthetic. For a
    real Snapdragon implementation, the training data should come
    from actual device telemetry such as CPU/GPU utilization,
    thermal sensors, battery current/voltage, skin temperature,
    battery temperature, workload type, and power states.
    """
)


# ============================================================
# TELEMETRY TABLE
# ============================================================

with st.expander("📊 View Raw Telemetry"):

    if not history.empty:
        display_df = history.copy()

        display_df["time"] = display_df["time"].dt.strftime(
            "%H:%M:%S"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No telemetry generated yet.")


# ============================================================
# PROTOCOL ARCHITECTURE
# ============================================================

st.subheader("🏗️ Protocol Architecture")

st.code(
    """
┌───────────────────────────────┐
│       Device Telemetry        │
│ CPU | GPU | Temp | Battery    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Feature Processing       │
│ Normalize / Validate Inputs   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       ML Prediction           │
│     Random Forest Model       │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
 Future Temperature   Battery Drain
        │                │
        └───────┬────────┘
                ▼
┌───────────────────────────────┐
│       Policy Engine           │
│                               │
│ <65°C → Normal                │
│ 65-75°C → Optimization        │
│ >75°C → Predictive Throttle   │
│ >85°C → Critical Mitigation   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Simulated Performance Policy  │
│ Frequency Scaling / Voltage   │
│ Reduction / Workload Control  │
└───────────────────────────────┘
    """,
    language="text",
)


# ============================================================
# CONTINUOUS SIMULATION
# ============================================================

if continuous_mode:

    time.sleep(1)

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prototype only • Snapdragon hardware telemetry is simulated • "
    "No real frequency or voltage changes are performed"
)
