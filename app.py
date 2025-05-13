# demo_phase4_streamlit.py
# Streamlit UI for Phase 4: Conveyor Belt Performance
# Usage:
# 1. Install dependencies:
#    pip install streamlit numpy
# 2. Run via module:
#    python -m streamlit run demo_phase4_streamlit.py

import streamlit as st
import threading
import time
import random
from collections import deque
import numpy as np

# --- Mock AI Model Functions ---
def load_base_model():
    return {}

def fine_tune_model(model, dataset_size):
    # Simulate model tuning
    time.sleep(1)
    return model

def optimize_for_edge(model):
    # Simulate optimization
    time.sleep(0.5)
    return model

def detect_objects(model):
    count = random.randint(0, 5)
    return [{'class': random.choice(['Box','Cylinder','Package']),
             'confidence': round(random.uniform(0.5,0.99),2)}
            for _ in range(count)]

# --- Conveyor Control ---
class ConveyorControl:
    def __init__(self):
        self.speed = 1.0
    def adjust_speed(self, load_factor):
        self.speed = max(0.5, min(2.0, 1.0/(load_factor+0.1)))
    def divert_item(self, cls):
        pass

# --- Sensor Simulation ---
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = deque(maxlen=50)

def simulate_sensors():
    while True:
        st.session_state.sensor_data.append({
            'proximity': random.uniform(0,1),
            'load': random.uniform(0,100)
        })
        time.sleep(0.1)

# Start sensor simulation thread
sensor_thread = threading.Thread(target=simulate_sensors, daemon=True)
sensor_thread.start()

# --- UI Setup ---
st.set_page_config(page_title="Conveyor Belt Performance", layout="wide")
st.title(" Conveyor Belt System Performance")

# Sidebar: Controls
st.sidebar.header("Control Panel")

# AI Enhancement Button
if st.sidebar.button("Run AI Enhancement"):
    model = load_base_model()
    model = fine_tune_model(model, dataset_size=100)
    model = optimize_for_edge(model)
    st.sidebar.success("AI model fine-tuned and optimized.")

# Initialize metrics storage\if 'metrics' not in st.session_state:
    st.session_state.metrics = {'detections': [], 'times': []}

# Performance Test Button
if st.sidebar.button("Start Performance Test"):
    model = load_base_model()
    control = ConveyorControl()
    for _ in range(20):
        start = time.time()
        dets = detect_objects(model)
        elapsed = time.time() - start
        # Compute average load from sensor data
        loads = [d['load'] for d in st.session_state.sensor_data]
        avg_load = np.mean(loads) if loads else 0
        control.adjust_speed(avg_load/100)
        # Record metrics
        st.session_state.metrics['detections'].append(len(dets))
        st.session_state.metrics['times'].append(elapsed)
    st.sidebar.success("Performance test completed.")

# Main Dashboard
st.subheader("Live Sensor Data")
if st.session_state.sensor_data:
    df = np.array([[d['proximity'], d['load']] for d in st.session_state.sensor_data])
    st.line_chart(df, labels=['Proximity','Load'])

st.subheader("Test Metrics")
metrics = st.session_state.metrics
if metrics['detections']:
    col1, col2 = st.columns(2)
    col1.metric("Avg Detections", f"{np.mean(metrics['detections']):.2f}")
    col2.metric("Avg Process Time (s)", f"{np.mean(metrics['times']):.3f}")
    st.line_chart({
        'Detections': metrics['detections'],
        'Process Time': metrics['times']
    })

st.markdown("---")
st.write("Run `python -m streamlit run demo_phase4_streamlit.py` to launch the app. Ensure `streamlit` is installed and available in your PATH.")
