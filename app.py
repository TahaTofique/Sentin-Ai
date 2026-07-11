# =============================================================================
# app.py - The Interactive User Interface (Streamlit Dashboard)
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit.components.v1 as components
from senitai import train_ai_models 
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SentinAI Engine", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM DARK THEME CSS (Sidebar bug fixed) ---
st.markdown("""<style>
    .stApp { background-color: #060b18; color: #dce4f8; }
    section[data-testid="stSidebar"] { background-color: #0c1424; border-right: 1px solid #1a2744; }
    
    /* SAFE WAY TO HIDE ELEMENTS WITHOUT BREAKING THE SIDEBAR BUTTON */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background-color: #060b18; }
    
    block-container { padding-top: 2rem; }
    .metric-card { background-color: #0c1424; border: 1px solid #1a2744; border-radius: 12px; padding: 20px; text-align: center; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-card:hover { border-color: #00ccff; box-shadow: 0 0 15px rgba(0,204,255,0.15); transform: translateY(-2px); }
    .metric-value { font-size: 2.5rem; font-weight: 700; font-family: monospace; margin: 10px 0; }
    .metric-label { font-size: 0.85rem; color: #5a6a8a; text-transform: uppercase; letter-spacing: 1px; }
    .status-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    .pulse-dot { width: 10px; height: 10px; border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); } 70% { box-shadow: 0 0 0 10px rgba(0,255,136,0); } }
    .chart-container { background-color: #0c1424; border: 1px solid #1a2744; border-radius: 12px; padding: 15px; margin-bottom: 20px; }
    div.stButton > button { width: 100%; border: 1px solid #1a2744; background-color: #0a1020; color: #8892a8; transition: all 0.2s; border-radius: 8px; text-align: left; }
    div.stButton > button:hover { border-color: #e040fb; color: #e040fb; background-color: rgba(224, 64, 251, 0.05); }
    
    /* Terminal Log Aesthetic */
    .terminal-box { background-color: #000000; border: 1px solid #1a2744; border-radius: 8px; padding: 15px; font-family: monospace; font-size: 12px; color: #00ff88; height: 150px; overflow-y: auto; }
</style>""", unsafe_allow_html=True)

# --- CINEMATIC WELCOME / SPLASH SCREEN ---
if 'welcomed' not in st.session_state:
    # Hide Streamlit UI completely
    st.markdown("""<style>
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 0px; padding-bottom: 0px; margin: 0; }
        header { visibility: hidden; }
    </style>""", unsafe_allow_html=True)

    # Use components.html to bypass Streamlit's broken markdown parser
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; background: #020408; }
        .splash-wrapper { width: 100vw; height: 100vh; background: #020408; display: flex; flex-direction: column; justify-content: center; align-items: center; overflow: hidden; position: relative; }
        .ring1 { position: absolute; width: 400px; height: 400px; border: 1px solid rgba(0, 243, 255, 0.1); border-radius: 50%; animation: spin 6s linear infinite; }
        .ring2 { position: absolute; width: 300px; height: 300px; border: 1px dashed rgba(0, 243, 255, 0.15); border-radius: 50%; animation: spin 4s linear infinite reverse; }
        .ring3 { position: absolute; width: 200px; height: 200px; border: 2px solid rgba(0, 243, 255, 0.2); border-radius: 50%; animation: spin 2s linear infinite; box-shadow: 0 0 30px rgba(0, 243, 255, 0.1); }
        .shield { font-size: 60px; margin-bottom: 20px; animation: float 3s ease-in-out infinite; }
        .main-title { color: #fff; font-family: Consolas, monospace; font-size: 4rem; margin: 0; letter-spacing: 15px; text-shadow: 0 0 10px rgba(0, 243, 255, 0.8), 0 0 40px rgba(0, 243, 255, 0.4); }
        .sub-title { color: #5a6a8a; font-family: Consolas, monospace; font-size: 14px; letter-spacing: 8px; margin-top: 10px; }
        .progress-bg { width: 350px; height: 4px; background: #111; border-radius: 2px; margin-top: 60px; overflow: hidden; }
        .progress-fill { width: 0%; height: 100%; background: linear-gradient(90deg, #00f3ff, #00ff88); box-shadow: 0 0 15px #00f3ff; animation: load 4.5s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
        .boot-text { margin-top: 30px; font-family: Consolas, monospace; font-size: 12px; color: #5a6a8a; text-align: left; width: 300px; }
        .boot-line { animation: fadeIn 0.5s ease-in forwards; color: #00ff88; margin: 4px 0; opacity: 0; animation-fill-mode: forwards; }
        .boot-line-d1 { animation-delay: 1.2s; }
        .boot-line-d2 { animation-delay: 2.2s; }
        .boot-line-d3 { animation-delay: 3.2s; color: #00f3ff; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-15px); } }
        @keyframes load { 0% { width: 0%; } 100% { width: 100%; } }
        @keyframes fadeIn { to { opacity: 1; } }
    </style>
    </head>
    <body>
        <div class="splash-wrapper">
            <div class="ring1"></div>
            <div class="ring2"></div>
            <div class="ring3"></div>
            <div class="shield">🛡️</div>
            <h1 class="main-title">SENTINAI</h1>
            <p class="sub-title">PREDICTIVE INTELLIGENCE ENGINE</p>
            <div class="progress-bg">
                <div class="progress-fill"></div>
            </div>
            <div class="boot-text">
                <p class="boot-line">[OK] INITIALIZING AI CORE</p>
                <p class="boot-line boot-line-d1">[OK] LOADING RANDOM FOREST MODEL</p>
                <p class="boot-line boot-line-d2">[OK] LOADING ISOLATION FOREST MODEL</p>
                <p class="boot-line boot-line-d3">[..] ESTABLISHING SYSTEM TELEMETRY</p>
            </div>
        </div>
    </body>
    </html>
    """, height=800, scrolling=False)

    time.sleep(5)
    st.session_state.welcomed = True
    st.rerun()

# --- LOAD AI MODELS ---
@st.cache_resource
def load_models():
    return train_ai_models()

rf_model, if_model, lr_model, feature_cols = load_models()

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = {'time': [], 'cpu': [], 'ram': [], 'disk_io': [], 'network': [], 'anomaly_score': [], 'state': [], 'prediction': []}
    st.session_state.sim_mode = 'Normal'
    st.session_state.alerts = []
    st.session_state.logs = ["[SYSTEM] SentinAI Engine Booted Successfully.", "[AI] Models loaded. Monitoring active."]
    st.session_state.tick_count = 0
    st.session_state.prev_state = "Healthy"

# --- SIMULATION LOGIC ---
def get_next_telemetry(man_cpu, man_ram):
    t = st.session_state.tick_count
    mode = st.session_state.sim_mode
    
    # --- READ REAL HARDWARE STATS ---
    real_cpu = psutil.cpu_percent(interval=0.1)
    real_ram = psutil.virtual_memory().percent
    
    # Disk and Network approximations based on real byte activity
    disk_io = psutil.disk_io_counters()
    real_disk = min(99, 10 + (disk_io.read_bytes + disk_io.write_bytes) % 85) 
    
    net_io = psutil.net_io_counters()
    real_net = min(99, 5 + (net_io.bytes_sent + net_io.bytes_recv) % 90)
    
    # Default to real stats
    cpu = real_cpu
    ram = real_ram
    disk = real_disk
    net = real_net
    
    # Interactive Sliders override if you drag them
    if man_cpu > 0: cpu = man_cpu
    if man_ram > 0: ram = man_ram
    
    # Auto Simulations override if you click attack buttons
    if mode == 'CPU Spike': cpu = np.random.uniform(88, 98)
    elif mode == 'Memory Leak': ram = min(99, ram + (t * 0.5))
    elif mode == 'Disk Storm': disk = np.random.uniform(85, 99)
    elif mode == 'Silent Malware': cpu += 20; net += 45; ram += 15
    
    return np.clip(cpu, 1, 99), np.clip(ram, 1, 99), np.clip(disk, 1, 99), np.clip(net, 1, 99)

def process_tick(man_cpu, man_ram):
    cpu, ram, disk, net = get_next_telemetry(man_cpu, man_ram)
    st.session_state.tick_count += 1
    t_str = f"T-{st.session_state.tick_count}"
    h = st.session_state.history
    
    cpu_roc = cpu - (h['cpu'][-1] if len(h['cpu']) > 0 else cpu)
    ram_roc = ram - (h['ram'][-1] if len(h['ram']) > 0 else ram)
    cpu_std = np.std(h['cpu'][-5:] + [cpu]) if len(h['cpu']) >= 4 else abs(cpu_roc)
    ram_std = np.std(h['ram'][-5:] + [ram]) if len(h['ram']) >= 4 else abs(ram_roc)
    features = np.array([[cpu, ram, disk, net, cpu_roc, ram_roc, cpu_std, ram_std]])
    
    state = rf_model.predict(features)[0]
    is_anomaly = if_model.predict(features)[0] == -1
    anom_score = if_model.decision_function(features)[0]
    time_left = max(0, lr_model.predict([[cpu, cpu_roc, ram, ram_roc]])[0])
    
    # Logging & Alerts
    if is_anomaly and (len(st.session_state.alerts) == 0 or st.session_state.alerts[-1]['type'] != 'Anomaly'):
        st.session_state.alerts.insert(0, {"time": t_str, "type": "Anomaly", "msg": f"Silent anomaly detected! Score: {anom_score:.2f}"})
        st.session_state.logs.append(f"[ALERT] Anomaly detected. Score: {anom_score:.2f}")
    if state != st.session_state.prev_state:
        st.session_state.alerts.insert(0, {"time": t_str, "type": "State Change", "msg": f"System shifted to: {state}"})
        st.session_state.logs.append(f"[STATE] Shifted from {st.session_state.prev_state} to {state}")
        st.session_state.prev_state = state
    if time_left < 5.0:
         st.session_state.alerts.insert(0, {"time": t_str, "type": "Prediction", "msg": f"WARNING: Crash in {time_left:.1f} mins!"})
         st.session_state.logs.append(f"[CRITICAL] Exhaustion imminent: {time_left:.1f} mins.")

    st.session_state.alerts = st.session_state.alerts[:20]
    st.session_state.logs = st.session_state.logs[-50:] # Keep 50 log lines

    h['time'].append(t_str); h['cpu'].append(cpu); h['ram'].append(ram); h['disk_io'].append(disk); h['network'].append(net)
    h['anomaly_score'].append(-anom_score); h['state'].append(state); h['prediction'].append(time_left)
    for k in h:
        if len(h[k]) > 60: h[k] = h[k][-60:]
    return cpu, ram, disk, net, state, is_anomaly, time_left

# --- SIDEBAR LAYOUT ---
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")
    
    st.markdown("**Interactive Manual Override:**")
    st.caption("Drag sliders to manually stress the system.")
    man_cpu = st.slider("Force CPU %", 0, 100, 0, key="cpu_slider")
    man_ram = st.slider("Force RAM %", 0, 100, 0, key="ram_slider")
    
    st.markdown("---")
    st.markdown("**Automated Attack Simulation:**")
    if st.button("⚡ CPU Spike"): st.session_state.sim_mode = 'CPU Spike'
    if st.button("💧 Memory Leak"): st.session_state.sim_mode = 'Memory Leak'
    if st.button("💿 Disk Storm"): st.session_state.sim_mode = 'Disk Storm'
    if st.button("🦠 Silent Malware"): st.session_state.sim_mode = 'Silent Malware'
    if st.button("🔄 Reset System", use_container_width=True): 
        st.session_state.sim_mode = 'Normal'
        st.session_state.history = {'time': [], 'cpu': [], 'ram': [], 'disk_io': [], 'network': [], 'anomaly_score': [], 'state': [], 'prediction': []}
        st.session_state.logs.append("[SYSTEM] Reset to normal baseline.")
        
    st.markdown("---")
    auto_run = st.checkbox("Real-time Auto-Tick", value=True)
    st.markdown("<small style='color:#5a6a8a;'>SentinAI v2.0<br>System Health Engine</small>", unsafe_allow_html=True)
# --- PROCESS DATA ---
cpu, ram, disk, net, state, is_anomaly, time_left = process_tick(man_cpu, man_ram)

# --- MAIN UI LAYOUT ---
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("<h1 style='margin:0; font-size:2rem; font-weight:700;'>🛡️ SentinAI Engine</h1>", unsafe_allow_html=True)
with col_status:
    color = "#00ff88" if state=="Healthy" else "#ffaa00" if state=="Stressed" else "#ff3355"
    st.markdown(f"<div class='status-badge' style='background:{color}15; border:1px solid {color}40; color:{color}; float:right;'><div class='pulse-dot' style='background:{color};'></div> {state}</div>", unsafe_allow_html=True)

# --- TABS FOR INTERACTIVITY ---
tab1, tab2 = st.tabs(["🟢 Live Monitor", "🧠 AI Model Insights"])

# ==========================================
# TAB 1: LIVE MONITOR
# ==========================================
with tab1:
    # Metric Cards
    st.markdown("<div style='display:flex; gap:20px; margin-bottom:25px;'>", unsafe_allow_html=True)
    for label, val, color in [("CPU USAGE", cpu, "#00ccff"), ("MEMORY", ram, "#00ff88"), ("DISK I/O", disk, "#ffaa00"), ("NETWORK", net, "#e040fb")]:
        border_color = "#ff3355" if val > 80 else "#ffaa00" if val > 60 else color
        st.markdown(f"<div class='metric-card' style='flex:1; border-color: {border_color}30;'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{border_color};'>{val:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts Row
    col_chart1, col_chart2 = st.columns((2, 1))
    with col_chart1:
        st.markdown("<div class='chart-container'><b>System Telemetry</b>", unsafe_allow_html=True)
        fig_telem = go.Figure()
        h = st.session_state.history
        for col, name, color in [('cpu','CPU','#00ccff'), ('ram','RAM','#00ff88'), ('disk_io','Disk','#ffaa00'), ('network','Net','#e040fb')]:
            fig_telem.add_trace(go.Scatter(x=h['time'], y=h[col], mode='lines', name=name, line=dict(color=color, width=2)))
        fig_telem.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#8892a8', margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(range=[0, 100], gridcolor='#1a2744'))
        st.plotly_chart(fig_telem, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='chart-container'><b>Anomaly Score</b>", unsafe_allow_html=True)
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=h['time'], y=h['anomaly_score'], mode='lines', fill='tozeroy', line=dict(color='#e040fb', width=2), fillcolor='rgba(224,64,251,0.1)'))
        fig_anom.add_hline(y=0.6, line_dash="dash", line_color="#ff3355")
        fig_anom.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#8892a8', margin=dict(l=0, r=0, t=30, b=0), yaxis=dict(gridcolor='#1a2744'), showlegend=False)
        st.plotly_chart(fig_anom, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Predictions, Alerts, Data Table
    col_pred, col_alerts = st.columns((1, 1))
    
    with col_pred:
        st.markdown("<div class='chart-container'><b>AI Predictions</b>", unsafe_allow_html=True)
        st.markdown(f"**Classified State:** <span style='color:{color}'>{state}</span>", unsafe_allow_html=True)
        
        st.markdown("**Time to Exhaustion:**")
        pred_color = "#00ff88" if time_left > 15 else "#ffaa00" if time_left > 5 else "#ff3355"
        st.progress(min(1.0, time_left / 60.0))
        st.markdown(f"<span style='color:{pred_color}; font-weight:bold; font-size:1.2rem;'>{time_left:.1f} mins remaining</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_alerts:
        st.markdown("<div class='chart-container'><b>Alert History</b>", unsafe_allow_html=True)
        if not st.session_state.alerts: st.info("No alerts yet.")
        else:
            for alert in st.session_state.alerts[:5]:
                alert_color = "#ff3355" if "Anomaly" in alert['type'] else "#ffaa00" if "State" in alert['type'] else "#00ccff"
                st.markdown(f"<div style='background:{alert_color}10; border-left: 3px solid {alert_color}; padding:8px; margin-bottom:8px; border-radius:4px; font-size:0.85rem;'><b style='color:{alert_color};'>[{alert['type']}]</b> <span style='color:#5a6a8a; font-size:0.75rem;'>{alert['time']}</span><br>{alert['msg']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Live Data Table & Terminal Log
    col_table, col_term = st.columns((1, 1))
    with col_table:
        st.markdown("<div class='chart-container'><b>Live Data Table</b>", unsafe_allow_html=True)
        if len(h['time']) > 0:
            df_live = pd.DataFrame({
                'Time': h['time'][-10:],
                'CPU': h['cpu'][-10:],
                'RAM': h['ram'][-10:],
                'State': h['state'][-10:],
                'Crash In': [f"{x:.1f}m" for x in h['prediction'][-10:]]
            }).iloc[::-1]
            st.dataframe(df_live, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_term:
        st.markdown("<div class='chart-container'><b>System Log</b>", unsafe_allow_html=True)
        log_text = "<br>".join(st.session_state.logs[-15:])
        st.markdown(f"<div class='terminal-box'>{log_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: AI MODEL INSIGHTS
# ==========================================
with tab2:
    st.markdown("<div class='chart-container'><b>Random Forest: Feature Importance</b><br><span style='color:#5a6a8a; font-size:0.8rem;'>What the AI pays the most attention to when deciding if the system is Healthy, Stressed, or Critical.</span>", unsafe_allow_html=True)
    
    # Extract feature importances from the model
    importances = rf_model.feature_importances_
    feat_names = ['CPU %', 'RAM %', 'Disk I/O %', 'Network %', 'CPU Rate of Change', 'RAM Rate of Change', 'CPU Std Dev', 'RAM Std Dev']
    fig_imp = go.Figure(go.Bar(x=importances, y=feat_names, orientation='h', marker_color='#00ccff'))
    fig_imp.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#8892a8', margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(gridcolor='#1a2744'), xaxis=dict(gridcolor='#1a2744', title="Importance Score"))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("<div class='chart-container' style='text-align:center;'><b>Random Forest</b><br><span style='font-size:2rem; font-weight:bold; color:#00ff88;'>100</span><br><span style='color:#5a6a8a;'>Decision Trees</span></div>", unsafe_allow_html=True)
    with col_info2:
        st.markdown("<div class='chart-container' style='text-align:center;'><b>Isolation Forest</b><br><span style='font-size:2rem; font-weight:bold; color:#e040fb;'>100</span><br><span style='color:#5a6a8a;'>Isolation Trees</span></div>", unsafe_allow_html=True)
    with col_info3:
        st.markdown("<div class='chart-container' style='text-align:center;'><b>Linear Regression</b><br><span style='font-size:2rem; font-weight:bold; color:#00ccff;'>4</span><br><span style='color:#5a6a8a;'>Input Features</span></div>", unsafe_allow_html=True)

# --- AUTO RUN LOOP ---
if auto_run:
    time.sleep(1)
    st.rerun()