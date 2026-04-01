import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import time
import os
from groq import Groq

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd Safety Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0; font-size: 1rem; }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2.4rem; font-weight: 700; }
    .metric-card .label { color: #a0aec0; font-size: 0.85rem; margin-top: 4px; }

    .alert-critical { background:#2d1515; border:1px solid #e53e3e; border-radius:8px; padding:1rem; margin:0.5rem 0; }
    .alert-warning  { background:#2d2515; border:1px solid #dd6b20; border-radius:8px; padding:1rem; margin:0.5rem 0; }
    .alert-safe     { background:#152d15; border:1px solid #38a169; border-radius:8px; padding:1rem; margin:0.5rem 0; }

    .section-title {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem 1.2rem;
    }
    .stButton>button:hover { background: #c53030; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👥 Crowd Safety Analytics</h1>
    <p>AI-powered crowd density estimation · Heatmap generation · Real-time safety alerts</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password",
                                  help="Get your free key at console.groq.com")

    st.markdown("---")
    st.markdown("### 📊 Density Thresholds")
    safe_threshold     = st.slider("Safe (persons/m²)",     0.1, 1.0, 0.5, 0.1)
    warning_threshold  = st.slider("Warning (persons/m²)",  0.5, 3.0, 1.5, 0.1)
    critical_threshold = st.slider("Critical (persons/m²)", 1.0, 6.0, 3.0, 0.1)

    st.markdown("---")
    st.markdown("### 🎨 Heatmap Settings")
    colormap_choice = st.selectbox("Colormap",
        ["JET", "HOT", "COOL", "INFERNO", "PLASMA", "VIRIDIS"])
    heatmap_opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.6, 0.05)
    blur_radius     = st.slider("Blur Radius",      5, 51, 21, 2)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("**Batch 8** · Computer Vision\n\nMistral (HF API) + Groq\n\nModules: streamlit, groq, opencv")

# ─── Utilities ───────────────────────────────────────────────────────────────

COLORMAP_MAP = {
    "JET": cv2.COLORMAP_JET, "HOT": cv2.COLORMAP_HOT,
    "COOL": cv2.COLORMAP_COOL, "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA, "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def image_to_cv2(uploaded_file) -> np.ndarray:
    pil = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def estimate_crowd_count_hog(img_bgr: np.ndarray) -> dict:
    """Improved crowd estimation using face detection + skin region + edge analysis."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]

    # ── Method 1: Face detection (Haar Cascade) ──
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3,
        minSize=(20, 20), maxSize=(200, 200)
    )

    detections = []
    if len(faces):
        for (x, y, fw, fh) in faces:
            detections.append({
                "x": int(x), "y": int(y),
                "w": int(fw), "h": int(fh),
                "confidence": 0.85
            })

    # ── Method 2: Skin region detection (HSV) ──
    if len(detections) < 3:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_mask  = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 15000:
                x2, y2, cw, ch = cv2.boundingRect(cnt)
                detections.append({
                    "x": int(x2), "y": int(y2),
                    "w": int(cw), "h": int(ch),
                    "confidence": round(min(area / 5000, 0.95), 2)
                })

    count = len(detections)

    # ── Method 3: Edge density fallback ──
    if count < 5:
        edges        = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        estimated    = int(edge_density * 300)
        count        = max(count, estimated)

    area_m2 = max(1, (h * w) / (100 * 100))
    density  = count / area_m2

    return {"count": count, "density": density, "detections": detections}

def generate_heatmap(img_bgr: np.ndarray, detections: list,
                     blur_r: int, colormap: int, opacity: float):
    h, w = img_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    if detections:
        for d in detections:
            cx = d["x"] + d["w"] // 2
            cy = d["y"] + d["h"] // 2
            cv2.circle(heat, (cx, cy), max(d["w"], d["h"]) // 2, d["confidence"], -1)
    else:
        # Use edge map as fallback heatmap base
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        heat = cv2.Canny(gray, 50, 150).astype(np.float32)

    heat = cv2.GaussianBlur(heat, (blur_r | 1, blur_r | 1), 0)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = heat.astype(np.uint8)

    colored = cv2.applyColorMap(heat, colormap)
    overlay = cv2.addWeighted(img_bgr, 1 - opacity, colored, opacity, 0)
    return overlay, colored

def draw_bboxes(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections:
        conf  = d["confidence"]
        color = (0, 255, 0) if conf > 0.7 else (0, 200, 255)
        cv2.rectangle(out, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]), color, 2)
        cv2.putText(out, f"{conf:.2f}", (d["x"], d["y"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def get_safety_level(density: float, safe_t: float, warn_t: float, crit_t: float):
    if density >= crit_t:
        return "CRITICAL", "#e53e3e", "🔴"
    elif density >= warn_t:
        return "WARNING", "#dd6b20", "🟡"
    else:
        return "SAFE", "#38a169", "🟢"

def analyze_with_groq(api_key: str, image_b64: str, count: int,
                      density: float, safety_level: str) -> str:
    """Use Groq LLaMA 4 Scout to analyze the crowd image."""
    if not api_key:
        return "_No Groq API key provided. Add it in the sidebar for AI analysis._"
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are a crowd safety expert analyzing CCTV footage.

Crowd statistics detected:
- Estimated person count: {count}
- Crowd density: {density:.2f} persons/m²
- Safety level: {safety_level}

Analyze this crowd image and provide:
1. Brief description of what you observe
2. Key safety concerns (if any)
3. Specific recommendations for crowd management
4. Estimated risk level explanation

Keep it concise and actionable (under 200 words)."""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"_AI analysis error: {str(e)}_"

def extract_video_frame(video_file) -> np.ndarray:
    """Extract middle frame from uploaded video."""
    tmp_path = "/tmp/crowd_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_file.read())
    cap   = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ─── Main UI ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "📖 How It Works"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_up, col_info = st.columns([2, 1])

    with col_up:
        st.markdown('<div class="section-title">Upload Crowd Image</div>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader(
            "Drop a crowd image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="img_upload"
        )
        use_demo = st.checkbox("Use demo image (synthetic crowd)")

    with col_info:
        st.markdown('<div class="section-title">Quick Guide</div>', unsafe_allow_html=True)
        st.markdown("""
**Steps:**
1. Upload a crowd photo
2. System detects people (Face + Skin + Edge)
3. Heatmap is generated
4. AI analyzes safety
5. Alerts are shown
        """)

    if uploaded_img or use_demo:
        with st.spinner("🔍 Analyzing crowd..."):
            if use_demo:
                demo = np.ones((480, 640, 3), dtype=np.uint8) * 50
                for _ in range(40):
                    x = np.random.randint(50, 590)
                    y = np.random.randint(50, 430)
                    cv2.ellipse(demo, (x, y), (15, 30), 0, 0, 360, (
                        np.random.randint(80, 180),
                        np.random.randint(80, 180),
                        np.random.randint(80, 180)
                    ), -1)
                img_bgr = demo
            else:
                img_bgr = image_to_cv2(uploaded_img)

            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_b64 = pil_to_base64(img_pil)

            result  = estimate_crowd_count_hog(img_bgr)
            count   = result["count"]
            density = result["density"]
            dets    = result["detections"]

            safety_level, safety_color, safety_icon = get_safety_level(
                density, safe_threshold, warning_threshold, critical_threshold
            )

            cm = COLORMAP_MAP[colormap_choice]
            heatmap_overlay, heatmap_pure = generate_heatmap(
                img_bgr, dets, blur_radius, cm, heatmap_opacity
            )
            bbox_img = draw_bboxes(img_bgr, dets)

        # ── Metrics ──
        st.markdown("### 📊 Detection Results")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#e94560">{count}</div>
                <div class="label">Persons Detected</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#4299e1">{density:.2f}</div>
                <div class="label">Density (p/m²)</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:{safety_color}">{safety_icon}</div>
                <div class="label">{safety_level}</div></div>""", unsafe_allow_html=True)
        with m4:
            avg_conf = np.mean([d["confidence"] for d in dets]) if dets else 0
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#68d391">{avg_conf:.2f}</div>
                <div class="label">Avg Confidence</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Images ──
        st.markdown("### 🖼️ Visual Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("📷 Original Image")
            st.image(img_pil, use_container_width=True)
        with c2:
            st.caption("🔲 Person Detection")
            st.image(Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)), use_container_width=True)
        with c3:
            st.caption("🌡️ Density Heatmap Overlay")
            st.image(Image.fromarray(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)), use_container_width=True)

        st.markdown("---")

        # ── Alerts ──
        st.markdown("### 🚨 Safety Alerts")
        if safety_level == "CRITICAL":
            st.markdown(f"""<div class="alert-critical">
                🔴 <strong>CRITICAL ALERT</strong> — Density {density:.2f} p/m² exceeds critical threshold ({critical_threshold} p/m²)<br>
                ⚠️ Immediate crowd dispersal required. Alert emergency personnel. Close entry points.
                </div>""", unsafe_allow_html=True)
        elif safety_level == "WARNING":
            st.markdown(f"""<div class="alert-warning">
                🟡 <strong>WARNING</strong> — Density {density:.2f} p/m² approaching critical levels<br>
                ⚠️ Monitor closely. Consider limiting entry. Deploy crowd management staff.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-safe">
                🟢 <strong>SAFE</strong> — Density {density:.2f} p/m² is within safe limits<br>
                ✅ Crowd levels are normal. Continue standard monitoring.
                </div>""", unsafe_allow_html=True)

        if count > 20:
            st.markdown("""<div class="alert-warning">
                🟡 <strong>HIGH OCCUPANCY</strong> — Over 20 persons detected in frame.<br>
                Consider activating secondary monitoring zones.
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── AI Analysis ──
        st.markdown("### 🤖 AI Safety Analysis (Groq)")
        if groq_api_key:
            with st.spinner("🧠 Getting AI analysis..."):
                ai_analysis = analyze_with_groq(
                    groq_api_key, img_b64, count, density, safety_level
                )
            st.info(ai_analysis)
        else:
            st.warning("🔑 Add your Groq API key in the sidebar to enable AI analysis.")

        st.markdown("---")

        # ── Downloads ──
        st.markdown("### 💾 Export Results")
        d1, d2, d3 = st.columns(3)

        hm_bytes = io.BytesIO()
        Image.fromarray(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)).save(hm_bytes, format="PNG")
        with d1:
            st.download_button("⬇️ Download Heatmap", hm_bytes.getvalue(),
                               "heatmap_overlay.png", "image/png")

        bbox_bytes = io.BytesIO()
        Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)).save(bbox_bytes, format="PNG")
        with d2:
            st.download_button("⬇️ Download Detections", bbox_bytes.getvalue(),
                               "crowd_detections.png", "image/png")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "crowd_count": count,
            "density_per_m2": round(density, 3),
            "safety_level": safety_level,
            "avg_confidence": round(avg_conf, 3),
            "detections": dets,
            "thresholds": {
                "safe": safe_threshold,
                "warning": warning_threshold,
                "critical": critical_threshold
            }
        }
        with d3:
            st.download_button("⬇️ Download JSON Report",
                               json.dumps(report, indent=2),
                               "crowd_report.json", "application/json")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Upload Crowd Video</div>', unsafe_allow_html=True)
    st.info("📹 The system extracts the middle frame of the video for analysis.")

    uploaded_vid = st.file_uploader(
        "Drop a video file (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_vid:
        with st.spinner("🎬 Extracting and analyzing video frame..."):
            frame = extract_video_frame(uploaded_vid)

        if frame is not None:
            st.success("✅ Frame extracted successfully!")
            result_v  = estimate_crowd_count_hog(frame)
            count_v   = result_v["count"]
            density_v = result_v["density"]
            dets_v    = result_v["detections"]
            safety_v, color_v, icon_v = get_safety_level(
                density_v, safe_threshold, warning_threshold, critical_threshold
            )
            cm = COLORMAP_MAP[colormap_choice]
            hm_overlay_v, _ = generate_heatmap(frame, dets_v, blur_radius, cm, heatmap_opacity)
            bbox_v = draw_bboxes(frame, dets_v)

            mv1, mv2, mv3 = st.columns(3)
            with mv1:
                st.metric("Persons Detected", count_v)
            with mv2:
                st.metric("Density (p/m²)", f"{density_v:.2f}")
            with mv3:
                st.metric("Safety Level", f"{icon_v} {safety_v}")

            cv1, cv2_col, cv3 = st.columns(3)
            with cv1:
                st.caption("Extracted Frame")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            with cv2_col:
                st.caption("Person Detections")
                st.image(cv2.cvtColor(bbox_v, cv2.COLOR_BGR2RGB), use_container_width=True)
            with cv3:
                st.caption("Heatmap Overlay")
                st.image(cv2.cvtColor(hm_overlay_v, cv2.COLOR_BGR2RGB), use_container_width=True)

            if safety_v == "CRITICAL":
                st.error(f"🔴 CRITICAL: Density {density_v:.2f} p/m² — Immediate action required!")
            elif safety_v == "WARNING":
                st.warning(f"🟡 WARNING: Density {density_v:.2f} p/m² — Monitor closely.")
            else:
                st.success(f"🟢 SAFE: Density {density_v:.2f} p/m² — All clear.")
        else:
            st.error("❌ Could not extract frame from video. Try a different file.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## How Crowd Safety Analytics Works")
    st.markdown("""
### 🔁 Project Pipeline

```
Upload Image/Video
        ↓
Frame Extraction (for video)
        ↓
Face Detection (Haar Cascade)
  → If faces < 3: Skin Region Detection (HSV)
  → If still low: Edge Density Estimation
        ↓
Crowd Count + Density Calculation
  → persons per m²
        ↓
Heatmap Generation (OpenCV)
  → Gaussian blur + colormap overlay
        ↓
Safety Level Assessment
  → SAFE / WARNING / CRITICAL
        ↓
AI Analysis (Groq + LLaMA 4 Scout)
  → Natural language safety report
        ↓
Output: Heatmap + Alerts + Count + JSON Report
```

---

### 🧠 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | **Streamlit** | Web interface |
| Face Detection | **OpenCV Haar Cascade** | Detect faces in crowds |
| Skin Detection | **OpenCV HSV** | Detect people via skin regions |
| Edge Estimation | **OpenCV Canny** | Estimate density from edges |
| Heatmap | **OpenCV + NumPy** | Visualize crowd density |
| AI Analysis | **Groq (LLaMA 4 Scout)** | Natural language safety insights |

---

### 📐 Density Calculation

```
Crowd Density = Person Count / Area (m²)

Area = (height_px × width_px) / (100 × 100)

Safety Thresholds (adjustable in sidebar):
  🟢 SAFE     < 0.5 persons/m²
  🟡 WARNING  0.5 – 3.0 persons/m²
  🔴 CRITICAL > 3.0 persons/m²
```

---

### 🚀 Setup Instructions

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\\Scripts\\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

**Get a free Groq API key:** https://console.groq.com
    """)