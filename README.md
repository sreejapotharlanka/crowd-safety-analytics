# 👥 Crowd Safety Analytics — Batch 8

> AI-powered crowd density estimation with heatmaps, safety alerts, and Groq AI analysis.

---

## 📋 Project Details

| Field | Value |
|-------|-------|
| Batch | 8 |
| Domain | Computer Vision |
| LLM/API | Mistral (HF API) / Groq |
| Python Modules | streamlit, groq, opencv-python |

---

## 🔁 How It Works

```
Upload Image/Video
      ↓
HOG Person Detection (OpenCV)
      ↓
Crowd Count + Density Calculation
      ↓
Heatmap Generation (OpenCV colormaps)
      ↓
Safety Level → SAFE / WARNING / CRITICAL
      ↓
AI Analysis (Groq + LLaMA 4 Scout)
      ↓
Output: Heatmap + Alerts + JSON Report
```

---

## 🚀 Setup & Run

### Step 1 — Clone or copy files
```bash
mkdir crowd_safety && cd crowd_safety
# copy app.py and requirements.txt here
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Activate (Linux/Mac):
source venv/bin/activate

# Activate (Windows):
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## 🔑 API Keys

### Groq API Key (Free)
1. Go to https://console.groq.com
2. Sign up for free
3. Create an API key
4. Paste it in the app sidebar

---

## 📦 Features

- **Image Upload** — Analyze JPG/PNG crowd images
- **Video Upload** — Extract & analyze frames from MP4/AVI
- **HOG Detection** — OpenCV's Histogram of Oriented Gradients person detector
- **Density Heatmap** — Gaussian blur + colormap overlay (JET, HOT, PLASMA, etc.)
- **Safety Alerts** — SAFE / WARNING / CRITICAL with configurable thresholds
- **AI Analysis** — Groq LLaMA 4 Scout provides natural language safety report
- **Export** — Download heatmap PNG, detection PNG, and JSON report
- **Demo Mode** — Built-in synthetic crowd for testing without uploads

---

## 📁 File Structure

```
crowd_safety_analytics/
├── app.py              ← Main Streamlit app
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| `ModuleNotFoundError: groq` | Run `pip install groq` |
| App not opening | Check if port 8501 is free |
| HOG detects 0 people | Image may be too small or low contrast — try a clearer crowd photo |
| Groq error | Check API key is correct and has credits |

---

## 📊 Density Thresholds

| Level | Density | Action |
|-------|---------|--------|
| 🟢 SAFE | < 0.5 p/m² | Normal monitoring |
| 🟡 WARNING | 0.5–3.0 p/m² | Increased monitoring |
| 🔴 CRITICAL | > 3.0 p/m² | Immediate intervention |

*All thresholds are adjustable in the sidebar.*
