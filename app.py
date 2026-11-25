import os
import joblib
import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog, local_binary_pattern

# ============================================================
# SMART WASTE CLASSIFICATION — MODERN STREAMLIT UI
# Algorithm: Random Forest | Features: HOG + LBP + HSV
# ============================================================

st.set_page_config(
    page_title="EcoSort | Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DESIGN -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #071b18 0%, #0b2923 48%, #102f29 100%);
    color: #f3f8f4;
}
.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #071713;
    border-right: 1px solid rgba(164, 220, 183, .15);
}
[data-testid="stSidebar"] * { color: #e8f4eb; }
.hero {
    background: linear-gradient(120deg, rgba(30,91,68,.92), rgba(13,52,44,.95));
    border: 1px solid rgba(166,226,188,.22);
    border-radius: 26px;
    padding: 34px 36px;
    margin-bottom: 24px;
    box-shadow: 0 18px 50px rgba(0,0,0,.20);
}
.eyebrow {
    color: #a9e6bd;
    text-transform: uppercase;
    letter-spacing: 2.4px;
    font-size: 11px;
    font-weight: 800;
}
.hero h1 {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.08;
    margin: 10px 0 12px 0;
    font-weight: 800;
}
.hero p {
    color: #d1e9d9;
    font-size: 16px;
    max-width: 720px;
    margin-bottom: 0;
}
.pill {
    display: inline-block;
    background: rgba(175,236,194,.12);
    border: 1px solid rgba(175,236,194,.25);
    color: #b9f0ca;
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 700;
    margin: 4px 5px 0 0;
}
.section-title {
    color: #f3fff5;
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
    margin: 12px 0 5px 0;
}
.muted { color: #a9c6b4; font-size: 13px; }
.panel {
    background: rgba(16,49,40,.82);
    border: 1px solid rgba(177,225,192,.16);
    border-radius: 20px;
    padding: 22px;
    min-height: 130px;
}
.metric-label {
    color: #a9c6b4;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}
.metric-value {
    color: #f4fff6;
    font-family: 'Manrope', sans-serif;
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}
.result-card {
    background: linear-gradient(135deg, #174c39, #123a31);
    border: 1px solid rgba(170,235,188,.28);
    border-radius: 22px;
    padding: 24px;
    margin-top: 12px;
}
.result-name {
    color: #c2f5ce;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 800;
}
.result-class {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin: 4px 0;
}
div.stButton > button {
    background: linear-gradient(90deg, #8be0a7, #b4f0bd);
    color: #08251a;
    border: 0;
    border-radius: 12px;
    padding: 0.72rem 1.4rem;
    font-weight: 800;
    min-height: 48px;
    width: 100%;
}
div.stButton > button:hover {
    background: #d0f8d8;
    color: #08251a;
    border: 0;
}
[data-testid="stFileUploader"] {
    background: rgba(13,43,35,.75);
    border: 1px dashed rgba(173,229,190,.42);
    border-radius: 18px;
    padding: 14px;
}
[data-testid="stFileUploader"] label { color: #eaf7ed; }
.stAlert { border-radius: 14px; }
hr { border-color: rgba(177,225,192,.16); }
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING ------------------------
import os
import requests
import joblib

MODEL_PATH = "waste_random_forest.pkl"
MODEL_URL = "https://github.com/msdalal13/waste_classification_ml/releases/download/v1.0/waste_random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

model = joblib.load(MODEL_PATH)

# -------------------- FEATURE EXTRACTION -------------------
# Must remain identical to the improved Random Forest notebook.
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
        color_features.extend(cv2.normalize(hist, hist).flatten())

    return np.concatenate([
        hog_features.astype(np.float32),
        lbp_hist,
        np.asarray(color_features, dtype=np.float32)
    ])

# -------------------- SIDEBAR ------------------------------
with st.sidebar:
    st.markdown("## ♻️ EcoSort")
    st.caption("SMART WASTE INTELLIGENCE")
    st.markdown("---")
    st.markdown("### About the project")
    st.write(
        "An image-based waste classification prototype using "
        "a Random Forest machine-learning model."
    )
    st.markdown("### Model pipeline")
    st.markdown("""
    - Image preprocessing
    - HOG shape features
    - LBP texture features
    - HSV colour histograms
    - Random Forest prediction
    """)
    st.markdown("---")
    st.markdown("**Developed by**")
    st.write("Ankush Patil")
    st.write("Mihir Dalal")
    st.caption("Project demonstration • ENTC")

# -------------------- HERO ---------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered recycling assistant</div>
  <h1>Give waste a<br>smarter destination.</h1>
  <p>Upload a photo of a waste item and let the Random Forest model
  identify its likely material category in seconds.</p>
  <div style="margin-top:18px">
    <span class="pill">♻️ 6 waste categories</span>
    <span class="pill">🌿 HOG + LBP + HSV</span>
    <span class="pill">⚙️ Random Forest</span>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------- MODEL STATUS -------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load the model file: {exc}")
    st.stop()

if model is None:
    st.error(
        "Model file not found. Place **waste_random_forest.pkl** "
        "in the same folder as app.py."
    )
    st.stop()

# top metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        '<div class="panel"><div class="metric-label">Classifier</div>'
        '<div class="metric-value">Random Forest</div></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        '<div class="panel"><div class="metric-label">Image size</div>'
        '<div class="metric-value">128 × 128</div></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="panel"><div class="metric-label">Model features</div>'
        f'<div class="metric-value">{getattr(model, "n_features_in_", "N/A")}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# -------------------- MAIN WORKSPACE -----------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-title">01 · Upload an image</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Choose a clear photo of one waste item.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    image = None
    if uploaded_file is not None:
        raw = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("This image could not be opened. Please try another file.")
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="Your uploaded image", use_container_width=True)
    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:42px 20px">
          <div style="font-size:52px">📷</div>
          <div style="font-size:17px;font-weight:800;color:#eaf7ed;margin-top:10px">
            Your image preview will appear here
          </div>
          <div class="muted" style="margin-top:6px">
            Tip: use a single object with good lighting and a simple background.
          </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">02 · Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Run the trained Random Forest model.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div style="font-size:18px;font-weight:800;color:#eaf7ed">What happens next?</div>
      <p class="muted" style="margin:10px 0 0 0">
        The image is resized, visual features are extracted, and the trained
        Random Forest model predicts one of six categories.
      </p>
      <div style="margin-top:16px">
        <span class="pill">Cardboard</span><span class="pill">Glass</span>
        <span class="pill">Metal</span><span class="pill">Paper</span>
        <span class="pill">Plastic</span><span class="pill">Trash</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    classify_clicked = st.button(
        "✨  Classify my waste",
        type="primary",
        disabled=(image is None)
    )

    if image is None:
        st.caption("Upload an image to activate classification.")

# -------------------- PREDICTION ---------------------------
if uploaded_file is not None and image is not None and classify_clicked:
    with st.spinner("Analysing image with Random Forest..."):
        features = extract_features(image).reshape(1, -1)
        expected_features = getattr(model, "n_features_in_", None)

        if expected_features is not None and features.shape[1] != expected_features:
            st.error(
                f"Feature mismatch: this app generated {features.shape[1]} features, "
                f"but the model expects {expected_features}. Use the matching model "
                f"and feature extractor from the supplied training notebook."
            )
            st.stop()

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        top_score = float(np.max(probabilities)) * 100

    descriptions = {
        "cardboard": ("📦", "Cardboard", "Keep dry and separate for cardboard recycling where accepted."),
        "glass": ("🥛", "Glass", "Separate glass items and follow local glass collection guidance."),
        "metal": ("🥫", "Metal", "Metal containers can often be recovered through metal recycling."),
        "paper": ("📄", "Paper", "Keep paper clean and dry for paper recycling where accepted."),
        "plastic": ("🧴", "Plastic", "Check the plastic type and local recycling rules before disposal."),
        "trash": ("🗑️", "General Trash", "Check local disposal guidance for non-recyclable mixed waste.")
    }
    icon, pretty_name, advice = descriptions.get(
        str(prediction).lower(),
        ("♻️", str(prediction).title(), "Check local disposal guidance.")
    )

    st.markdown("---")
    st.markdown('<div class="section-title">03 · Your result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
      <div class="result-name">Predicted waste category</div>
      <div class="result-class">{icon} {pretty_name}</div>
      <div style="color:#c3e6ce;font-size:14px">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Highest class-vote share", f"{top_score:.1f}%")
    with c2:
        st.metric("Categories considered", len(model.classes_))

    st.caption(
        "The percentage is the proportion of Random Forest tree votes for the top class. "
        "It is not a calibrated probability or a guarantee of correctness."
    )

    st.markdown('<div class="section-title">Class-vote breakdown</div>', unsafe_allow_html=True)
    chart_data = {
        str(label).title(): float(prob) * 100
        for label, prob in zip(model.classes_, probabilities)
    }
    st.bar_chart(chart_data, y_label="Tree vote share (%)")

    if top_score < 40:
        st.warning(
            "The model's votes are spread across categories. Treat this result as uncertain "
            "and try a clearer image with one object, good lighting, and a plain background."
        )

# -------------------- FOOTER -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8eae9a;font-size:12px'>"
    "ECOSORT · SMART WASTE CLASSIFICATION SYSTEM<br>"
    "Random Forest • HOG + LBP + HSV • Ankush Patil & Mihir Dalal"
    "</div>",
    unsafe_allow_html=True
)
