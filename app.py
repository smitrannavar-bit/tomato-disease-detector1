import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time

st.set_page_config(page_title="Tomato Crop Health Detector", layout="wide")
st.title("🍅 AI Tomato Crop Health Detector")
st.write("Upload a clear picture of a tomato plant leaf to scan for diseases.")

LOG_FILE = "prediction_logs.csv"

def log_prediction(filename, predicted_class, confidence, execution_time):
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Predicted Class": predicted_class,
        "Confidence (%)": round(float(confidence), 2),
        "Execution Time (s)": round(float(execution_time), 4),
    }
    new_df = pd.DataFrame([new_row])
    if not os.path.exists(LOG_FILE):
        new_df.to_csv(LOG_FILE, index=False)
    else:
        new_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("tomato_disease_model.keras")

model = None
try:
    model = load_my_model()
    st.sidebar.success("🤖 AI model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model: {e}")

CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight", "Yellow Leaf Curl Virus"]

TREATMENT_GUIDE = {
    "Early Blight": "⚠️ **Treatment Plan:** Prune affected lower leaves to improve airflow. Apply copper-based fungicides or organic alternatives. Avoid overhead watering.",
    "Late Blight": "🚨 **Critical Alert:** Destructive disease! Remove and destroy highly infected plants. Apply a suitable fungicide such as chlorothalonil. Keep foliage dry.",
    "Healthy": "✅ **Excellent:** The leaf looks healthy. Continue regular watering, fertilization and monitoring.",
    "Yellow Leaf Curl Virus": "🪰 **Pest Control Required:** Spread by whiteflies. Use yellow sticky traps, insecticidal soap or neem oil, and remove infected plants.",
}

# ---------------- Sidebar KPIs ----------------
st.sidebar.title("📊 Monitoring KPIs")
if os.path.exists(LOG_FILE):
    try:
        df_logs = pd.read_csv(LOG_FILE)
        if not df_logs.empty:
            st.sidebar.metric("Total Scans", len(df_logs))
            st.sidebar.metric("Avg Confidence", f"{df_logs['Confidence (%)'].mean():.2f}%")
            st.sidebar.metric("Avg Latency", f"{df_logs['Execution Time (s)'].mean():.3f}s")
            st.sidebar.write("**Class Distribution**")
            st.sidebar.bar_chart(df_logs["Predicted Class"].value_counts())
    except Exception as e:
        st.sidebar.error(f"Error reading logs: {e}")
else:
    st.sidebar.info("No predictions recorded yet.")

# ---------------- Main flow ----------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    start_time = time.time()
    image = Image.open(uploaded_file).convert("RGB")

    # Image on the left, diagnosis + recommendations on the right
    # (on phones Streamlit stacks these automatically)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with col2:
        with st.spinner("🔬 Analyzing image..."):
            try:
                img_array = np.array(image.resize((224, 224)))
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                probabilities = model.predict(img_array, verbose=0)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[idx]
        confidence = 100 * float(probabilities[idx])
        execution_time = time.time() - start_time

        # Log only once per uploaded file (Streamlit reruns the script on interaction)
        file_key = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state.get("last_logged") != file_key:
            try:
                log_prediction(uploaded_file.name, predicted_class, confidence, execution_time)
                st.session_state["last_logged"] = file_key
            except Exception:
                pass  # logging must never block the result

        st.subheader(f"Diagnosis: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        if confidence < 70:
            st.warning("Low confidence. Try a clearer, closer photo of a single leaf.")
        st.write(f"⏱️ Latency: {execution_time:.3f} seconds")
        st.markdown("---")
        st.subheader("📋 Recommendations")
        st.write(TREATMENT_GUIDE.get(predicted_class, "Consult local agricultural guidance."))
