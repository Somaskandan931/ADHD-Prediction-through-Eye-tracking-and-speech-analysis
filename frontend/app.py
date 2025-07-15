# streamlit_app.py (Updated: User reads passage → prediction with SHAP)
import streamlit as st
import requests
import base64
import time

st.set_page_config(page_title="ADHD Prediction", layout="centered")
st.title("🧠 ADHD Prediction Through Reading")

PASSAGE = """
When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. 
The rainbow is a division of white light into many beautiful colors. 
These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. 
There is, according to legend, a boiling pot of gold at one end. 
People look, but no one ever finds it. 
When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.
"""

st.markdown("""
### Instructions:
1. Read the passage below **aloud** while your eye and voice data are captured.
2. Once done, click **Submit for Prediction**.

_Note: This is a simulated interface. Eye and speech inputs will be mocked as part of synthetic testing._
""")

# Display the passage
st.text_area("📖 Please read this passage aloud:", PASSAGE, height=200, disabled=True)

# Simulate "Start Recording"
if "recording" not in st.session_state:
    st.session_state.recording = False

if not st.session_state.recording:
    if st.button("🎙️ Start Recording"):
        st.session_state.recording = True
        st.session_state.start_time = time.time()
        st.rerun()
else:
    elapsed = time.time() - st.session_state.start_time
    st.info(f"Recording... {int(elapsed)} seconds")
    if elapsed > 10:
        st.session_state.recording = False
        st.success("✅ Recording complete. Now analyzing...")

        # === Replace this mock data with real sensors integration ===
        features = {
            "fixation_duration": 340,
            "saccadic_amplitude": 6.2,
            "saccadic_velocity": 360,
            "speech_rate": 118,
            "pitch_variability": 33.5,
            "jitter": 1.1,
            "shimmer": 4.2,
            "pause_count": 6,
            "avg_pause_duration": 0.5
        }

        with st.spinner("🔎 Predicting ADHD and generating explanations..."):
            try:
                response = requests.post("http://localhost:5000/predict", json=features)
                result = response.json()

                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.subheader("🧠 Prediction Result")
                    st.write("Prediction:", "**ADHD**" if result['prediction'] == 1 else "**Non-ADHD**")
                    st.write(f"Probability: {result['probability']*100:.2f}%")

                    st.subheader("📊 SHAP Explanation - Summary Plot")
                    st.image(base64.b64decode(result['shap_summary_img']), use_column_width=True)

                    st.subheader("📈 SHAP Force Plot")
                    st.image(base64.b64decode(result['shap_force_img']), use_column_width=True)
            except Exception as e:
                st.error(f"Backend Error: {e}")
