import streamlit as st
import pandas as pd
import pickle
import base64

# Load Saved Files

model = pickle.load(open('cheating_pred_rmfrst.sav', 'rb'))
scaler = pickle.load(open('cheating_pred_scaler.sav', 'rb'))
encoder = pickle.load(open('cheating_pred_encoder.sav', 'rb'))


# Page Configuration

st.set_page_config(
    page_title="Online Exam Cheating Prediction",
    page_icon="🎓",
    layout="wide"
)
def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

bg_image = get_base64("ml_image.jpg")


# Custom CSS

st.markdown(f"""
<style>

/* Background Image */
[data-testid="stAppViewContainer"] {{
    background-image: 
        linear-gradient(
        rgba(0,0,0,0.45),
        rgba(0,0,0,0.45)
        ),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Make main content semi-transparent */
.main {{
    background-color: rgba(255,255,255,0.15);
}}

.block-container {{
    max-width: 1100px;
    padding-top: 2rem;
}}

/* Header Card */
.header-card {{
    background: rgba(30,136,229,0.90);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}}

/* Result Cards */
.result-success {{
    background-color: rgba(212,237,218,0.95);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}}

.result-danger {{
    background-color: rgba(248,215,218,0.95);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}}

/* Input Containers */
[data-testid="stVerticalBlock"] {{
    border-radius: 15px;
}}

</style>
""", unsafe_allow_html=True)


# Header

st.markdown("""
<div class="header-card">
    <h1>🎓 Online Exam Cheating Prediction System</h1>
    <h4>Machine Learning Based Student Behavior Analysis</h4>
</div>
""", unsafe_allow_html=True)


# Two Column Layout

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Exam Information")

    exam_platform = st.selectbox(
        "Exam Platform",
        encoder.classes_
    )

    time_taken = st.number_input(
        "Time Taken (Minutes)",
        min_value=0,
        value=60
    )

    tab_switch = st.number_input(
        "Tab Switch Count",
        min_value=0,
        value=0
    )

with col2:
    st.subheader("📊 Student Performance")

    score_percent = st.number_input(
        "Score Percent",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    previous_gpa = st.number_input(
        "Previous GPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0
    )

    answer_similarity = st.number_input(
        "Answer Similarity",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

st.divider()


# Behaviour Analysis

st.subheader("🧠 Behaviour Analysis")

eye_movement = st.slider(
    "Eye Movement Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

typing_speed = st.slider(
    "Typing Speed (WPM)",
    min_value=0.0,
    max_value=120.0,
    value=40.0
)

internet_usage = st.slider(
    "Internet Usage (MB)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0
)

st.write("")


# Prediction Button

if st.button("🔍 Predict Cheating", use_container_width=True):

    platform_encoded = encoder.transform([exam_platform])[0]

    input_data = pd.DataFrame({
        "Exam_Platform": [platform_encoded],
        "Time_Taken_Min": [time_taken],
        "Tab_Switch_Count": [tab_switch],
        "Eye_Movement_Score": [eye_movement],
        "Typing_Speed_WPM": [typing_speed],
        "Answer_Similarity": [answer_similarity],
        "Score_Percent": [score_percent],
        "Previous_GPA": [previous_gpa],
        "Internet_Usage_MB": [internet_usage]
    })

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)

    st.write("")
    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.markdown("""
        <div style="
            background: linear-gradient(90deg,#ff416c,#ff4b2b);
            color:white;
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;">
            ⚠️ Cheating Detected
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background: linear-gradient(90deg,#11998e,#38ef7d);
            color:white;
            padding:20px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;">
            ✅ No Cheating Detected
        </div>
        """, unsafe_allow_html=True)