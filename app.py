import streamlit as st
import joblib as jb
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="Hello Streamlit", page_icon=":wave:")
st.title("Welcome to HR Analytics App")
st.image(
    "https://c8.alamy.com/comp/2K9FCP7/people-analytics-and-hr-analytics-concept-collection-and-application-of-human-resources-data-to-improve-critical-talent-and-business-results-3d-il-2K9FCP7.jpg",
    width=700,
)

# Get input data from user
st.header("Input Employee Details")
job_satisfaction = st.slider("Job Satisfaction (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.15, step=0.1)
# last evalation from 0.0 to 1
last_performance_rating = st.slider(
    "Last Performance Rating (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.55, step=0.1
)
number_project = st.number_input("Number of Projects", min_value=1, max_value=20, value=7)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=150)
time_spend_company = st.number_input("Time Spent at Company (Years)", min_value=0, max_value=20, value=6)
# promotion last 5 years in 0 or 1 yes or not denoted by 1 or 0
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", options=["Yes", "No"])
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0
# low is field name with value true false
low = st.selectbox("Is Salary Low?", options=["Yes", "No"])
low = 0 if low == "Yes" else 1
medieum = st.selectbox("Is Salary Medium?", options=["Yes", "No"])
medieum = 0 if medieum == "Yes" else 1
#work accident in 1 or 0 yes or not denoted by 1 or 0
w0rk_accident = st.selectbox("Is there any Work Accident", options=["Yes", "No"])
work_accident = 0 if w0rk_accident == "Yes" else 1  

# Model path (relative to this file)
model_path = Path(__file__).resolve().parent / "finalized_model.pkl"

def try_load_model():
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        return jb.load(model_path)
    except Exception as e:
        st.error("Failed to load model.")
        st.exception(e)
        # Helpful hint for common Windows binary issues (pyarrow/scikit-learn)
        if isinstance(e, ImportError) or "pyarrow" in str(e).lower() or "dll" in str(e).lower():
            st.info(
                "If you see DLL import errors (pyarrow), install binary dependencies. "
                "Recommended (conda-forge):\n\n"
                "`conda install -n myenv -c conda-forge pyarrow scikit-learn numpy joblib -y`"
            )
        return None

model = None  # lazy-loaded

def make_prediction(model, input_array):
    try:
        pred = model.predict(input_array)
        return pred
    except Exception as e:
        st.error("Prediction failed:")
        st.exception(e)
        return None

# When user clicks predict, load model if needed and run prediction
if st.button("Predict Attrition"):
    if model is None:
        model = try_load_model()
    if model is None:
        st.warning("Model not available; cannot make prediction.")
    else:
        input_data = np.array(
            [
                [
                    job_satisfaction,
                    last_performance_rating,
                    number_project,
                    average_montly_hours,
                    time_spend_company,
                    work_accident,
                    promotion_last_5years,
                    low,
                    medieum
                ]
            ]
        )
        prediction = make_prediction(model, input_data)
        if prediction is None:
            # error already shown by make_prediction
            pass
        else:
            if prediction[0] == 1:
                st.error("The employee is likely to leave the company.")
            else:
                st.success("The employee is likely to stay with the company.")