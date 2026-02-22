import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# adding path of Dataset 

DATASET_PATH = "real_cleaned_student_performance_dataset"  # <- your cleaned CSV path


# Loading trained model here

model = pickle.load(open("ORIGINAL_MODEL_features.pkl", "rb"))
scaler = pickle.load(open("ORIGINAL_SCALER_features.pkl", "rb"))

st.set_page_config(page_title="Student Grade Predictor", layout="wide")


# Page Title

st.title("🎓 Student Grade Prediction System")
st.markdown("Predict final grade (G3), Pass/Fail, and Confidence Score based on student info. ")
st.title("📌 Objective:")

st.markdown("➤ Predict the final grade (G3) from student background information.")
st.markdown("➤ Determine Pass or Fail status.")
st.markdown("➤ Generate a confidence score for prediction reliability.")


# Load original dataset

df = pd.read_csv(DATASET_PATH)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# Sidebar 

with st.sidebar.expander("📊 Show Dataset Analysis", expanded=False):
    
    st.write("Click to expand each chart below")

# Grade Distribution (G3)
    if "G3" in df.columns:
        with st.expander("📈 Grade Distribution (G3)"):
            fig1, ax1 = plt.subplots(figsize=(4,3))
            sns.histplot(df["G3"], kde=True, bins=20, ax=ax1, color='skyblue')
            ax1.set_xlabel("G3")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

# Average Grade vs Study Time
    if "studytime" in df.columns and "G3" in df.columns:
        with st.expander("📊 Average Grade vs Study Time"):
            avg_grade_studytime = df.groupby("studytime")["G3"].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.barplot(x="studytime", y="G3", data=avg_grade_studytime, palette="viridis", ax=ax2)
            ax2.set_xlabel("Study Time")
            ax2.set_ylabel("Average G3")
            st.pyplot(fig2)

# G1 vs G3 Scatter Plot
    if "G1" in df.columns and "G3" in df.columns:
        with st.expander("🔹 G1 vs G3"):
            fig3, ax3 = plt.subplots(figsize=(4,3))
            sns.scatterplot(x="G1", y="G3", data=df, hue="studytime", palette="coolwarm", ax=ax3)
            ax3.set_xlabel("G1 Grade")
            ax3.set_ylabel("G3 Grade")
            st.pyplot(fig3)

# Input sliders for prediction

st.subheader("📝 Enter Student Details")

G1 = st.slider("G1 Grade", min_value=0, max_value=20, value=10)
G2 = st.slider("G2 Grade", min_value=0, max_value=20, value=10)
studytime = st.slider("Study Time (1=Low, 4=High)", 1, 4, 2)
failures = st.slider("Past Failures", 0, 4, 0)
absences = st.slider("Number of Absences", 0, 100, 5)


# Prediction button

if st.button("Predict Grade"):

# Prepare input dataframe
    input_df = pd.DataFrame([[G1, G2, studytime, failures, absences]],
                            columns=["G1","G2","studytime","failures","absences"])

# Scale features
    input_scaled = scaler.transform(input_df)

# Predict grade
    pred_grade = model.predict(input_scaled)[0]

# Pass/Fail logic
    status = "PASS ✅" if pred_grade >= 10 else "FAIL ❌"

# Confidence score
    confidence = max(0, min(100, int(abs(pred_grade - 10) * 10)))

# Display results
    st.success(f"Predicted Final Grade: {pred_grade:.2f}")
    st.info(f"Status: {status}")
    st.metric("Confidence Score", f"{confidence}%")

# Visualization: Predicted vs Threshold
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(["Passing Threshold", "Predicted Grade"], [10, pred_grade], color=['red','green'])
    ax.set_ylim(0,20)
    ax.set_ylabel("Grade")
    ax.set_title("Predicted Grade vs Passing Threshold")
    st.pyplot(fig)

# Footer part of the website 

st.markdown("---")
st.markdown("💡 **Note:** Passing mark = 10. Confidence is based on distance from threshold.")
