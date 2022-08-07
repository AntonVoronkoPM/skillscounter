from pathlib import Path

import streamlit as st
from config import config
from skillscounter import main, utils

# Title
st.title("Skillscounter")

# ToC
st.markdown("🔢 [Data](#data)", unsafe_allow_html=True)
st.markdown("📊 [Performance](#performance)", unsafe_allow_html=True)
st.markdown("🚀 [Inference](#inference)", unsafe_allow_html=True)

# Sections
st.header("🔢 Data")
projects_fp = Path(config.DATA_DIR, "full_dataset.csv")
df = utils.load_frames(filepath=projects_fp)
st.text(f"Projects (count: {len(df)})")
st.write(df.head(5))

st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Precision")
st.write(performance["precision"])
st.text("Recall")
st.write(performance["recall"])

st.header("🚀 Inference")
text = st.text_input("Enter text:", "Docker")
run_id = st.text_input("Enter run ID:", open(Path("run_id.txt")).read())
prediction = main.predict(text=[text], run_id=run_id)
st.write(prediction)
