import streamlit as st
import pandas as pd

# --- PAGE SETUP ---
st.set_page_config(page_title="Redrob AI Recruiter", page_icon="🤖", layout="wide")

# --- HEADER ---
st.title("🤖 Redrob AI Talent Intelligence")
st.markdown("### Smart Discovery & Semantic Ranking Engine")
st.markdown("This system filters out inactive candidates and uses `Sentence-Transformers` to semantically match the best talent to the Job Description.")
st.divider()

# --- LOAD DATA ---
try:
    df = pd.read_csv("ranked_shortlist.csv")
    
    # --- METRICS ROW ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates Scanned", "100,000")
    col2.metric("Candidates Surviving Smart Filter", "64,511")
    col3.metric("Top Match Score", f"{df.iloc[0]['Match_Score']}%")
    
    st.divider()
    
    # --- DATA TABLE ---
    st.subheader("🏆 Top Ranked Candidates (With Explainability)")
    
    st.data_editor(
        df,
        column_config={
            "Match_Score": st.column_config.ProgressColumn(
                "Match Quality",
                help="Semantic vector distance between resume and JD",
                format="%f%%",
                min_value=0,
                max_value=100,
            ),
            "Why_They_Match": st.column_config.TextColumn(
                "AI Reasoning",
                width="large"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
except FileNotFoundError:
    st.error("Could not find 'ranked_shortlist.csv'. Please run the analyzer.py script first!") 