# 🤖 Redrob AI Talent Intelligence (INDIA.RUNS Challenge)

## 📌 Project Overview
This project is an end-to-end AI Recruiter Proof of Concept (PoC) built for the Redrob Data & AI Challenge. It moves beyond simple keyword matching to deeply understand candidate context, automatically filtering out poor fits and ranking the remaining talent using semantic vector search.

## 🏗️ System Architecture
The pipeline is divided into three core engines:

### 1. The Smart Filter (Anti-Hallucination & JD Traps)
The system strictly enforces the disqualifiers laid out in the Redrob Job Description:
* **Activity Check:** Drops candidates who have been inactive for over 6 months.
* **The Consulting Trap:** Automatically disqualifies candidates whose *entire* career history consists only of IT consulting/services firms (TCS, Infosys, Wipro, etc.) without product company experience.

### 2. The Semantic Brain (`Sentence-Transformers`)
Instead of Boolean keyword searches, the engine maps the Job Description and candidate profiles into a 384-dimensional mathematical space using `all-MiniLM-L6-v2`. It calculates the Cosine Similarity to find candidates whose experience structurally matches the JD, even if they use different terminology.

### 3. Explainability Layer & Streamlit UI
To solve the "black box" AI problem, the system generates a deterministic, one-sentence explanation for *why* a candidate received their specific match score based on their tenure, title, and top skills. This is visualized in a reactive Streamlit dashboard.

## 🚀 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend engine: `python analyzer.py`
4. Launch the dashboard: `streamlit run app.py`
