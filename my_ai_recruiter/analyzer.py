import os
import docx
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. DATA INGESTION
# ==========================================

def read_job_description(file_path):
    if not os.path.exists(file_path):
        return f"Error: Could not find JD file at {file_path}"
    
    doc = docx.Document(file_path)
    text_lines = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(text_lines)

def read_candidates(file_path):
    if not os.path.exists(file_path):
        return f"Error: Could not find Candidate file at {file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            return [json.loads(line) for line in file if line.strip()]

# ==========================================
# 2. THE SMART FILTER
# ==========================================

def filter_candidates(candidates_list):
    print("\n--- 2. RUNNING SMART FILTER ---")
    valid_candidates = []
    consulting_firms = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]
    cutoff_date = datetime(2025, 12, 30)

    for candidate in candidates_list:
        is_disqualified = False
        
        # Activity Check
        signals = candidate.get("redrob_signals", {})
        last_active_str = signals.get("last_active_date", "2020-01-01")
        try:
            if datetime.strptime(last_active_str, "%Y-%m-%d") < cutoff_date:
                is_disqualified = True
        except ValueError:
            pass
            
        # Consulting Firm Trap
        career_history = candidate.get("career_history", [])
        has_product_experience = False
        if isinstance(career_history, list):
            for job in career_history:
                if job.get("company", "") not in consulting_firms:
                    has_product_experience = True
                    break 
                
        if len(career_history) > 0 and not has_product_experience:
            is_disqualified = True

        if not is_disqualified:
            valid_candidates.append(candidate)

    print(f"✅ Filter Complete! Remaining Pool: {len(valid_candidates)} candidates.")
    return valid_candidates

# ==========================================
# 3. SEMANTIC SCORING & EXPLAINABILITY
# ==========================================

def compile_candidate_text(candidate):
    profile = candidate.get("profile", {})
    return f"Headline: {profile.get('headline', '')}. Summary: {profile.get('summary', '')}. Skills: {', '.join([s.get('name', '') for s in candidate.get('skills', [])])}."

def generate_explanation(candidate):
    title = candidate.get("profile", {}).get("current_title", "Professional")
    exp_years = candidate.get("profile", {}).get("years_of_experience", "Several")
    skills = candidate.get("skills", [])
    top_skills = [s.get("name") for s in skills[:3]]
    skills_str = ", ".join(top_skills) if top_skills else "relevant domain skills"
    return f"Strong match based on {exp_years} years as a {title}, with core expertise in {skills_str}."

def rank_candidates(jd_text, valid_candidates):
    print("\n--- 3. INITIALIZING AI BRAIN ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    candidate_ids = [c.get("candidate_id") for c in valid_candidates]
    names = [c.get("profile", {}).get("anonymized_name", "Unknown") for c in valid_candidates]
    candidate_texts = [compile_candidate_text(c) for c in valid_candidates]
    explanations = [generate_explanation(c) for c in valid_candidates]
    
    jd_embedding = model.encode([jd_text])
    candidate_embeddings = model.encode(candidate_texts)
    scores = cosine_similarity(jd_embedding, candidate_embeddings)[0]
    
    results_df = pd.DataFrame({
        "Candidate_ID": candidate_ids,
        "Name": names,
        "Match_Score": [round(score * 100, 2) for score in scores],
        "Why_They_Match": explanations
    })
    
    return results_df.sort_values(by="Match_Score", ascending=False).reset_index(drop=True)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    jd_path = "India_runs_data_and_ai_challenge/job_description.docx"
    json_path = "India_runs_data_and_ai_challenge/candidates.json"
    jsonl_path = "India_runs_data_and_ai_challenge/candidates.jsonl"
    
    candidates_path = jsonl_path if os.path.exists(jsonl_path) else json_path
        
    print("--- 1. LOADING DATA ---")
    jd_text = read_job_description(jd_path)
    all_candidates = read_candidates(candidates_path)
    
    if isinstance(jd_text, str) and jd_text.startswith("Error"):
        print(jd_text)
    elif isinstance(all_candidates, str) and all_candidates.startswith("Error"):
        print(all_candidates)
    else:
        print(f"✅ System successfully loaded {len(all_candidates)} raw candidates!")
        clean_pool = filter_candidates(all_candidates)
        
        if len(clean_pool) > 0:
            # ⚡ TEST MODE: Set to False to score all 64,000+ candidates
            TEST_MODE = True 
            
            if TEST_MODE:
                print("\n⚠️ RUNNING IN TEST MODE: Only scoring the first 50 candidates!")
                pool_to_score = clean_pool[:50]
            else:
                pool_to_score = clean_pool
                
            final_rankings = rank_candidates(jd_text, pool_to_score)
            final_rankings.to_csv("ranked_shortlist.csv", index=False)
            print("\n✅ Full rankings saved to 'ranked_shortlist.csv' in your folder.")
        else:
            print("❌ No candidates survived the filter!")