import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import torch

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Portfolio Analyzer", page_icon="🧠", layout="wide")

# --- Deep Learning Model Caching ---
# We use @st.cache_resource so the heavy ML models are only downloaded/loaded into memory ONCE.
@st.cache_resource
def load_ml_models():
    # We import transformers here to avoid slowing down the initial page render
    from transformers import pipeline
    
    # 1. Summarization Model (Smaller model for cloud hosting to avoid Out-Of-Memory errors)
    summarizer = pipeline("summarization", model="Falconsai/text_summarization", framework="pt")
    
    # 2. Zero-Shot Classifier (DistilBERT is much lighter than BART-large)
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt")
    
    # 3. Named Entity Recognition
    ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER", framework="pt")
    
    return summarizer, classifier, ner

# --- Web Scraping Function ---
def extract_text_from_url(url):
    """Scrapes the URL and extracts clean visible text."""
    try:
        # Define headers to mimic a real browser to prevent 403 Forbidden errors
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy elements (scripts, styles, navbars, footers)
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()
            
        # Extract text and clean up whitespaces
        text = soup.get_text(separator=' ')
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        return clean_text
    except Exception as e:
        st.error(f"Error scraping URL: {e}")
        return None

# --- Main Application UI ---
def main():
    st.title("🧠 ProXcelerate: Deep Learning Portfolio Analyzer")
    st.markdown("Enter a portfolio URL below. Our AI engine will scrape the site, run deep learning models (BART & BERT) to summarize the profile, extract verified skills, and identify key entities.")

    # 1. URL Input
    url_input = st.text_input("Portfolio Website URL", placeholder="https://johndoe.dev")
    
    if st.button("Analyze Portfolio", type="primary"):
        if not url_input.startswith("http"):
            st.warning("Please enter a valid URL starting with http:// or https://")
            return

        # 2. Loading Models
        with st.spinner("Loading Deep Learning Models into memory (this takes a moment on first run)..."):
            summarizer, classifier, ner = load_ml_models()
        
        # 3. Scraping Data
        with st.spinner("Scraping portfolio data..."):
            raw_text = extract_text_from_url(url_input)
            
        if not raw_text:
            return
            
        # Truncate text to avoid exceeding transformer model token limits (max 512 tokens)
        # 2000 characters is a safer approximation for ~400 tokens
        safe_text = raw_text[:2000] 

        if len(safe_text) < 100:
            st.warning("Not enough text could be extracted from this URL for AI analysis.")
            return

        # 4. Running AI Predictions
        with st.spinner("Running Deep Learning Inference (Summarization, NER, Zero-Shot Classification)..."):
            # A. Summarization
            try:
                # Added truncation=True to prevent indexing errors on long text
                summary_out = summarizer(safe_text, max_length=130, min_length=30, do_sample=False, truncation=True)
                summary_text = summary_out[0]['summary_text']
            except Exception as e:
                summary_text = "Could not generate summary due to text format."

            # B. Skill Extraction (Zero-Shot)
            tech_skills = [
                "Python", "JavaScript", "React", "Node.js", "Machine Learning", 
                "Deep Learning", "Data Science", "AWS", "Docker", "Kubernetes", 
                "SQL", "NoSQL", "UI/UX", "Java", "C++", "Cybersecurity", "DevOps"
            ]
            try:
                skill_predictions = classifier(safe_text, tech_skills, multi_label=True)
                # Filter skills with a confidence score > 60%
                detected_skills = [
                    {"skill": skill, "score": score} 
                    for skill, score in zip(skill_predictions['labels'], skill_predictions['scores']) 
                    if score > 0.60
                ]
            except Exception:
                detected_skills = []

            # C. Entity Extraction (NER)
            try:
                entities = ner(safe_text)
                orgs = list(set([ent['word'] for ent in entities if ent['entity_group'] == 'ORG']))
                names = list(set([ent['word'] for ent in entities if ent['entity_group'] == 'PER']))
            except Exception:
                orgs, names = [], []

        # 5. Displaying Results Dashboard
        st.success("Analysis Complete!")
        
        st.header("📊 AI Career Intelligence Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 AI Profile Summary (BART Model)")
            st.info(summary_text)
            
            st.subheader("⚙️ Detected Technical Skills (Zero-Shot NLI)")
            if detected_skills:
                for item in detected_skills:
                    # Display skill with a progress bar representing AI confidence
                    st.write(f"**{item['skill']}** (Confidence: {int(item['score']*100)}%)")
                    st.progress(float(item['score']))
            else:
                st.write("No specific technical skills from our benchmark list were confidently detected.")

        with col2:
            st.subheader("🏢 Extracted Entities (BERT NER)")
            st.markdown("**Organizations Mentioned:**")
            if orgs:
                for org in orgs:
                    st.markdown(f"- 🏢 `{org}`")
            else:
                st.write("*None detected*")
                
            st.markdown("**People Mentioned:**")
            if names:
                for name in names:
                    st.markdown(f"- 👤 `{name}`")
            else:
                st.write("*None detected*")
                
        # Expander for raw data verification
        with st.expander("View Raw Extracted Text"):
            st.write(safe_text)

if __name__ == "__main__":
    main()
