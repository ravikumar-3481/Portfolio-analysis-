import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import torch
import traceback

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Portfolio Analyzer", page_icon="🧠", layout="wide")

# --- Deep Learning Model Caching ---
# Hum @st.cache_resource ka use kar rahe hain taaki bhari ML models sirf ek baar memory mein load hon.
@st.cache_resource
def load_ml_models():
    # Page jaldi load ho isliye hum transformers ko function ke andar import kar rahe hain
    from transformers import pipeline
    
    try:
        # 1. Summarization Model (Cloud hosting ke liye chhota model taaki Out-Of-Memory error na aaye)
        summarizer = pipeline("summarization", model="Falconsai/text_summarization", framework="pt")
        
        # 2. Zero-Shot Classifier (Skill nikalne ke liye DistilBERT, jo BART-large se kafi halka hai)
        classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt")
        
        # 3. Named Entity Recognition (Naam aur Company nikalne ke liye)
        ner = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER", framework="pt")
        
        return summarizer, classifier, ner
    except Exception as e:
        # Agar error aaye toh app crash na ho, balki error message return ho jaye
        return None, None, f"Model Loading Error: {str(e)}\n\n{traceback.format_exc()}"

# --- Web Scraping Function ---
def extract_text_from_url(url):
    """URL se data nikalta hai aur saaf text return karta hai."""
    try:
        # Asli browser jaisa dikhne ke liye headers set kar rahe hain taaki 403 error na aaye
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Faltu elements (scripts, styles, navbars) ko hata rahe hain
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()
            
        # Text nikalkar extra spaces ko saaf kar rahe hain
        text = soup.get_text(separator=' ')
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        return clean_text
    except Exception as e:
        st.error(f"URL se data nikalne mein error aaya: {e}")
        return None

# --- Main Application UI ---
def main():
    st.title("🧠 ProXcelerate: Deep Learning Portfolio Analyzer")
    st.markdown("Neeche kisi bhi portfolio ka URL daalein. Hamara AI us site ka data padhega aur Deep Learning models (BART & BERT) ka use karke profile ki summary banayega, skills nikalega aur zaruri details batayega.")

    # 1. URL Input Box
    url_input = st.text_input("Portfolio Website URL", placeholder="https://johndoe.dev")
    
    if st.button("Portfolio Analyze Karein", type="primary"):
        if not url_input.startswith("http"):
            st.warning("Kripya ek sahi URL daalein jo http:// ya https:// se shuru hota ho.")
            return

        # 2. Loading Models
        with st.spinner("Deep Learning Models memory mein load ho rahe hain (Pehli baar thoda samay lag sakta hai)..."):
            summarizer, classifier, ner = load_ml_models()
            
        # Agar models load hone mein fail ho gaye
        if isinstance(ner, str) and "Error" in ner:
            st.error("⚠️ AI Models load nahi ho paye. Kripya neeche error check karein:")
            st.code(ner)
            st.info("Tip: Agar aap Streamlit Cloud par hain, toh 'Manage App' -> ⋮ -> 'Reboot app' par click karein.")
            return
        
        # 3. Scraping Data
        with st.spinner("Portfolio se data nikala ja raha hai..."):
            raw_text = extract_text_from_url(url_input)
            
        if not raw_text:
            return
            
        # AI models limit cross na karein isliye text ko chhota kar rahe hain (max ~400 tokens)
        safe_text = raw_text[:2000] 

        if len(safe_text) < 100:
            st.warning("Is URL se AI analysis ke liye kafi data nahi mil paya.")
            return

        # 4. Running AI Predictions
        with st.spinner("AI apka data analyze kar raha hai (Summarization, NER, Zero-Shot Classification)..."):
            # A. Summarization
            try:
                summary_out = summarizer(safe_text, max_length=130, min_length=30, do_sample=False, truncation=True)
                summary_text = summary_out[0]['summary_text']
            except Exception as e:
                summary_text = "Data ka format sahi na hone ke karan summary nahi ban payi."

            # B. Skill Extraction (Zero-Shot)
            tech_skills = [
                "Python", "JavaScript", "React", "Node.js", "Machine Learning", 
                "Deep Learning", "Data Science", "AWS", "Docker", "Kubernetes", 
                "SQL", "NoSQL", "UI/UX", "Java", "C++", "Cybersecurity", "DevOps"
            ]
            try:
                skill_predictions = classifier(safe_text, tech_skills, multi_label=True)
                # Jo skills 60% se zyada confidence ke sath mile hain, unhe filter karein
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
        st.success("Analysis Poori Ho Gayi!")
        
        st.header("📊 AI Career Intelligence Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 AI Profile Summary")
            st.info(summary_text)
            
            st.subheader("⚙️ Pehchani Gayi Technical Skills")
            if detected_skills:
                for item in detected_skills:
                    # AI confidence score dikhane ke liye progress bar
                    st.write(f"**{item['skill']}** (Confidence: {int(item['score']*100)}%)")
                    st.progress(float(item['score']))
            else:
                st.write("Hamari list mein se koi khaas technical skill detect nahi ho payi.")

        with col2:
            st.subheader("🏢 Nikali Gayi Entities (NER)")
            st.markdown("**Organizations (Company/Sanstha):**")
            if orgs:
                for org in orgs:
                    st.markdown(f"- 🏢 `{org}`")
            else:
                st.write("*Kuch nahi mila*")
                
            st.markdown("**Naam (Log):**")
            if names:
                for name in names:
                    st.markdown(f"- 👤 `{name}`")
            else:
                st.write("*Kuch nahi mila*")
                
        # Asli text dekhne ke liye expander
        with st.expander("Website se nikala gaya asli Text dekhein"):
            st.write(safe_text)

if __name__ == "__main__":
    main()
