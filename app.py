import nltk
import spacy
import gensim
import streamlit as st
from nltk.corpus import stopwords
from transformers import pipeline
from rouge_score import rouge_scorer

# Setup once
nltk.download('punkt')
nltk.download('stopwords')

# Load models
nlp = spacy.load("en_core_web_sm")
bert_summarizer = pipeline("summarization")

# Load legal dictionary
LEGAL_TERMS = set([line.strip().lower() for line in open("legal_terms.txt", encoding="utf-8")])

# --- Functions ---

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return [w.lower() for w in tokens if w.isalpha() and w.lower() not in stopwords.words('english')]

def textrank_summary(text):
    try:
        return gensim.summarization.summarize(text, ratio=0.3)
    except:
        return "⚠️ Text too short for TextRank summarization."

def bert_summary(text):
    if len(text.split()) < 50:
        return "⚠️ Text too short for BERT summarization."
    return bert_summarizer(text[:1024])[0]['summary_text']

def detect_illegal_words(text):
    tokens = nltk.word_tokenize(text)
    return list(set([word for word in tokens if word.lower() not in LEGAL_TERMS and word.isalpha()]))

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def evaluate_rouge(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)

# --- Streamlit UI ---

st.set_page_config(page_title="⚖️ Legal Document Analyzer", layout="wide")
st.title("⚖️ Legal Document Summarizer & Illegal Word Detector")

uploaded_file = st.file_uploader("📤 Upload your legal document (.txt)", type=["txt"])
summary_type = st.selectbox("📌 Choose Summarization", ["TextRank (Extractive)", "BERT (Abstractive)"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8").strip()

    if not raw_text:
        st.error("❌ File is empty.")
    elif len(raw_text.split()) < 30:
        st.warning("⚠️ Document too short to analyze.")
    else:
        st.subheader("📖 Original Document")
        st.code(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

        # Summary
        st.subheader("📝 Summary")
        summary = textrank_summary(raw_text) if summary_type == "TextRank (Extractive)" else bert_summary(raw_text)
        st.success(summary)

        # Illegal words
        st.subheader("🚫 Illegal Words")
        illegal_words = detect_illegal_words(raw_text)
        if illegal_words:
            st.warning(", ".join(sorted(illegal_words)))
        else:
            st.success("✅ No illegal words found.")

        # NER
        st.subheader("🔍 Named Entities")
        entities = extract_entities(raw_text)
        if entities:
            st.info(entities)
        else:
            st.info("No named entities found.")

        # ROUGE
        if st.checkbox("📊 Evaluate with ROUGE Score (Optional)"):
            reference_summary = st.text_area("✍️ Paste human-written summary")
            if reference_summary:
                scores = evaluate_rouge(summary, reference_summary)
                st.write(scores)

else:
    st.info("📄 Please upload a `.txt` file to get started.")
