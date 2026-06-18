import streamlit as st
import torch
import requests
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
import pytesseract
import re

st.set_page_config(
    page_title="Hybrid Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_path):
    """Load CSS from an external file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠ CSS file not found: {file_path}")

load_css("style.css")

API_KEY = st.secrets.get("NEWSAPI_KEY", "efc79cdc2bb745c68f92095654c74d52")

import shutil

tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

trusted_sources = [
    "Reuters", "BBC", "BBC News", "Associated Press", "AP News",
    "The Hindu", "New York Times", "NYTimes", "DW", "CNN",
    "The Guardian", "The Indian Express", "Times of India",
    "Hindustan Times", "NDTV", "India Today", "News18", "Firstpost",
    "Deccan Herald", "Business Standard", "The Economic Times",
    "Al Jazeera", "Al Jazeera English", "NPR", "Bloomberg",
    "The Washington Post", "ABC News", "CBS News", "NBC News",
    "Financial Times", "The Wall Street Journal"
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "and", "or", "but",
    "if", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "this", "that", "these", "those",
    "said", "says", "according", "reported", "morning", "evening",
    "today", "yesterday", "tomorrow", "saturday", "sunday", "monday",
    "tuesday", "wednesday", "thursday", "friday", "despite", "earlier",
    "continued", "emerging", "indications", "chief", "party", "process",
    "figure", "pivotal", "support", "formal", "withholding", "uncertainty",
    "hang", "over", "vote", "counting", "rule", "first", "time", "state"
}


@st.cache_resource
def load_model():
    # Load from Hugging Face Hub (replace with your username)
    model_name = "alexjesuraj-29/fake-news-distilbert"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="Hybrid Fake News Detector", layout="wide")
st.title("🗞️ Hybrid Fake News Detection System")

def bert_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()
    return prediction, confidence


def extract_proper_nouns(text):
    proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
    proper_nouns = [p for p in proper_nouns if len(p) > 2]
    seen = set()
    unique = []
    for p in proper_nouns:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    return unique


def extract_important_words(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    important = [w for w in words if w.lower() not in STOPWORDS]
    seen = set()
    unique = []
    for w in important:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return unique


def call_newsapi(query, page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": API_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
        return []
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def is_relevant(article, key_terms, min_matches=2):
    title = (article.get("title") or "").lower()
    desc = (article.get("description") or "").lower()
    content = (article.get("content") or "").lower()
    combined = f"{title} {desc} {content}"

    matches = 0
    matched_terms = []
    for term in key_terms:
        if re.search(r'\b' + re.escape(term.lower()) + r'\b', combined):
            matches += 1
            matched_terms.append(term)
    return matches >= min_matches, matches, matched_terms


def filter_relevant_articles(articles, key_terms, min_matches=2):
    scored = []
    for art in articles:
        relevant, score, matched = is_relevant(art, key_terms, min_matches)
        if relevant:
            scored.append((score, art, matched))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def compute_confirmation_strength(articles, key_terms):
    """Confirmation score 0-1: how strongly articles confirm the claim."""
    if not articles or not key_terms:
        return 0.0

    total_terms = len(key_terms)
    title_match_ratios = []

    for art in articles[:5]:
        title = (art.get("title") or "").lower()
        desc = (art.get("description") or "").lower()
        title_matches = sum(
            1 for term in key_terms
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', title)
        )
        desc_matches = sum(
            1 for term in key_terms
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', desc)
        )
        weighted = (title_matches * 2 + desc_matches) / (total_terms * 2)
        title_match_ratios.append(min(weighted, 1.0))

    avg_strength = sum(title_match_ratios) / len(title_match_ratios)
    return avg_strength


def search_news(query):
    headline = " ".join(re.split(r'(?<=[.!?])\s+', query.strip())[:2])
    proper_nouns = extract_proper_nouns(headline)
    important_words = extract_important_words(headline)

    key_terms = proper_nouns + [
        w for w in important_words[:5]
        if w.lower() not in [p.lower() for p in proper_nouns]
    ]

    if len(proper_nouns) >= 3:
        min_matches = 2
    elif len(proper_nouns) >= 2:
        min_matches = 2
    else:
        min_matches = 1

    all_articles = []
    tried_queries = []
    seen_urls = set()

    def add_unique(arts):
        for a in arts:
            url = a.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_articles.append(a)

    if len(proper_nouns) >= 2:
        q1 = " AND ".join([f'"{p}"' for p in proper_nouns[:3]])
        tried_queries.append(q1)
        add_unique(call_newsapi(q1))

    if len(proper_nouns) >= 2:
        q2 = " AND ".join([f'"{p}"' for p in proper_nouns[:2]])
        tried_queries.append(q2)
        add_unique(call_newsapi(q2))

    if proper_nouns:
        q3 = " ".join(proper_nouns[:4])
        tried_queries.append(q3)
        add_unique(call_newsapi(q3))

    mix = proper_nouns[:2] + important_words[:3]
    if mix:
        q4 = " ".join(mix)
        tried_queries.append(q4)
        add_unique(call_newsapi(q4))

    if important_words and not all_articles:
        q5 = " ".join(important_words[:5])
        tried_queries.append(q5)
        add_unique(call_newsapi(q5))

    relevant_scored = filter_relevant_articles(all_articles, key_terms, min_matches)

    if not relevant_scored and proper_nouns:
        relevant_scored = filter_relevant_articles(all_articles, proper_nouns, 1)

    final_articles = [art for _, art, _ in relevant_scored]
    confirmation_strength = compute_confirmation_strength(final_articles, key_terms)

    with st.expander("🔍 Search Debug Info"):
        st.write("**Proper nouns extracted:**", proper_nouns)
        st.write("**Important words:**", important_words[:10])
        st.write("**Key terms used for relevance check:**", key_terms)
        st.write("**Min matches required:**", min_matches)
        st.write("**Queries tried:**")
        for q in tried_queries:
            st.code(q)
        st.write("**Total raw articles fetched:**", len(all_articles))
        st.write("**Relevant articles after filtering:**", len(final_articles))
        st.write(f"**🎯 Confirmation Strength: {confirmation_strength:.2f}**")
        if relevant_scored:
            st.write("**Top match scores:**")
            for score, art, matched in relevant_scored[:5]:
                st.write(f"- Score `{score}` | matched: `{matched}` | {art.get('title')}")

    return final_articles, confirmation_strength


def clean_ocr_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,:;!?()%₹$ -]", "", text)
    return text.strip()


def display_results(text):
    prediction, confidence = bert_predict(text)
    raw_ai_real_score = confidence if prediction == 1 else (1 - confidence)
    model_score = 0.6 * raw_ai_real_score + 0.2

    articles, confirmation_strength = search_news(text)
    article_count = len(articles)
    trusted_found = False
    trusted_count = 0

    st.subheader("🧠 AI Model Result")
    st.progress(int(model_score * 100))

    if prediction == 1:
        st.success(f"Model Prediction: Real (raw confidence: {raw_ai_real_score:.2f})")
    else:
        st.error(f"Model Prediction: Fake (raw confidence: {1-raw_ai_real_score:.2f})")

    st.write(f"AI Smoothed Score: {model_score:.2f}")
    st.caption("⚠ Note: AI model trained on US-style news may misclassify regional/non-Western news.")

    st.subheader("🌐 Internet Verification")

    if article_count > 0:
        st.metric("Relevant Articles Retrieved", article_count)
        st.metric("Confirmation Strength", f"{confirmation_strength:.2f} / 1.00")

        for article in articles[:5]:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            url = article.get("url", "")

            st.markdown(f'<a href="{url}" target="_blank">{title}</a>',
                        unsafe_allow_html=True)
            st.write("Source:", source)

            if any(ts.lower() in source.lower() for ts in trusted_sources):
                trusted_found = True
                trusted_count += 1
                st.success("✅ Trusted Source")

            st.write("---")
    else:
        st.warning("No related articles found.")


    st.subheader("🔎 Final Verdict")

    ai_strong_fake = prediction == 0 and (1 - raw_ai_real_score) >= 0.90
    ai_strong_real = prediction == 1 and raw_ai_real_score >= 0.85

    strong_evidence   = confirmation_strength >= 0.40 and trusted_count >= 1
    medium_evidence   = confirmation_strength >= 0.25 and article_count >= 2
    weak_evidence     = confirmation_strength >= 0.15 and article_count >= 1
    no_evidence       = article_count == 0

    verdict_given = False

    if strong_evidence:
        st.success(f"✅ Likely REAL — {trusted_count} trusted source(s) confirm this claim "
                   f"(confirmation strength: {confirmation_strength:.2f}). "
                   f"This overrides the AI model, which may be biased on this topic.")
        verdict_given = True

    elif medium_evidence and article_count >= 3:
        st.success(f"✅ Likely REAL — {article_count} articles confirm related details "
                   f"(strength: {confirmation_strength:.2f}).")
        verdict_given = True

    elif trusted_count >= 1 and confirmation_strength >= 0.25:
        st.success(f"✅ Likely REAL — Trusted source supports the claim "
                   f"(confirmation strength: {confirmation_strength:.2f}).")
        verdict_given = True

    elif no_evidence and ai_strong_fake:
        st.error("❌ Likely FAKE — Strong AI detection AND no online presence "
                 "for this claim.")
        verdict_given = True

    elif no_evidence and prediction == 0:
        st.error("❌ Likely FAKE — AI detects fake patterns and no online "
                 "confirmation found.")
        verdict_given = True

    elif no_evidence and ai_strong_real:
        st.warning("⚠ AI says Real but no online confirmation found — "
                   "verify manually (could be very recent or niche news).")
        verdict_given = True

    elif weak_evidence and prediction == 1:
        st.success("✅ Likely REAL — AI and some online sources weakly agree.")
        verdict_given = True

    elif weak_evidence and prediction == 0:
        st.warning(f"⚠ Inconclusive — AI flags as fake but {article_count} weakly related "
                   f"article(s) exist. The AI may be biased on this topic. "
                   f"Manual verification recommended.")
        verdict_given = True

    if not verdict_given:
        st.warning("⚠ Inconclusive — Mixed signals between AI and retrieved evidence.")

    with st.expander("🧮 Verdict Reasoning Breakdown"):
        st.write(f"- **AI Prediction:** {'REAL' if prediction == 1 else 'FAKE'}")
        st.write(f"- **AI Raw Confidence:** {raw_ai_real_score:.2f} (real)")
        st.write(f"- **Articles found:** {article_count}")
        st.write(f"- **Trusted source count:** {trusted_count}")
        st.write(f"- **Confirmation strength:** {confirmation_strength:.2f}")
        st.write(f"- **Strong evidence:** {strong_evidence}")
        st.write(f"- **Medium evidence:** {medium_evidence}")
        st.write(f"- **Weak evidence:** {weak_evidence}")
        st.write(f"- **No evidence:** {no_evidence}")

    st.caption("Verdict prioritizes external evidence (especially trusted sources) "
               "over AI opinion, since AI may be biased toward its training data.")



st.subheader("📝 Verify News via Text")
user_input = st.text_area("Enter News Article Text:")

if st.button("Analyze Text News"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        with st.spinner("Analyzing text..."):
            display_results(user_input)



st.subheader("👀 Verify News via Screenshot")
uploaded_file = st.file_uploader(
    "Upload Screenshot (Only News Area for Best Results)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", width="stretch")

    raw_text = pytesseract.image_to_string(image)
    extracted_text = clean_ocr_text(raw_text)

    st.subheader("📄 Extracted Text")
    st.write(extracted_text)

    if st.button("Analyze Screenshot News"):
        if not extracted_text.strip():
            st.warning("No readable text detected.")
        else:
            display_results(extracted_text)