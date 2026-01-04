import streamlit as st
import pandas as pd
import string
import nltk
import joblib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

# ---------------- THEME SETTINGS ---------------- #
card_bg = "rgba(255,255,255,0.88)" if theme == "Light" else "rgba(2,6,23,0.88)"
text_color = "#020617" if theme == "Light" else "#e5e7eb"

# ---------------- GLASSMORPHISM BACKGROUND ---------------- #
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://images.unsplash.com/photo-1530497610245-94d3c16cda28");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    .main {{
        background: {card_bg};
        backdrop-filter: blur(14px);
        padding: 2.5rem;
        border-radius: 18px;
        color: {text_color};
        box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    }}

    textarea {{
        border-radius: 12px;
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- NLTK ---------------- #
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- TEXT CLEANING ---------------- #
def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = ''.join([i for i in txt if i.isascii()])
    words = txt.split()
    return " ".join([w for w in words if w not in stop_words])

# ---------------- EMOTION COLORS ---------------- #
emotion_colors = {
    "joy": "#22c55e",
    "happy": "#16a34a",
    "sadness": "#2563eb",
    "anger": "#dc2626",
    "fear": "#7c3aed",
    "love": "#db2777",
    "surprise": "#f59e0b"
}

# ---------------- LOAD / TRAIN MODEL ---------------- #
@st.cache_data
def load_model():
    try:
        model = joblib.load("emotion_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        reverse_map = joblib.load("label_map.pkl")
    except:
        df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

        emo_map = {emo: idx for idx, emo in enumerate(df["emotion"].unique())}
        reverse_map = {v: k for k, v in emo_map.items()}
        df["emotion"] = df["emotion"].map(emo_map)

        df["text"] = df["text"].apply(clean_text)

        X_train, _, y_train, _ = train_test_split(
            df["text"], df["emotion"], test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)

        joblib.dump(model, "emotion_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        joblib.dump(reverse_map, "label_map.pkl")

    return model, vectorizer, reverse_map

model, vectorizer, reverse_map = load_model()

# ---------------- UI ---------------- #
st.title("🧠 Emotion Detection AI")
st.write("Analyze emotions from text using NLP & Machine Learning")

st.divider()

user_text = st.text_area(
    "Enter your text",
    placeholder="I feel confident, motivated and excited today!",
    height=120
)

if st.button("✨ Analyze Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])

        pred = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        emotion = reverse_map[pred]
        color = emotion_colors.get(emotion.lower(), "#020617")

        # ---------- Emotion Card ---------- #
        st.markdown(
            f"""
            <div style="
                padding:1.2rem;
                border-radius:16px;
                background:{color};
                color:white;
                text-align:center;
                font-size:24px;
                font-weight:600;
                margin-top:15px;
                box-shadow:0 10px 25px rgba(0,0,0,0.25);
            ">
                Detected Emotion: {emotion.capitalize()}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- Confidence Chart ---------- #
        st.subheader("📊 Emotion Confidence")
        labels = [reverse_map[i] for i in range(len(probs))]

        fig, ax = plt.subplots()
        ax.barh(labels, probs)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")

        st.pyplot(fig)

st.divider()
st.caption("Built with NLP • TF-IDF • Logistic Regression • Streamlit")
