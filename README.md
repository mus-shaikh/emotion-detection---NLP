# 🧠 Emotion Detection AI (NLP Project)

An end-to-end **Emotion Detection system** built using **Natural Language Processing (NLP)** and **Machine Learning**, deployed with an interactive **Streamlit UI**.

The application predicts the **emotion behind a given text input** and visualizes model confidence in real time.

---

## 🚀 Features

- 🔍 Emotion detection from raw text
- 🧠 NLP preprocessing (cleaning, stopwords removal, TF-IDF)
- 🤖 Machine Learning model (Logistic Regression)
- 📊 Emotion confidence visualization
- 🎨 Dynamic UI with Light/Dark mode
- 💾 Saved model loading (no retraining on deployment)
- 🌍 Deployed using Streamlit Cloud

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **NLP:** NLTK, TF-IDF Vectorizer  
- **Machine Learning:** Logistic Regression (scikit-learn)  
- **Frontend:** Streamlit  
- **Visualization:** Matplotlib  
- **Deployment:** GitHub + Streamlit Cloud  

---

## 📁 Project Structure
emotion-detection---NLP/
│
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── emotion_model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── label_map.pkl # Label mapping
├── train.txt # Dataset (used for training)
└── README.md # Project documentation

