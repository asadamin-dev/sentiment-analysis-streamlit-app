# Sentiment Analysis Web App (Streamlit + NLP + ML)

A simple and complete **Sentiment Analysis Web Application** built using **Python, NLP, Machine Learning, and Streamlit**.

---

## 🚀 Features
- Text preprocessing & cleaning  
- TF-IDF vectorization  
- Machine learning model (Naive Bayes)  
- Real-time sentiment prediction  
- Interactive Streamlit UI  

---

## 📁 Project Structure
sentiment_streamlit_app/
├── train_model.py
├── streamlit_app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── readme.md


---

## ⚙ Installation & Run

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```
Open in browser:
```bash
http://localhost:8501
```
## 🧪 Sample Test Inputs
I love this product → Positive 😊

Very bad service → Negative 😞

## 🛠 Tech Stack
Python · NLP · Scikit-learn · Streamlit

