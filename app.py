import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("legal_documents.csv")

# Change column names if your dataset is different
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Legal Document Classifier")

st.write(f"Model Accuracy: {accuracy:.2f}")

user_input = st.text_area("Enter Legal Document Text")

if st.button("Predict"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])
        st.success(f"Predicted Category: {prediction[0]}")
    else:
        st.warning("Please enter some text")
