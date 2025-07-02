import torch
if not hasattr(torch, "classes"):
    torch.classes = type('classes', (), {})()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PIL import Image
import numpy as np
import re
from collections import Counter
import os

# --------- Utility Functions ---------

def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, low_cpu_mem_usage=False)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

def analyze_texts(texts, classifier):
    results = []
    for text in texts:
        try:
            output = classifier(text)[0]
            label = output['label'].capitalize()
            score = output['score']
            keywords = extract_keywords(text)
            explanation = generate_explanation(text, label, score, keywords)
            results.append({
                "Text": text,
                "Sentiment": label,
                "Confidence": round(score, 3),
                "Keywords": ", ".join(keywords),
                "Explanation": explanation
            })
        except Exception:
            results.append({
                "Text": text,
                "Sentiment": "Error",
                "Confidence": 0.0,
                "Keywords": "",
                "Explanation": "Failed to analyze text"
            })
    return results

def extract_keywords(text):
    stopwords = set([
        "the", "and", "is", "in", "to", "a", "of", "it", "that", "i", "for", "you", "was", "on",
        "with", "as", "at", "this", "but", "be", "have", "are", "not", "my", "me", "so", "if"
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    counts = Counter(filtered)
    most_common = counts.most_common(5)
    return [w for w, c in most_common]

def generate_explanation(text, sentiment, confidence, keywords):
    # Basic explanation logic:
    expl = f"The sentiment is classified as {sentiment} with a confidence of {confidence:.2f}."
    if keywords:
        expl += f" Key words influencing this sentiment include: {', '.join(keywords)}."
    else:
        expl += " No key sentiment-driving words detected."
    if confidence < 0.6:
        expl += " Confidence is low, so this classification might be uncertain."
    return expl

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_json(df):
    return df.to_json(orient="records", indent=4).encode("utf-8")

def convert_df_to_pdf(df, bar_fig, pie_fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Sentiment Analysis Report", styles["Title"]), Spacer(1, 12)]

    # Table data
    data = [df.columns.tolist()] + df.values.tolist()
    elements.append(Table(data, repeatRows=1))
    elements.append(Spacer(1, 12))

    def fig_to_buf(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        return buf

    # Bar chart
    elements.append(Paragraph("Sentiment Distribution - Bar Chart", styles["Heading2"]))
    elements.append(RLImage(fig_to_buf(bar_fig), width=5*inch, height=3*inch))
    elements.append(Spacer(1, 12))

    # Pie chart
    elements.append(Paragraph("Sentiment Distribution - Pie Chart", styles["Heading2"]))
    elements.append(RLImage(fig_to_buf(pie_fig), width=4*inch, height=4*inch))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def plot_sentiment_distribution(df, title="Sentiment Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Sentiment", palette="pastel", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    return fig

def plot_sentiment_pie(df, title="Sentiment Distribution"):
    fig, ax = plt.subplots()
    counts = df["Sentiment"].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title(title)
    ax.axis("equal")
    return fig

def read_texts_from_file(uploaded_file):
    # Save temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".txt"):
        with open(temp_path, "r", encoding="utf-8") as f:
            texts = f.read().splitlines()
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(temp_path)
        # Assume first column has text
        texts = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        texts = []
    return texts

# --------- Streamlit App ---------

st.set_page_config(page_title="Enhanced Sentiment Analysis", layout="wide")

st.title("🧠 Enhanced Sentiment Analysis Dashboard")

# Load model once
@st.cache_resource
def get_classifier():
    return load_model()

classifier = get_classifier()

# Input section
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input method:", ["Direct Text Input", "File Upload", "Compare Two Files"])

texts_1 = []
texts_2 = []

if input_mode == "Direct Text Input":
    text = st.text_area("Enter one or multiple sentences (separate by new lines):", height=150)
    if text:
        texts_1 = [line.strip() for line in text.strip().splitlines() if line.strip()]

elif input_mode == "File Upload":
    uploaded_file = st.file_uploader("Upload a .txt or .csv file with text data:", type=["txt", "csv"])
    if uploaded_file:
        texts_1 = read_texts_from_file(uploaded_file)
        st.success(f"Loaded {len(texts_1)} texts.")

else:  # Compare Two Files
    uploaded_file1 = st.file_uploader("Upload FIRST .txt or .csv file:", type=["txt", "csv"], key="file1")
    uploaded_file2 = st.file_uploader("Upload SECOND .txt or .csv file:", type=["txt", "csv"], key="file2")
    if uploaded_file1:
        texts_1 = read_texts_from_file(uploaded_file1)
        st.success(f"Loaded {len(texts_1)} texts from first file.")
    if uploaded_file2:
        texts_2 = read_texts_from_file(uploaded_file2)
        st.success(f"Loaded {len(texts_2)} texts from second file.")

# Analyze button
if st.button("Analyze Sentiment"):
    if not texts_1:
        st.error("No text data to analyze.")
    else:
        with st.spinner("Analyzing first dataset..."):
            results_1 = analyze_texts(texts_1, classifier)
        df1 = pd.DataFrame(results_1)

        st.subheader("Dataset 1 Results")
        st.dataframe(df1)

        fig1_bar = plot_sentiment_distribution(df1, "Dataset 1 Sentiment Distribution")
        st.pyplot(fig1_bar)
        fig1_pie = plot_sentiment_pie(df1, "Dataset 1 Sentiment Distribution")
        st.pyplot(fig1_pie)

        # Downloads for dataset 1
        csv1 = convert_df_to_csv(df1)
        json1 = convert_df_to_json(df1)
        pdf1 = convert_df_to_pdf(df1, fig1_bar, fig1_pie)
        col1, col2, col3 = st.columns(3)
        col1.download_button("Download Dataset 1 CSV", csv1, "dataset1_sentiments.csv", "text/csv")
        col2.download_button("Download Dataset 1 PDF", pdf1, "dataset1_sentiments.pdf", "application/pdf")
        col3.download_button("Download Dataset 1 JSON", json1, "dataset1_sentiments.json", "application/json")

        # If comparing two files
        if input_mode == "Compare Two Files" and texts_2:
            with st.spinner("Analyzing second dataset..."):
                results_2 = analyze_texts(texts_2, classifier)
            df2 = pd.DataFrame(results_2)

            st.subheader("Dataset 2 Results")
            st.dataframe(df2)

            fig2_bar = plot_sentiment_distribution(df2, "Dataset 2 Sentiment Distribution")
            st.pyplot(fig2_bar)
            fig2_pie = plot_sentiment_pie(df2, "Dataset 2 Sentiment Distribution")
            st.pyplot(fig2_pie)

            # Downloads for dataset 2
            csv2 = convert_df_to_csv(df2)
            json2 = convert_df_to_json(df2)
            pdf2 = convert_df_to_pdf(df2, fig2_bar, fig2_pie)
            col4, col5, col6 = st.columns(3)
            col4.download_button("Download Dataset 2 CSV", csv2, "dataset2_sentiments.csv", "text/csv")
            col5.download_button("Download Dataset 2 PDF", pdf2, "dataset2_sentiments.pdf", "application/pdf")
            col6.download_button("Download Dataset 2 JSON", json2, "dataset2_sentiments.json", "application/json")

            # Comparative Visualization side-by-side
            st.subheader("Comparative Sentiment Distribution")

            comp_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            sns.countplot(data=df1, x="Sentiment", palette="pastel", ax=ax1)
            ax1.set_title("Dataset 1")
            ax1.set_xlabel("")
            ax1.set_ylabel("Count")

            sns.countplot(data=df2, x="Sentiment", palette="muted", ax=ax2)
            ax2.set_title("Dataset 2")
            ax2.set_xlabel("")
            ax2.set_ylabel("")

            st.pyplot(comp_fig)
