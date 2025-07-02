import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re

# Load dataset with manual labels
df = pd.read_csv("sentiment_samples.csv")

# Load tokenizer and model for CardiffNLP twitter-roberta-base-sentiment-latest
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to preprocess text (replace URLs, usernames, etc.)
def preprocess(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)     # remove mentions
    return text.strip()

# Label mapping from model outputs to sentiment labels
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

predicted_labels = []
confidence_scores = []

for text in df["Text"]:
    text_proc = preprocess(text)
    encoded_input = tokenizer(text_proc, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits[0].softmax(dim=0)
    pred_label = torch.argmax(scores).item()
    confidence = scores[pred_label].item()
    predicted_labels.append(label_map[pred_label])
    confidence_scores.append(confidence)

df["Predicted Sentiment"] = predicted_labels
df["Confidence Score"] = confidence_scores

# Save predictions CSV
df.to_csv("sentiment_predictions.csv", index=False)

# Generate classification report and save
report = classification_report(df["Manual Sentiment"], df["Predicted Sentiment"], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv")

# Confusion matrix
labels = ["Positive", "Negative", "Neutral"]
cm = confusion_matrix(df["Manual Sentiment"], df["Predicted Sentiment"], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot confusion matrix and save image
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix - Sentiment Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close(fig)

print("✅ Sentiment analysis complete. Files saved:")
print("- sentiment_predictions.csv")
print("- classification_report.csv")
print("- confusion_matrix.png")

