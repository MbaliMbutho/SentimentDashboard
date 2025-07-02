# utils.py

def analyze_texts(texts, classifier):
    results = []
    for text in texts:
        try:
            output = classifier(text)[0]
            sentiment = output['label'].capitalize()
            confidence = output['score']
        except Exception:
            sentiment = "Error"
            confidence = 0.0
        results.append({"Text": text, "Sentiment": sentiment, "Confidence": round(confidence, 2)})
    return results

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")
