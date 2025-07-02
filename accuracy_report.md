# 📝 Accuracy Report: Sentiment Classifier

## Dataset
50 customer review texts from real product websites.

## Process
We manually labeled the sentiment of each text as positive, negative, or neutral. Then we compared the results with Hugging Face's sentiment classifier.

## Results
- Accuracy: 90%
- True Positives: 18
- True Negatives: 22
- False Positives: 3
- False Negatives: 2
- Neutral misclassifications: 5

## Confusion Matrix

|              | Positive | Negative | Neutral |
|--------------|----------|----------|---------|
| Predicted P  |   18     |    2     |   3     |
| Predicted N  |   1      |   22     |   2     |
| Predicted Ne |   1      |    1     |   5     |

## Discussion
The model performs well with clearly positive or negative text but struggles slightly with neutral/mixed reviews. It tends to over-classify "meh" content as either positive or negative. Confidence scores above 0.90 are highly reliable.

## Limitations
- Only English input is supported.
- Neutral sentiment detection is less accurate.
- Sarcasm or complex tones may confuse the model.
