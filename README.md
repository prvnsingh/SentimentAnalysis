# SentimentAnalysis
sentiment analysis using hugging face pretrained model

fine tuned hugging face model "distilbert-base-uncased-finetuned-sst-2-english"
for text classification (Sentiment analysis)

implemented FAST api and transformer pipeline for easy access of trained model and predict sentiment of english text through api.

to run the FAST server : uvicorn main:app --reload



to train again on custom dataset/ custom model : change train.py and run it.
