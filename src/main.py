from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI()


@app.get("/")
def heath_check():
    return {"response": "The application is up and running"}


@app.get("/sentiment_analysis/{text}")
def sentiment_analysis(text: str):
    model_path = "../results/checkpoint-500"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, labels=2)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return {"response": pipe(text)}
