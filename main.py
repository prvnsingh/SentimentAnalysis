from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def heath_check():
    return {"response": "The application is up and running"}


@app.get("/sentiment_analysis/{text}")
def sentiment_analysis(text: str):
    return {"response": text}
