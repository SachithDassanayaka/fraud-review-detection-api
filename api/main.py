from fastapi import FastAPI

app = FastAPI(title="Fraud Review Detection API")

@app.get("/")
def root():
    return {"message": "Fraud Review Detection API running"}