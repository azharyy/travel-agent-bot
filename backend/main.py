from fastapi import FastAPI

app = FastAPI(title="Travel Agent Bot API")

@app.get("/")
def read_root():
    return {"status": "Backend is running!", "message": "Ready to route AI agents."}
