from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers.fraud_router import router


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(
    title="Fraud Detection API",
    version="1.0",
)


app.include_router(router)
app.mount(
    "/ui",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running. Open /ui/ for the web interface."
    }
