from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import scoring, features

app = FastAPI(
    title="Risk Classifier API",
    description="API for calculating risk scores and extracting features from proctoring data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scoring.router)
app.include_router(features.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 