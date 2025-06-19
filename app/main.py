from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Vehicle Recommendation API"
)
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)
from .routes import recommendation_routes
app.include_router(recommendation_routes.router)

@app.get("/")
async def root():
    return {"message": "Vehicle Recommendation API is running."}