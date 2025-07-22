from fastapi import FastAPI, APIRouter

from app.api.router.healthcheck.healthcheck_router import health_check

app = FastAPI(
    title="Prueba Técnica"
)

api_router = APIRouter()

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root Get
    """
    return {"message": "Hello World"}

app.include_router(api_router)
app.include_router(health_check)