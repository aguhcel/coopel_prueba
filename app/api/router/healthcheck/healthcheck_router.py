import logging

from fastapi import APIRouter

from app.api.router.healthcheck.model.response_model import HealthCheckResponse
health_check = APIRouter()
logger = logging.getLogger(__name__)

@health_check.get(
    "/health",
    tags=["health"],
    status_code=200,
    responses={200: {"status": "ok"}},
)
def healt_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")