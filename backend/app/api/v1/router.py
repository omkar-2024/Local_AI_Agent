from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.llm_router import router_service

router = APIRouter()

class RouteRequest(BaseModel):
    prompt: str
    attachment_types: List[str] = []

@router.post("/route")
async def route_task(request: RouteRequest):
    route_result = await router_service.classify_and_route(
        user_prompt=request.prompt,
        attachment_types=request.attachment_types
    )
    return route_result