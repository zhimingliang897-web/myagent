from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from services.agent_service import AgentService, global_search_progress

router = APIRouter(prefix="/api/agent", tags=["AI助手"])


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    agent_service = AgentService(db)
    result = agent_service.chat(request.message, request.context)
    return result


@router.get("/progress")
async def get_progress():
    def generate():
        import time
        last_msg = ""
        last_count = -1
        
        while True:
            current_status = global_search_progress["status"]
            current_msg = global_search_progress["message"]
            current_count = global_search_progress["found_count"]
            
            if current_status != "idle" or last_msg != current_msg or last_count != current_count:
                yield f"data: {json.dumps({'status': current_status, 'message': current_msg, 'found_count': current_count})}\n\n"
                last_msg = current_msg
                last_count = current_count
            
            if current_status == "idle" and last_msg != "":
                yield f"data: {json.dumps({'status': 'idle', 'message': '', 'found_count': 0})}\n\n"
                break
            
            time.sleep(0.5)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )