from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from pathlib import Path
from urllib.parse import quote
import io
import base64
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from services.preview_service import PreviewService

router = APIRouter(prefix="/api/preview", tags=["预览"])


def get_base_url(request: Request) -> str:
    return str(request.base_url).rstrip('/')


@router.get("")
async def preview_file(
    request: Request,
    path: str = Query(...),
    user: str = Depends(get_current_user)
):
    preview_service = PreviewService()
    base_url = get_base_url(request)

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    ext = file_path.suffix.lower()
    preview_type = preview_service.get_preview_type(ext)
    encoded_path = quote(path, safe='')

    if preview_type == "text":
        content = preview_service.get_text_preview(path)
        return {
            "type": "text",
            "content": content,
            "filename": file_path.name
        }

    elif preview_type == "image":
        try:
            img_data = preview_service.get_image_base64(path)
            return {
                "type": "image",
                "data": img_data,
                "filename": file_path.name
            }
        except Exception as e:
            return {
                "type": "image",
                "url": f"{base_url}/api/preview/file?path={encoded_path}",
                "filename": file_path.name,
                "error": str(e)
            }

    elif preview_type == "video":
        return {
            "type": "video",
            "url": f"{base_url}/api/preview/file?path={encoded_path}",
            "filename": file_path.name
        }

    elif preview_type == "audio":
        return {
            "type": "audio",
            "url": f"{base_url}/api/preview/file?path={encoded_path}",
            "filename": file_path.name
        }

    elif preview_type == "pdf":
        return {
            "type": "pdf",
            "url": f"{base_url}/api/preview/file?path={encoded_path}",
            "filename": file_path.name
        }

    else:
        return {
            "type": "unknown",
            "message": "不支持预览此类型文件",
            "filename": file_path.name
        }


@router.get("/file")
async def get_file(
    path: str = Query(...),
    user: str = Depends(get_current_user)
):
    preview_service = PreviewService()
    
    try:
        return preview_service.get_file_response(path)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在")


@router.get("/thumb")
async def get_thumbnail(
    path: str = Query(...),
    user: str = Depends(get_current_user)
):
    preview_service = PreviewService()
    
    try:
        thumb_io = preview_service.get_image_thumbnail(path)
        return StreamingResponse(
            thumb_io,
            media_type="image/jpeg"
        )
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="文件不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))