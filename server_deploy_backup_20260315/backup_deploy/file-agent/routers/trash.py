from fastapi import APIRouter, Depends, Query
from typing import List
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from services.trash_service import TrashService

router = APIRouter(prefix="/api/trash", tags=["回收站"])


class RestoreRequest(BaseModel):
    ids: List[int]


class DeleteRequest(BaseModel):
    ids: List[int]


@router.get("")
async def list_trash(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trash_service = TrashService(db)
    return trash_service.list_trash(page=page, page_size=page_size)


@router.post("/restore")
async def restore_files(
    request: RestoreRequest,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trash_service = TrashService(db)
    result = trash_service.restore(request.ids)
    return result


@router.delete("")
async def delete_permanently(
    request: DeleteRequest,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trash_service = TrashService(db)
    result = trash_service.delete_permanently(request.ids)
    return result


@router.delete("/empty")
async def empty_trash(
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    trash_service = TrashService(db)
    result = trash_service.empty_trash()
    return result