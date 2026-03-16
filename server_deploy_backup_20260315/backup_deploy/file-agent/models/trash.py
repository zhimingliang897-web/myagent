from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TrashItemBase(BaseModel):
    name: str
    original_path: str
    is_dir: bool = False
    size: int = 0


class TrashItemResponse(TrashItemBase):
    id: int
    trash_path: str
    deleted_at: datetime
    
    class Config:
        from_attributes = True


class TrashListResponse(BaseModel):
    items: List[TrashItemResponse]
    total: int


class TrashRestoreRequest(BaseModel):
    ids: List[int]


class TrashDeleteRequest(BaseModel):
    ids: List[int]