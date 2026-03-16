from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class FileBase(BaseModel):
    name: str
    path: str
    is_dir: bool = False
    size: int = 0


class FileResponse(FileBase):
    id: int
    parent_path: Optional[str] = None
    ext: Optional[str] = None
    mime_type: Optional[str] = None
    is_starred: bool = False
    thumbnail: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    files: List[FileResponse]
    total: int
    path: str


class FileCreate(BaseModel):
    name: str
    parent_path: str
    is_dir: bool = False


class FileUpdate(BaseModel):
    name: Optional[str] = None
    is_starred: Optional[bool] = None


class FileMoveRequest(BaseModel):
    file_ids: List[int]
    target_path: str


class FileCopyRequest(BaseModel):
    file_ids: List[int]
    target_path: str


class FileDeleteRequest(BaseModel):
    file_ids: List[int]


class FileDownloadRequest(BaseModel):
    file_ids: List[int]


class FolderCreate(BaseModel):
    name: str
    parent_path: str


class StarRequest(BaseModel):
    file_id: int
    starred: bool = True


class FileSearchParams(BaseModel):
    keyword: Optional[str] = None
    file_types: Optional[List[str]] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_starred: Optional[bool] = None