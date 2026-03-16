from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from typing import List, Optional
from pathlib import Path
import shutil
import zipfile
import tempfile
import os
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.config import settings
from services.file_service import FileService
from services.upload_service import UploadService
from services.trash_service import TrashService

router = APIRouter(prefix="/api/files", tags=["文件管理"])


@router.get("")
async def list_files(
    path: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort_by: str = Query(default="name"),
    sort_order: str = Query(default="asc"),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    files, total, current_path = file_service.list_files(
        path=path,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    breadcrumb = file_service.get_breadcrumb(current_path)
    
    return {
        "files": files,
        "total": total,
        "path": current_path,
        "breadcrumb": breadcrumb,
        "page": page,
        "page_size": page_size
    }


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    target_path: Optional[str] = Form(default=None),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    upload_service = UploadService(db)
    result = await upload_service.upload_files(files, target_path)
    return result


@router.post("/folder")
async def create_folder(
    name: str = Form(...),
    parent_path: str = Form(default=""),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    
    try:
        result = file_service.create_folder(name, parent_path)
        return {"success": True, "folder": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/download")
async def download_files(
    paths: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    path_list = paths.split(",")
    
    for p in path_list:
        if not file_service._is_path_allowed(p):
            raise HTTPException(status_code=403, detail="路径不允许访问")
    
    if len(path_list) == 1:
        file_path = Path(path_list[0])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name
        )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_file.close()
    
    try:
        with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in path_list:
                file_path = Path(path)
                if file_path.exists():
                    if file_path.is_file():
                        zf.write(str(file_path), file_path.name)
                    else:
                        for root, dirs, files in os.walk(file_path):
                            for file in files:
                                file_full_path = Path(root) / file
                                arcname = str(file_full_path.relative_to(file_path.parent))
                                zf.write(str(file_full_path), arcname)
        
        return FileResponse(
            path=temp_file.name,
            filename="files.zip",
            media_type="application/zip"
        )
    except Exception as e:
        os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
async def delete_files(
    paths: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    path_list = paths.split(",")
    trash_service = TrashService(db)
    result = trash_service.move_to_trash(path_list)
    return result


@router.post("/move")
async def move_files(
    paths: str = Query(...),
    target: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    path_list = paths.split(",")
    file_service = FileService(db)
    
    try:
        result = file_service.move(path_list, target)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/copy")
async def copy_files(
    paths: str = Query(...),
    target: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    path_list = paths.split(",")
    file_service = FileService(db)
    
    try:
        result = file_service.copy(path_list, target)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/rename")
async def rename_file(
    path: str = Query(...),
    new_name: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    
    try:
        result = file_service.rename(None, path, new_name)
        return {"success": True, "file": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/info")
async def get_file_info(
    path: str = Query(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    
    try:
        info = file_service.get_file_info(path)
        return info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats")
async def get_stats(
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    return file_service.get_stats()


@router.get("/folders")
async def get_folders(
    path: str = Query(default=""),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_service = FileService(db)
    folders = file_service.get_folders_tree(path)
    return {"folders": folders}