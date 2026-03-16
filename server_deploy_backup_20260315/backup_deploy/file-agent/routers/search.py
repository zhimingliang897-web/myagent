from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from services.search_service import SearchService

router = APIRouter(prefix="/api/search", tags=["搜索"])


@router.get("")
async def search_files(
    keyword: str = Query(default=""),
    file_types: Optional[str] = Query(default=None),
    max_results: int = Query(default=50, ge=1, le=500),
    path: Optional[str] = Query(default=None),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    search_service = SearchService(db)
    
    types_list = file_types.split(",") if file_types else None
    
    results = search_service.search(
        keyword=keyword,
        file_types=types_list,
        max_results=max_results,
        path=path
    )
    
    return {
        "results": results,
        "total": len(results),
        "keyword": keyword
    }


@router.get("/type/{file_type}")
async def search_by_type(
    file_type: str,
    max_results: int = Query(default=50, ge=1, le=500),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    search_service = SearchService(db)
    results = search_service.search_by_type(file_type, max_results)
    
    return {
        "results": results,
        "total": len(results),
        "type": file_type
    }