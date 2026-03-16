import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Trash, OperationLog


class TrashService:
    def __init__(self, db: Session):
        self.db = db
        self.trash_path = Path(settings.trash_path)
        self.root_path = Path(settings.root_path)
        self.trash_path.mkdir(parents=True, exist_ok=True)
    
    def _is_path_allowed(self, path: str) -> bool:
        try:
            abs_path = Path(path).resolve()
            root = self.root_path.resolve()
            return str(abs_path).startswith(str(root))
        except Exception:
            return False
    
    def _is_in_trash(self, path: str) -> bool:
        try:
            abs_path = Path(path).resolve()
            trash = self.trash_path.resolve()
            return str(abs_path).startswith(str(trash))
        except Exception:
            return False
    
    def move_to_trash(self, paths: List[str]) -> dict:
        moved = []
        failed = []
        
        for path in paths:
            try:
                src = Path(path)
                
                if not self._is_path_allowed(str(src)):
                    failed.append({"path": path, "error": "路径不允许访问"})
                    continue
                
                if not src.exists():
                    failed.append({"path": path, "error": "文件不存在"})
                    continue
                
                if self._is_in_trash(str(src)):
                    failed.append({"path": path, "error": "文件已在回收站中"})
                    continue
                
                trash_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{src.name}"
                trash_file = self.trash_path / trash_name
                
                stat = src.stat() if src.is_file() else None
                size = stat.st_size if stat else self._get_dir_size(src)
                
                shutil.move(str(src), str(trash_file))
                
                trash_item = Trash(
                    name=src.name,
                    original_path=str(src),
                    trash_path=str(trash_file),
                    is_dir=src.is_dir(),
                    size=size
                )
                self.db.add(trash_item)
                
                moved.append({
                    "name": src.name,
                    "original_path": str(src),
                    "trash_path": str(trash_file)
                })
                
            except Exception as e:
                failed.append({"path": path, "error": str(e)})
        
        self.db.commit()
        
        return {
            "moved": moved,
            "failed": failed,
            "total": len(paths),
            "success_count": len(moved),
            "failed_count": len(failed)
        }
    
    def list_trash(self, page: int = 1, page_size: int = 50) -> dict:
        query = self.db.query(Trash).order_by(Trash.deleted_at.desc())
        
        total = query.count()
        items = query.offset((page - 1) * page_size).limit(page_size).all()
        
        return {
            "items": [{
                "id": item.id,
                "name": item.name,
                "original_path": item.original_path,
                "trash_path": item.trash_path,
                "is_dir": item.is_dir,
                "size": item.size,
                "size_str": self._format_size(item.size) if item.size else "-",
                "deleted_at": item.deleted_at.isoformat() if item.deleted_at else None
            } for item in items],
            "total": total,
            "page": page,
            "page_size": page_size
        }
    
    def restore(self, ids: List[int]) -> dict:
        restored = []
        failed = []
        
        for item_id in ids:
            try:
                trash_item = self.db.query(Trash).filter(Trash.id == item_id).first()
                
                if not trash_item:
                    failed.append({"id": item_id, "error": "回收站记录不存在"})
                    continue
                
                trash_file = Path(trash_item.trash_path)
                
                if not trash_file.exists():
                    failed.append({"id": item_id, "error": "回收站文件不存在"})
                    self.db.delete(trash_item)
                    continue
                
                original_path = Path(trash_item.original_path)
                original_path.parent.mkdir(parents=True, exist_ok=True)
                
                if original_path.exists():
                    failed.append({"id": item_id, "error": "原位置已存在同名文件"})
                    continue
                
                shutil.move(str(trash_file), str(original_path))
                
                restored.append({
                    "id": item_id,
                    "name": trash_item.name,
                    "restored_path": str(original_path)
                })
                
                self.db.delete(trash_item)
                
            except Exception as e:
                failed.append({"id": item_id, "error": str(e)})
        
        self.db.commit()
        
        return {
            "restored": restored,
            "failed": failed,
            "total": len(ids),
            "success_count": len(restored),
            "failed_count": len(failed)
        }
    
    def delete_permanently(self, ids: List[int]) -> dict:
        deleted = []
        failed = []
        
        for item_id in ids:
            try:
                trash_item = self.db.query(Trash).filter(Trash.id == item_id).first()
                
                if not trash_item:
                    failed.append({"id": item_id, "error": "回收站记录不存在"})
                    continue
                
                trash_file = Path(trash_item.trash_path)
                
                if trash_file.exists():
                    if trash_file.is_dir():
                        shutil.rmtree(str(trash_file))
                    else:
                        trash_file.unlink()
                
                deleted.append({
                    "id": item_id,
                    "name": trash_item.name
                })
                
                self.db.delete(trash_item)
                
            except Exception as e:
                failed.append({"id": item_id, "error": str(e)})
        
        self.db.commit()
        
        return {
            "deleted": deleted,
            "failed": failed,
            "total": len(ids),
            "success_count": len(deleted),
            "failed_count": len(failed)
        }
    
    def empty_trash(self) -> dict:
        count = 0
        
        for item in self.db.query(Trash).all():
            try:
                trash_file = Path(item.trash_path)
                if trash_file.exists():
                    if trash_file.is_dir():
                        shutil.rmtree(str(trash_file))
                    else:
                        trash_file.unlink()
                self.db.delete(item)
                count += 1
            except Exception:
                pass
        
        self.db.commit()
        
        return {"cleared_count": count}
    
    def _get_dir_size(self, path: Path) -> int:
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total
    
    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024*1024):.1f} MB"
        else:
            return f"{size / (1024*1024*1024):.1f} GB"