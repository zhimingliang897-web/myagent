import os
import shutil
import mimetypes
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.database import File, OperationLog


class FileService:
    def __init__(self, db: Session):
        self.db = db
        self.root_path = Path(settings.root_path)
    
    def _get_mime_type(self, filepath: str) -> Optional[str]:
        mime_type, _ = mimetypes.guess_type(filepath)
        return mime_type
    
    def _is_path_allowed(self, path: str) -> bool:
        try:
            abs_path = Path(path).resolve()
            root = self.root_path.resolve()
            if str(abs_path).startswith(str(root)):
                return True
            for mount in settings.mounts:
                mount_path = Path(mount.get("path", "")).resolve()
                if str(abs_path).startswith(str(mount_path)):
                    return True
            return False
        except Exception:
            return False
    
    def _is_readonly(self, path: str) -> bool:
        return settings.is_readonly_path(path)
    
    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024*1024):.1f} MB"
        else:
            return f"{size / (1024*1024*1024):.1f} GB"
    
    def list_files(self, path: str = "", page: int = 1, page_size: int = 50,
                   sort_by: str = "name", sort_order: str = "asc") -> Tuple[List[dict], int, str]:
        if not path:
            target_path = self.root_path
        else:
            target_path = Path(path)
            if not self._is_path_allowed(str(target_path)):
                return [], 0, str(self.root_path)
        
        if not target_path.exists():
            return [], 0, str(self.root_path)
        
        dirs = []
        files = []
        
        try:
            for entry in target_path.iterdir():
                if entry.name.startswith('.') and entry.name != '.trash':
                    continue
                
                stat = entry.stat()
                item = {
                    "name": entry.name,
                    "path": str(entry),
                    "parent_path": str(entry.parent),
                    "is_dir": entry.is_dir(),
                    "size": stat.st_size if entry.is_file() else 0,
                    "ext": entry.suffix.lower() if entry.is_file() else None,
                    "mime_type": self._get_mime_type(str(entry)) if entry.is_file() else None,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime),
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                }
                
                if entry.is_dir():
                    dirs.append(item)
                else:
                    files.append(item)
        except PermissionError:
            pass
        
        reverse = sort_order == "desc"
        
        if sort_by == "name":
            dirs.sort(key=lambda x: x["name"].lower(), reverse=reverse)
            files.sort(key=lambda x: x["name"].lower(), reverse=reverse)
        elif sort_by == "size":
            dirs.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["size"], reverse=reverse)
        elif sort_by == "modified":
            dirs.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["modified_at"], reverse=reverse)
        else:
            dirs.sort(key=lambda x: x["name"].lower())
            files.sort(key=lambda x: x["name"].lower())
        
        all_items = dirs + files
        total = len(all_items)
        
        start = (page - 1) * page_size
        end = start + page_size
        paginated_items = all_items[start:end]
        
        return paginated_items, total, str(target_path)
    
    def create_folder(self, name: str, parent_path: str) -> dict:
        if not parent_path:
            parent_path = str(self.root_path)
        
        parent = Path(parent_path)
        if not self._is_path_allowed(str(parent)):
            raise ValueError("路径不允许访问")
        
        if self._is_readonly(str(parent)):
            raise ValueError("挂载目录为只读，不允许创建文件夹")
        
        new_folder = parent / name
        if new_folder.exists():
            raise ValueError("文件夹已存在")
        
        new_folder.mkdir(parents=True, exist_ok=True)
        
        return {
            "name": name,
            "path": str(new_folder),
            "is_dir": True,
            "parent_path": str(parent)
        }
    
    def rename(self, file_id: Optional[int], old_path: str, new_name: str) -> dict:
        old = Path(old_path)
        if not self._is_path_allowed(str(old)):
            raise ValueError("路径不允许访问")
        
        if self._is_readonly(str(old)):
            raise ValueError("挂载目录为只读，不允许重命名")
        
        if not old.exists():
            raise ValueError("文件不存在")
        
        new_path = old.parent / new_name
        if new_path.exists():
            raise ValueError("目标名称已存在")
        
        old.rename(new_path)
        
        self._log_operation("rename", str(old), f"renamed to {new_name}")
        
        return {
            "name": new_name,
            "path": str(new_path),
            "old_path": str(old)
        }
    
    def move(self, file_paths: List[str], target_path: str) -> dict:
        target = Path(target_path)
        if not self._is_path_allowed(str(target)):
            raise ValueError("目标路径不允许访问")
        
        if self._is_readonly(str(target)):
            raise ValueError("挂载目录为只读，不允许移动文件到此")
        
        if not target.exists() or not target.is_dir():
            raise ValueError("目标路径不存在或不是文件夹")
        
        moved = []
        for fp in file_paths:
            src = Path(fp)
            if not self._is_path_allowed(str(src)):
                continue
            
            if self._is_readonly(str(src)):
                continue
            
            if not src.exists():
                continue
            
            dst = target / src.name
            if dst.exists():
                continue
            
            try:
                shutil.move(str(src), str(dst))
                moved.append({
                    "name": src.name,
                    "old_path": str(src),
                    "new_path": str(dst)
                })
                self._log_operation("move", str(src), f"moved to {target_path}")
            except Exception:
                pass
        
        return {"moved": moved, "count": len(moved)}
    
    def copy(self, file_paths: List[str], target_path: str) -> dict:
        target = Path(target_path)
        if not self._is_path_allowed(str(target)):
            raise ValueError("目标路径不允许访问")
        
        if self._is_readonly(str(target)):
            raise ValueError("挂载目录为只读，不允许复制文件到此")
        
        if not target.exists() or not target.is_dir():
            raise ValueError("目标路径不存在或不是文件夹")
        
        copied = []
        for fp in file_paths:
            src = Path(fp)
            if not self._is_path_allowed(str(src)):
                continue
            
            if not src.exists():
                continue
            
            dst = target / src.name
            if dst.exists():
                base = src.stem
                ext = src.suffix
                counter = 1
                while dst.exists():
                    dst = target / f"{base} ({counter}){ext}"
                    counter += 1
            
            try:
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
                
                copied.append({
                    "name": src.name,
                    "old_path": str(src),
                    "new_path": str(dst)
                })
                self._log_operation("copy", str(src), f"copied to {target_path}")
            except Exception:
                pass
        
        return {"copied": copied, "count": len(copied)}
    
    def get_file_info(self, path: str) -> dict:
        file_path = Path(path)
        if not self._is_path_allowed(str(file_path)):
            raise ValueError("路径不允许访问")
        
        if not file_path.exists():
            raise ValueError("文件不存在")
        
        stat = file_path.stat()
        
        return {
            "name": file_path.name,
            "path": str(file_path),
            "parent_path": str(file_path.parent),
            "is_dir": file_path.is_dir(),
            "size": stat.st_size if file_path.is_file() else 0,
            "size_str": self._format_size(stat.st_size) if file_path.is_file() else "-",
            "ext": file_path.suffix.lower() if file_path.is_file() else None,
            "mime_type": self._get_mime_type(str(file_path)) if file_path.is_file() else None,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }
    
    def get_breadcrumb(self, path: str) -> List[dict]:
        if not path:
            return [{"name": "MyFiles", "path": str(self.root_path)}]
        
        current = Path(path)
        if not self._is_path_allowed(str(current)):
            return [{"name": "MyFiles", "path": str(self.root_path)}]
        
        parts = []
        while self._is_path_allowed(str(current)) and str(current) != str(self.root_path.parent):
            parts.insert(0, {
                "name": current.name if current != self.root_path else "MyFiles",
                "path": str(current)
            })
            current = current.parent
            if current == self.root_path.parent:
                break
        
        parts.insert(0, {"name": "MyFiles", "path": str(self.root_path)})
        return parts
    
    def get_stats(self) -> dict:
        total_files = 0
        total_dirs = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            if root_path.name.startswith('.') and root_path.name != '.trash':
                continue
            
            total_dirs += len(dirs)
            total_files += len(files)
            
            for f in files:
                try:
                    total_size += (root_path / f).stat().st_size
                except:
                    pass
        
        return {
            "total_files": total_files,
            "total_dirs": total_dirs,
            "total_size": total_size,
            "total_size_str": self._format_size(total_size)
        }
    
    def get_folders_tree(self, path: str = "") -> List[dict]:
        if not path:
            target_path = self.root_path
        else:
            target_path = Path(path)
            if not self._is_path_allowed(str(target_path)):
                return []
        
        if not target_path.exists():
            return []
        
        result = []
        try:
            for entry in sorted(target_path.iterdir(), key=lambda x: x.name.lower()):
                if not entry.is_dir():
                    continue
                if entry.name.startswith('.'):
                    continue
                
                item = {
                    "name": entry.name,
                    "path": str(entry),
                    "children": []
                }
                
                try:
                    children = self.get_folders_tree(str(entry))
                    if children:
                        item["children"] = children
                except:
                    pass
                
                result.append(item)
        except PermissionError:
            pass
        
        return result
    
    def _log_operation(self, action: str, file_path: str, details: str = None, ip: str = None):
        log = OperationLog(
            action=action,
            file_path=file_path,
            details=details,
            ip_address=ip
        )
        self.db.add(log)
        self.db.commit()