import os
import fnmatch
from datetime import datetime
from typing import List, Optional, Callable
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings


class SearchService:
    def __init__(self, db: Session):
        self.db = db
        self.root_path = Path(settings.root_path)
        self.excluded_dirs = settings.excluded_dirs
    
    def _get_all_roots(self) -> List[Path]:
        roots = [self.root_path]
        for mount in settings.mounts:
            mount_path = Path(mount.get("path", ""))
            if mount_path.exists():
                roots.append(mount_path)
        return roots
    
    def _get_mount_name(self, path: str) -> Optional[str]:
        for mount in settings.mounts:
            if str(path).startswith(mount.get("path", "")):
                return mount.get("name", "")
        return None
    
    def _should_skip_dir(self, dirpath: str) -> bool:
        dirpath_lower = dirpath.lower()
        dirname = os.path.basename(dirpath).lower()
        
        for excl in self.excluded_dirs:
            excl_lower = excl.lower()
            if excl_lower in dirpath_lower:
                return True
            if fnmatch.fnmatch(dirname, excl_lower):
                return True
        return False
    
    def _format_size(self, size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024*1024):.1f} MB"
        else:
            return f"{size / (1024*1024*1024):.1f} GB"
    
    def search(self, keyword: str, file_types: Optional[List[str]] = None,
               max_results: int = 100, path: Optional[str] = None,
               progress_callback: Optional[Callable] = None,
               search_all_mounts: bool = False) -> List[dict]:
        if not keyword and not file_types:
            return []
        
        keywords = [k.lower() for k in keyword.split()] if keyword else []
        file_types_lower = set(ft.lower() for ft in file_types) if file_types else None
        
        search_roots = []
        if path and self._is_path_allowed(path):
            search_roots = [Path(path)]
        elif search_all_mounts:
            search_roots = self._get_all_roots()
        else:
            search_roots = [self.root_path]
        
        results = []
        scanned = 0
        
        for search_root in search_roots:
            for root, dirs, files in os.walk(search_root):
                if self._should_skip_dir(root):
                    dirs[:] = []
                    continue
                
                dirs[:] = [d for d in dirs if not self._should_skip_dir(os.path.join(root, d))]
                
                scanned += 1
                if progress_callback and scanned % 100 == 0:
                    progress_callback(f"正在扫描: {root}", len(results))
                
                for filename in files:
                    if len(results) >= max_results:
                        break
                    
                    ext = os.path.splitext(filename)[1].lower()
                    
                    if file_types_lower and ext not in file_types_lower:
                        continue
                    
                    if keywords:
                        filename_lower = filename.lower()
                        root_lower = root.lower()
                        matched = any(kw in filename_lower or kw in root_lower for kw in keywords)
                        if not matched:
                            continue
                    
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        
                        score = 0
                        for kw in keywords:
                            if kw in filename_lower:
                                score += 10
                            elif kw in root_lower:
                                score += 3
                        
                        results.append({
                            "name": filename,
                            "path": filepath,
                            "parent_path": root,
                            "is_dir": False,
                            "size": stat.st_size,
                            "size_str": self._format_size(stat.st_size),
                            "ext": ext,
                            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "mount_name": self._get_mount_name(filepath),
                            "_score": score
                        })
                    except (PermissionError, OSError, FileNotFoundError):
                        continue
        
        results.sort(key=lambda x: (-x.get("_score", 0), x["name"]))
        for r in results:
            r.pop("_score", None)
        
        return results[:max_results]
    
    def search_by_type(self, file_type: str, max_results: int = 100) -> List[dict]:
        type_map = {
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico"],
            "video": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv", ".webm"],
            "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"],
            "document": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md"],
            "archive": [".zip", ".rar", ".7z", ".tar", ".gz"],
            "code": [".py", ".js", ".ts", ".html", ".css", ".json", ".xml", ".java", ".cpp", ".c", ".h"],
        }
        
        file_types = type_map.get(file_type.lower(), [f".{file_type.lower()}"])
        return self.search("", file_types=file_types, max_results=max_results)
    
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