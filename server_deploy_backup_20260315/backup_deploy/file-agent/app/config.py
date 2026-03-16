import json
import os
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


class Settings:
    def __init__(self):
        self._config = self._load_config()
        
    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def reload(self):
        self._config = self._load_config()
    
    def save(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    @property
    def storage(self) -> dict:
        return self._config.get("storage", {})
    
    @property
    def root_path(self) -> str:
        return self.storage.get("root", "F:\\MyFiles")
    
    @property
    def uploads_path(self) -> str:
        return self.storage.get("uploads", "F:\\MyFiles\\uploads")
    
    @property
    def trash_path(self) -> str:
        return self.storage.get("trash", "F:\\MyFiles\\.trash")
    
    @property
    def auth(self) -> dict:
        return self._config.get("auth", {})
    
    @property
    def password(self) -> str:
        return self.auth.get("password", "admin123")
    
    @property
    def session_secret(self) -> str:
        return self.auth.get("session_secret", "default-secret-change-me")
    
    @property
    def session_expire_hours(self) -> int:
        return self.auth.get("session_expire_hours", 24)
    
    @property
    def upload(self) -> dict:
        return self._config.get("upload", {})
    
    @property
    def max_upload_size_mb(self) -> int:
        return self.upload.get("max_size_mb", 2048)
    
    @property
    def search(self) -> dict:
        return self._config.get("search", {})
    
    @property
    def excluded_dirs(self) -> list:
        return self.search.get("excluded_dirs", [])
    
    @property
    def llm(self) -> dict:
        return self._config.get("llm", {})
    
    @property
    def llm_api_key(self) -> str:
        return self.llm.get("api_key", "")
    
    @property
    def llm_model(self) -> str:
        return self.llm.get("model", "qwen-plus")
    
    @property
    def llm_base_url(self) -> str:
        return self.llm.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    @property
    def natapp(self) -> dict:
        return self._config.get("natapp", {})
    
    @property
    def natapp_token(self) -> str:
        return self.natapp.get("token", "")
    
    @property
    def natapp_enabled(self) -> bool:
        return self.natapp.get("enabled", True)
    
    @property
    def server(self) -> dict:
        return self._config.get("server", {})
    
    @property
    def server_host(self) -> str:
        return self.server.get("host", "0.0.0.0")
    
    @property
    def server_port(self) -> int:
        return self.server.get("port", 5000)
    
    @property
    def cors(self) -> dict:
        return self._config.get("cors", {})
    
    @property
    def cors_origins(self) -> list:
        return self.cors.get("origins", ["*"])
    
    @property
    def email(self) -> dict:
        return self._config.get("email", {})
    
    @property
    def email_smtp_server(self) -> str:
        return self.email.get("smtp_server", "smtp.qq.com")
    
    @property
    def email_smtp_port(self) -> int:
        return self.email.get("smtp_port", 465)
    
    @property
    def email_sender(self) -> str:
        return self.email.get("sender", "")
    
    @property
    def email_password(self) -> str:
        return self.email.get("password", "")
    
    @property
    def mounts(self) -> list:
        return self.storage.get("mounts", [])
    
    def get_mount_by_path(self, path: str) -> dict:
        from pathlib import Path
        try:
            abs_path = str(Path(path).resolve())
            for mount in self.mounts:
                mount_path = str(Path(mount.get("path", "")).resolve())
                if abs_path.startswith(mount_path):
                    return mount
            return None
        except Exception:
            return None
    
    def is_readonly_path(self, path: str) -> bool:
        mount = self.get_mount_by_path(path)
        if mount:
            return mount.get("readonly", True)
        return False
    
    def get_all_search_roots(self) -> list:
        roots = [self.root_path]
        for mount in self.mounts:
            mount_path = mount.get("path", "")
            if mount_path and Path(mount_path).exists():
                roots.append(mount_path)
        return roots
    
    def update_password(self, new_password: str):
        if "auth" not in self._config:
            self._config["auth"] = {}
        self._config["auth"]["password"] = new_password
        self.save()
    
    def update_llm_api_key(self, api_key: str):
        if "llm" not in self._config:
            self._config["llm"] = {}
        self._config["llm"]["api_key"] = api_key
        self.save()
    
    def update_natapp_token(self, token: str):
        if "natapp" not in self._config:
            self._config["natapp"] = {}
        self._config["natapp"]["token"] = token
        self.save()
    
    def update_email(self, sender: str, password: str):
        if "email" not in self._config:
            self._config["email"] = {}
        self._config["email"]["sender"] = sender
        self._config["email"]["password"] = password
        self.save()
    
    def add_mount(self, name: str, path: str, icon: str = "📁", readonly: bool = True):
        if "storage" not in self._config:
            self._config["storage"] = {}
        if "mounts" not in self._config["storage"]:
            self._config["storage"]["mounts"] = []
        
        for mount in self._config["storage"]["mounts"]:
            if mount.get("path") == path:
                return False
        
        self._config["storage"]["mounts"].append({
            "name": name,
            "path": path,
            "icon": icon,
            "readonly": readonly
        })
        self.save()
        return True
    
    def remove_mount(self, path: str):
        if "storage" not in self._config:
            return False
        if "mounts" not in self._config["storage"]:
            return False
        
        mounts = self._config["storage"]["mounts"]
        for i, mount in enumerate(mounts):
            if mount.get("path") == path:
                mounts.pop(i)
                self.save()
                return True
        return False


settings = Settings()