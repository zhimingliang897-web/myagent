import json
import os
import smtplib
import zipfile
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
import httpx
from openai import OpenAI

from app.config import settings


global_search_progress = {
    "status": "idle",
    "message": "",
    "found_count": 0
}


SYSTEM_PROMPT = """你是私人文件系统的智能助手，帮助用户管理文件。请用JSON格式输出，不要输出其他内容。

支持的文件操作：
1. 搜索文件：{"action":"search","search_params":{"keyword":"搜索词","file_types":[".pdf"],"max_results":20,"search_all":true}}
2. 浏览目录：{"action":"browse","browse_params":{"path":"F:\\\\MyFiles"}}
3. 上传文件：{"action":"upload","upload_params":{"target_path":"F:\\\\MyFiles\\\\文档"}}
4. 删除文件：{"action":"delete","delete_params":{"paths":["路径1","路径2"]}}
5. 移动文件：{"action":"move","move_params":{"paths":["路径1"],"target":"目标路径"}}
6. 重命名：{"action":"rename","rename_params":{"path":"原路径","new_name":"新名称"}}
7. 新建文件夹：{"action":"create_folder","folder_params":{"name":"文件夹名","parent_path":"父路径"}}
8. 发邮件：{"action":"email","email_params":{"recipient":"xxx@qq.com","paths":["文件路径"]}}
9. 普通聊天：{"action":"chat","reply":"回复内容"}

挂载点（只读目录）：
- 主目录MyFiles：可读可写
- 其他挂载目录：只读，只能浏览和搜索，不能修改

规则：
- 搜索时使用多同义词提高命中率，如"简历"->["简历","resume","cv"]
- search_all为true时搜索所有挂载点
- 图片类型：[".jpg",".png",".jpeg",".gif",".webp",".bmp"]
- 文档类型：[".pdf",".doc",".docx",".xls",".xlsx",".ppt",".pptx",".txt",".md"]
- 视频类型：[".mp4",".avi",".mov",".mkv",".webm"]
- 音频类型：[".mp3",".wav",".flac",".aac"]
- 路径使用双反斜杠转义，如 "F:\\\\MyFiles\\\\文档"
- 只输出JSON，不要任何其他内容"""


class AgentService:
    def __init__(self, db: Session):
        self.db = db
        self.root_path = Path(settings.root_path)
        self.last_results: List[dict] = []
        
        if settings.llm_api_key:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=httpx.Timeout(60, connect=10)
            )
        else:
            self.client = None
    
    def chat(self, user_message: str, context: Optional[dict] = None) -> dict:
        global global_search_progress
        
        if not self.client:
            return {
                "reply": "LLM API 未配置，请在设置中配置 API Key",
                "action": "chat",
                "files": [],
                "result": None
            }
        
        global_search_progress["status"] = "analyzing"
        global_search_progress["message"] = "正在理解您的需求..."
        global_search_progress["found_count"] = 0
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        context_msg = ""
        if self.last_results:
            file_list = ", ".join([f"[{i}]{r['name']}" for i, r in enumerate(self.last_results[:5])])
            context_msg = f"[上次搜索结果: {file_list}]\n\n"
        
        if context and context.get("current_path"):
            context_msg += f"[当前目录: {context['current_path']}]\n\n"
        
        messages.append({"role": "user", "content": context_msg + user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            global_search_progress["status"] = "idle"
            return {
                "reply": f"AI 服务暂时不可用: {str(e)[:100]}",
                "action": "chat",
                "files": [],
                "result": None
            }
        
        parsed = self._parse_response(raw)
        action = parsed.get("action", "chat")
        
        result = None
        files = []
        reply = ""
        
        if action == "search":
            files, reply = self._execute_search(parsed.get("search_params", {}))
            self.last_results = files
        elif action == "browse":
            result, reply = self._execute_browse(parsed.get("browse_params", {}))
        elif action == "delete":
            result, reply = self._execute_delete(parsed.get("delete_params", {}))
        elif action == "move":
            result, reply = self._execute_move(parsed.get("move_params", {}))
        elif action == "rename":
            result, reply = self._execute_rename(parsed.get("rename_params", {}))
        elif action == "create_folder":
            result, reply = self._execute_create_folder(parsed.get("folder_params", {}))
        elif action == "email":
            result, reply = self._execute_email(parsed.get("email_params", {}))
        else:
            reply = parsed.get("reply", "我可以帮你管理文件，比如搜索、移动、删除等操作。")
        
        global_search_progress["status"] = "idle"
        
        return {
            "reply": reply,
            "action": action,
            "files": files,
            "result": result
        }
    
    def _parse_response(self, raw: str) -> dict:
        raw = raw.strip()
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            if len(lines) > 2:
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            raw = raw.strip()
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        start = raw.find("{")
        if start != -1:
            depth = 0
            end = start
            for i, c in enumerate(raw[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        
        return {"action": "chat", "reply": raw if raw else "抱歉，我没理解您的意思"}
    
    def _execute_search(self, params: dict) -> tuple:
        from services.search_service import SearchService
        
        keyword = params.get("keyword", "")
        file_types = params.get("file_types")
        max_results = params.get("max_results", 20)
        search_all = params.get("search_all", False)
        
        search_service = SearchService(self.db)
        
        global_search_progress["status"] = "searching"
        global_search_progress["message"] = f"正在搜索: {keyword or '全部文件'}..."
        
        def on_progress(msg, count):
            global_search_progress["message"] = msg
            global_search_progress["found_count"] = count
        
        files = search_service.search(
            keyword=keyword,
            file_types=file_types,
            max_results=max_results,
            progress_callback=on_progress,
            search_all_mounts=search_all
        )
        
        if files:
            reply = f"找到 {len(files)} 个文件"
        else:
            reply = f"没有找到匹配的文件"
        
        return files, reply
    
    def _execute_browse(self, params: dict) -> tuple:
        from services.file_service import FileService
        
        path = params.get("path", "")
        
        file_service = FileService(self.db)
        
        if not path:
            path = str(self.root_path)
        
        files, total, current_path = file_service.list_files(path=path)
        
        result = {
            "path": current_path,
            "files": files,
            "total": total
        }
        
        dirs_count = sum(1 for f in files if f["is_dir"])
        files_count = len(files) - dirs_count
        
        reply = f"📁 {current_path}\n{dirs_count} 个文件夹，{files_count} 个文件"
        
        return result, reply
    
    def _execute_delete(self, params: dict) -> tuple:
        from services.trash_service import TrashService
        
        paths = params.get("paths", [])
        
        if not paths:
            return None, "请指定要删除的文件"
        
        trash_service = TrashService(self.db)
        result = trash_service.move_to_trash(paths)
        
        if result["success_count"] > 0:
            reply = f"已将 {result['success_count']} 个文件移入回收站"
        else:
            reply = "删除失败，请检查文件是否存在"
        
        return result, reply
    
    def _execute_move(self, params: dict) -> tuple:
        from services.file_service import FileService
        
        paths = params.get("paths", [])
        target = params.get("target", "")
        
        if not paths or not target:
            return None, "请指定要移动的文件和目标路径"
        
        file_service = FileService(self.db)
        result = file_service.move(paths, target)
        
        if result["count"] > 0:
            reply = f"已移动 {result['count']} 个文件到 {target}"
        else:
            reply = "移动失败，请检查路径是否正确"
        
        return result, reply
    
    def _execute_rename(self, params: dict) -> tuple:
        from services.file_service import FileService
        
        path = params.get("path", "")
        new_name = params.get("new_name", "")
        
        if not path or not new_name:
            return None, "请指定原文件路径和新名称"
        
        file_service = FileService(self.db)
        
        try:
            result = file_service.rename(None, path, new_name)
            reply = f"已重命名为: {new_name}"
        except Exception as e:
            result = None
            reply = f"重命名失败: {str(e)}"
        
        return result, reply
    
    def _execute_create_folder(self, params: dict) -> tuple:
        from services.file_service import FileService
        
        name = params.get("name", "")
        parent_path = params.get("parent_path", str(self.root_path))
        
        if not name:
            return None, "请指定文件夹名称"
        
        file_service = FileService(self.db)
        
        try:
            result = file_service.create_folder(name, parent_path)
            reply = f"已创建文件夹: {name}"
        except Exception as e:
            result = None
            reply = f"创建失败: {str(e)}"
        
        return result, reply
    
    def _execute_email(self, params: dict) -> tuple:
        recipient = params.get("recipient", "")
        paths = params.get("paths", [])
        
        if not recipient:
            return None, "请指定收件人邮箱"
        
        if not paths and self.last_results:
            paths = [f["path"] for f in self.last_results[:5]]
        
        if not paths:
            return None, "请指定要发送的文件"
        
        if not settings.email_sender or not settings.email_password:
            return None, "邮箱未配置，请在设置中配置邮箱"
        
        try:
            success, info = self._send_email(recipient, paths)
            return {"success": success, "info": info}, info
        except Exception as e:
            return None, f"发送失败: {str(e)}"
    
    def _send_email(self, recipient: str, file_paths: List[str]) -> tuple:
        msg = MIMEMultipart()
        msg['From'] = settings.email_sender
        msg['To'] = recipient
        msg['Subject'] = f"文件发送 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        body = f"共发送 {len(file_paths)} 个文件"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=os.path.basename(file_path)
                )
                msg.attach(part)
        
        with smtplib.SMTP_SSL(settings.email_smtp_server, settings.email_smtp_port) as server:
            server.login(settings.email_sender, settings.email_password)
            server.send_message(msg)
        
        return True, f"已发送 {len(file_paths)} 个文件到 {recipient}"
    
    def get_progress(self) -> dict:
        return global_search_progress.copy()