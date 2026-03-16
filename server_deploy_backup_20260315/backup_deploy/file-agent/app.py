import os
import sys
import socket
import subprocess
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.database import init_db
from routers import auth_router, files_router, search_router, preview_router, trash_router, agent_router, mounts_router

app = FastAPI(
    title="私人文件系统",
    description="智能文件管理系统，支持文件上传、下载、搜索、预览，集成了 LLM Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(auth_router)
app.include_router(files_router)
app.include_router(search_router)
app.include_router(preview_router)
app.include_router(trash_router)
app.include_router(agent_router)
app.include_router(mounts_router)


@app.on_event("startup")
async def startup_event():
    init_db()
    
    root_path = Path(settings.root_path)
    root_path.mkdir(parents=True, exist_ok=True)
    
    uploads_path = Path(settings.uploads_path)
    uploads_path.mkdir(parents=True, exist_ok=True)
    
    trash_path = Path(settings.trash_path)
    trash_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  私人文件系统 v1.0 已启动!")
    print(f"  本地访问:   http://localhost:{settings.server_port}")
    print(f"  API文档:    http://localhost:{settings.server_port}/docs")
    print(f"  文件根目录: {settings.root_path}")
    print(f"{'='*50}\n")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/mounts", response_class=HTMLResponse)
async def mounts_page(request: Request):
    return templates.TemplateResponse("mounts.html", {"request": request})


@app.get("/api/config")
async def get_config():
    return {
        "llm_configured": bool(settings.llm_api_key),
        "llm_model": settings.llm_model,
        "natapp_configured": bool(settings.natapp_token),
        "email_configured": bool(settings.email_sender and settings.email_password),
        "root_path": settings.root_path,
        "max_upload_size_mb": settings.max_upload_size_mb
    }


@app.post("/api/config")
async def update_config(request: Request):
    from app.config import settings as s
    
    data = await request.json()
    
    if "password" in data and data["password"]:
        s.update_password(data["password"])
    
    if "llm_api_key" in data and data["llm_api_key"]:
        s.update_llm_api_key(data["llm_api_key"])
    
    if "natapp_token" in data and data["natapp_token"]:
        s.update_natapp_token(data["natapp_token"])
    
    if "email_sender" in data and "email_password" in data:
        s.update_email(data["email_sender"], data["email_password"])
    
    s.reload()
    
    return {"success": True, "message": "配置已保存"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_natapp():
    if not settings.natapp_enabled or not settings.natapp_token:
        return None
    
    natapp_path = BASE_DIR / "natapp.exe"
    if not natapp_path.exists():
        print("[警告] natapp.exe 不存在，跳过内网穿透")
        return None
    
    try:
        process = subprocess.Popen(
            [str(natapp_path), "-authtoken=" + settings.natapp_token],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"[Natapp] 内网穿透已启动")
        return process
    except Exception as e:
        print(f"[警告] 启动 natapp 失败: {e}")
        return None


if __name__ == "__main__":
    natapp_process = start_natapp()
    
    try:
        uvicorn.run(
            app,
            host=settings.server_host,
            port=settings.server_port,
            log_level="info"
        )
    finally:
        if natapp_process:
            natapp_process.terminate()
            print("[Natapp] 已停止")
