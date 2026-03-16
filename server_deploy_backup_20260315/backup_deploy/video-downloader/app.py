#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21视频下载器 - FastAPI后端服务
提供REST API供Web界面和浏览器插件调用
"""

import os
import sys
import threading
from typing import Optional, List, Dict

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 导入核心模块
from core.downloader import VideoDownloaderCore

# 路径固定到项目根目录（不受启动目录影响）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
downloader = VideoDownloaderCore(
    download_dir=os.path.join(_project_root, "downloads"),
    cookies_file=os.path.join(_project_root, "cookies.txt")
)


# 创建FastAPI应用
app = FastAPI(
    title="21视频下载器",
    description="支持B站、NTU课程网站等多种视频平台下载",
    version="1.0.0"
)

# 配置CORS - 允许浏览器插件跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件和模板
base_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.dirname(base_dir)
static_dir = os.path.join(base_dir, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))


# ==================== 数据模型 ====================

class AddTaskRequest(BaseModel):
    """添加下载任务请求"""
    url: str
    source: Optional[str] = "web"

class ConfigRequest(BaseModel):
    """配置请求"""
    cookies: Optional[str] = None
    download_dir: Optional[str] = None

class CookieCaptureRequest(BaseModel):
    """从浏览器自动捕获的Cookie请求"""
    cookies: List[Dict]


# ==================== API路由 ====================

@app.get("/")
async def root(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """健康检查 - 用于浏览器插件检测服务是否在线"""
    return {"status": "ok", "service": "21Video Downloader"}


@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    cookies_content = downloader.load_cookies()
    return {
        "download_dir": downloader.download_dir,
        "cookies_exists": cookies_content is not None,
        "cookies_length": len(cookies_content) if cookies_content else 0
    }


@app.post("/api/config")
async def update_config(config: ConfigRequest):
    """更新配置"""
    if config.download_dir:
        downloader.download_dir = config.download_dir
        downloader._ensure_download_dir()
    
    if config.cookies:
        success = downloader.save_cookies(config.cookies)
        if success:
            return {"status": "ok", "message": "配置已保存"}
        else:
            raise HTTPException(status_code=500, detail="保存Cookie失败")
    
    return {"status": "ok", "message": "配置已更新"}


@app.post("/api/cookies/capture")
async def capture_cookies(request: CookieCaptureRequest):
    """接收从浏览器扩展自动获取的Cookie，转为Netscape格式保存"""
    lines = [
        "# Netscape HTTP Cookie File\n",
        "# Auto-captured by 21视频下载器\n\n"
    ]
    for c in request.cookies:
        domain = c.get("domain", "")
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = str(int(c.get("expirationDate", 0)))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{include_sub}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

    success = downloader.save_cookies("".join(lines))
    if success:
        return {"status": "ok", "message": f"已保存 {len(request.cookies)} 个Cookie", "count": len(request.cookies)}
    raise HTTPException(status_code=500, detail="保存Cookie失败")


@app.post("/api/cookies/upload")
async def upload_cookies_file(file: UploadFile = File(...)):
    """上传 J2Team / EditThisCookie 导出的 JSON 文件，自动转为 Netscape 格式保存"""
    content = await file.read()
    try:
        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="文件不是有效的 JSON 格式")

    # J2Team 格式：{"url": "...", "cookies": [...]}  或直接 [...]
    if isinstance(data, dict) and "cookies" in data:
        cookies = data["cookies"]
    elif isinstance(data, list):
        cookies = data
    else:
        raise HTTPException(status_code=400, detail="无法识别的 Cookie JSON 格式")

    lines = ["# Netscape HTTP Cookie File\n# Converted from J2Team export\n\n"]
    for c in cookies:
        domain = c.get("domain", "")
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = str(int(c.get("expirationDate", 0)))
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{include_sub}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

    success = downloader.save_cookies("".join(lines))
    if success:
        return {"status": "ok", "message": f"已保存 {len(cookies)} 个 Cookie", "count": len(cookies)}
    raise HTTPException(status_code=500, detail="保存失败")


@app.post("/api/add_task")
async def add_task(request: AddTaskRequest):
    """添加下载任务"""
    if not request.url:
        raise HTTPException(status_code=400, detail="URL不能为空")

    # 创建任务（会进行 URL 支持检查）
    task_id, warning = downloader.create_task(request.url)

    # 后台启动下载线程
    def run_download():
        downloader.start_download(task_id)

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()

    response = {
        "status": "ok",
        "task_id": task_id,
        "message": "任务已添加"
    }

    # 如果有警告（不支持的 URL 类型），返回给前端
    if warning:
        response["warning"] = warning

    return response


@app.get("/api/tasks")
async def get_tasks():
    """获取所有任务"""
    return {"tasks": downloader.get_all_tasks()}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """获取单个任务状态"""
    task = downloader.get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    success = downloader.cancel_task(task_id)
    if success:
        return {"status": "ok", "message": "任务已取消"}
    raise HTTPException(status_code=404, detail="任务不存在或无法取消")


@app.delete("/api/tasks")
async def clear_tasks():
    """清理已完成的任务"""
    downloader.clear_completed_tasks()
    return {"status": "ok", "message": "已完成任务已清理"}


@app.get("/api/dependencies")
async def check_dependencies():
    """检查系统依赖"""
    deps = downloader.check_dependencies()
    all_ok = all(deps.values())
    return {
        "dependencies": deps,
        "all_ok": all_ok
    }


# ==================== 主程序 ====================

def run_server(host: str = "0.0.0.0", port: int = int(os.environ.get("PORT", 5005))):
    """运行服务器"""
    print(f"""
    ╔═══════════════════════════════════════════╗
    ║         21 视频下载器 Web服务             ║
    ║    访问地址: http://{host}:{port}         ║
    ╚═══════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
