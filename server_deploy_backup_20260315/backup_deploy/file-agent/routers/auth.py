from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from pydantic import BaseModel
from typing import Optional
from app.security import verify_password, create_session_token
from app.deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["认证"])


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="密码错误")
    
    token = create_session_token()
    
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax"
    )
    
    return LoginResponse(success=True, message="登录成功")


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("session_token")
    return {"success": True, "message": "已退出登录"}


@router.get("/check")
async def check_auth(user: str = Depends(get_current_user)):
    return {"authenticated": True, "user": user}