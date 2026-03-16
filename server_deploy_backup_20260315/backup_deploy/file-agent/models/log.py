from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LogBase(BaseModel):
    action: str
    file_path: Optional[str] = None
    details: Optional[str] = None


class LogResponse(LogBase):
    id: int
    ip_address: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True