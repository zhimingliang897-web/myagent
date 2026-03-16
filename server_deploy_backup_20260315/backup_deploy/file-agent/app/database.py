import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR / 'files.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False)
    path = Column(String(1000), unique=True, nullable=False)
    parent_path = Column(String(1000))
    is_dir = Column(Boolean, default=False)
    size = Column(Integer, default=0)
    ext = Column(String(50))
    mime_type = Column(String(100))
    is_starred = Column(Boolean, default=False)
    thumbnail = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime)
    indexed_at = Column(DateTime, default=datetime.utcnow)


class Trash(Base):
    __tablename__ = "trash"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False)
    original_path = Column(String(1000), nullable=False)
    trash_path = Column(String(1000), nullable=False)
    is_dir = Column(Boolean, default=False)
    size = Column(Integer, default=0)
    deleted_at = Column(DateTime, default=datetime.utcnow)


class OperationLog(Base):
    __tablename__ = "operation_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(50), nullable=False)
    file_path = Column(String(1000))
    details = Column(Text)
    ip_address = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class SettingsDB(Base):
    __tablename__ = "settings"
    
    key = Column(String(100), primary_key=True)
    value = Column(Text)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()