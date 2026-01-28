"""
User model for authentication and authorization.
Defines the Users table with role-based access control.
"""

from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from database.connection import Base
import enum


class UserRole(str, enum.Enum):
    """Enumeration for user roles in the system."""
    ADMIN = "admin"
    USER = "user"


class User(Base):
    """
    User model for storing authentication information.
    
    Attributes:
        id: Primary key, auto-incremented
        username: Unique username for login
        password_hash: Bcrypt hashed password
        role: User role (admin or user)
        created_at: Timestamp of account creation
        updated_at: Timestamp of last update
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
