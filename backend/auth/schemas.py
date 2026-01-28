"""
Pydantic schemas for authentication requests and responses.
Defines data validation models for the auth module.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class RoleEnum(str, Enum):
    """User role enumeration for schema validation."""
    ADMIN = "admin"
    USER = "user"


class UserCreate(BaseModel):
    """Schema for user registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=6, max_length=100, description="User password")
    role: RoleEnum = Field(default=RoleEnum.USER, description="User role")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123",
                "role": "user"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123"
            }
        }


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class TokenData(BaseModel):
    """Schema for token payload data."""
    username: Optional[str] = None
    role: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user data response."""
    id: int
    username: str
    role: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Schema for simple message responses."""
    message: str
    success: bool = True
