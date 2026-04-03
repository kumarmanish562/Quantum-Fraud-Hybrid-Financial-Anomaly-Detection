from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from app.core.database import get_db
from app.services.user_auth_service import UserAuthService
from app.models.user import User, APIKey
from datetime import datetime, timedelta

router = APIRouter()

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    company_name: str = None
    phone: str = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

@router.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = UserAuthService.create_user(
        db, request.email, request.password, request.full_name, 
        request.company_name, request.phone
    )
    
    return {
        "message": "Registration successful. Please verify your email with OTP.",
        "user_id": user.id,
        "otp": user.otp_code
    }

@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest, db: Session = Depends(get_db)):
    if UserAuthService.verify_otp(db, request.email, request.otp):
        user = db.query(User).filter(User.email == request.email).first()
        api_key = UserAuthService.create_api_key(db, user.id)
        token = UserAuthService.create_access_token({"sub": user.email, "user_id": str(user.id)})
        
        return {
            "message": "Email verified successfully",
            "access_token": token,
            "token_type": "bearer",
            "api_key": api_key.api_key,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name
            }
        }
    raise HTTPException(status_code=400, detail="Invalid or expired OTP")

@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not UserAuthService.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Please verify your email first")
    
    token = UserAuthService.create_access_token({"sub": user.email, "user_id": str(user.id)})
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "company_name": user.company_name
        }
    }

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        return {"message": "If email exists, OTP has been sent"}
    
    user.otp_code = UserAuthService.generate_otp()
    user.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
    db.commit()
    
    return {
        "message": "OTP sent to your email",
        "otp": user.otp_code
    }

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.otp_code != request.otp or user.otp_expiry < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    user.password_hash = UserAuthService.hash_password(request.new_password)
    user.otp_code = None
    db.commit()
    
    return {"message": "Password reset successful"}

@router.get("/profile")
async def get_profile(token: str, db: Session = Depends(get_db)):
    payload = UserAuthService.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    api_keys = UserAuthService.get_user_api_keys(db, str(user.id))
    
    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "company_name": user.company_name,
            "phone": user.phone,
            "created_at": user.created_at.isoformat() if user.created_at else None
        },
        "api_keys": [
            {
                "id": str(key.id),
                "name": key.name,
                "api_key": key.api_key,
                "is_active": key.is_active,
                "requests_count": key.requests_count,
                "requests_limit": key.requests_limit,
                "created_at": key.created_at.isoformat() if key.created_at else None
            }
            for key in api_keys
        ]
    }

@router.post("/api-keys/create")
async def create_api_key(request: dict, token: str, db: Session = Depends(get_db)):
    payload = UserAuthService.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    key_name = request.get('name', 'API Key')
    new_key = UserAuthService.create_api_key(db, payload["user_id"], key_name)
    
    return {
        "message": "API key created successfully",
        "api_key": new_key.api_key,
        "key_id": new_key.id
    }

@router.delete("/api-keys/delete/{key_id}")
async def delete_api_key(key_id: str, token: str, db: Session = Depends(get_db)):
    payload = UserAuthService.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    api_key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == payload["user_id"]).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    db.delete(api_key)
    db.commit()
    
    return {"message": "API key deleted successfully"}
