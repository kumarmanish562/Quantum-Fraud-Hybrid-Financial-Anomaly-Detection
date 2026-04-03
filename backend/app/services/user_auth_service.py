from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import secrets
import random
from sqlalchemy.orm import Session
from app.models.user import User, APIKey

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-change-in-production-qfd-2024"
ALGORITHM = "HS256"

class UserAuthService:
    
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_otp() -> str:
        return str(random.randint(100000, 999999))
    
    @staticmethod
    def generate_api_key() -> str:
        return f"qfd_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def verify_token(token: str):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    @staticmethod
    def create_user(db: Session, email: str, password: str, full_name: str, company_name: str = None, phone: str = None):
        user = User(
            email=email,
            password_hash=UserAuthService.hash_password(password),
            full_name=full_name,
            company_name=company_name,
            phone=phone,
            otp_code=UserAuthService.generate_otp(),
            otp_expiry=datetime.utcnow() + timedelta(minutes=10)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def verify_otp(db: Session, email: str, otp: str):
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return False
        if user.otp_code == otp and user.otp_expiry > datetime.utcnow():
            user.is_verified = True
            user.is_active = True
            user.otp_code = None
            db.commit()
            return True
        return False
    
    @staticmethod
    def create_api_key(db: Session, user_id: str, name: str = "Default API Key"):
        api_key = APIKey(
            user_id=user_id,
            api_key=UserAuthService.generate_api_key(),
            name=name
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        return api_key
    
    @staticmethod
    def get_user_api_keys(db: Session, user_id: str):
        return db.query(APIKey).filter(APIKey.user_id == user_id).all()
    
    @staticmethod
    def regenerate_api_key(db: Session, api_key_id: str, user_id: str):
        api_key = db.query(APIKey).filter(APIKey.id == api_key_id, APIKey.user_id == user_id).first()
        if api_key:
            api_key.api_key = UserAuthService.generate_api_key()
            api_key.created_at = datetime.utcnow()
            db.commit()
            db.refresh(api_key)
            return api_key
        return None
