from datetime import datetime, timedelta
from typing import Optional
import uuid
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer

from app.schemas.auth import User, UserCreate, TokenData
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

class AuthService:
    def __init__(self):
        # In-memory user storage for demo (replace with database in production)
        self.users = {}
        self._create_demo_user()
    
    def _create_demo_user(self):
        """Create a demo user for testing"""
        demo_user = User(
            id=str(uuid.uuid4()),
            username="demo",
            email="demo@example.com",
            full_name="Demo User",
            is_active=True,
            created_at=datetime.utcnow()
        )
        # Store with hashed password
        self.users["demo"] = {
            "user": demo_user,
            "hashed_password": self.get_password_hash("demo123")
        }
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    async def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_data = self.users.get(username)
        return user_data["user"] if user_data else None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user_data = self.users.get(username)
        if not user_data:
            return None
        
        if not self.verify_password(password, user_data["hashed_password"]):
            return None
        
        # Update last login
        user_data["user"].last_login = datetime.utcnow()
        return user_data["user"]
    
    async def create_user(self, user: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        if user.username in self.users:
            raise ValueError("Username already registered")
        
        # Check if email already exists
        for existing_user_data in self.users.values():
            if existing_user_data["user"].email == user.email:
                raise ValueError("Email already registered")
        
        # Create new user
        new_user = User(
            id=str(uuid.uuid4()),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        # Store user with hashed password
        self.users[user.username] = {
            "user": new_user,
            "hashed_password": self.get_password_hash(user.password)
        }
        
        return new_user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """Get current user from JWT token"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
        except JWTError:
            raise credentials_exception
        
        user = await self.get_user(username=token_data.username)
        if user is None:
            raise credentials_exception
        
        return user