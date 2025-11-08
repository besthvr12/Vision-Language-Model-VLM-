"""
User authentication and management system.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import secrets
from enum import Enum


# Configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class User(BaseModel):
    user_id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    disabled: bool = False
    created_at: datetime = datetime.utcnow()
    preferences: Dict = {}


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserPreferences(BaseModel):
    """User personalization preferences."""
    favorite_categories: List[str] = []
    favorite_colors: List[str] = []
    preferred_gender: Optional[str] = None
    price_range_min: Optional[float] = None
    price_range_max: Optional[float] = None
    preferred_brands: List[str] = []
    size_preferences: Dict[str, str] = {}


class UserInteraction(BaseModel):
    """Track user interactions for personalization."""
    user_id: str
    product_id: str
    interaction_type: str  # view, click, add_to_cart, purchase
    timestamp: datetime = datetime.utcnow()
    session_id: Optional[str] = None
    metadata: Dict = {}


class AuthManager:
    """Manages user authentication and authorization."""

    def __init__(self):
        self.users_db: Dict[str, UserInDB] = {}
        self.user_interactions: List[UserInteraction] = []

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user from database."""
        return self.users_db.get(username)

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            return TokenData(username=username)
        except JWTError:
            return None

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user."""
        hashed_password = self.get_password_hash(password)
        user_id = f"user_{len(self.users_db) + 1}"

        user = UserInDB(
            user_id=user_id,
            email=email,
            username=username,
            full_name=full_name,
            role=role,
            hashed_password=hashed_password
        )

        self.users_db[username] = user
        return User(**user.dict(exclude={"hashed_password"}))

    def update_preferences(
        self,
        username: str,
        preferences: UserPreferences
    ) -> Optional[User]:
        """Update user preferences."""
        user = self.get_user(username)
        if not user:
            return None

        user.preferences = preferences.dict()
        return User(**user.dict(exclude={"hashed_password"}))

    def get_preferences(self, username: str) -> Optional[UserPreferences]:
        """Get user preferences."""
        user = self.get_user(username)
        if not user:
            return None

        return UserPreferences(**user.preferences) if user.preferences else UserPreferences()

    def track_interaction(self, interaction: UserInteraction):
        """Track user interaction for personalization."""
        self.user_interactions.append(interaction)

    def get_user_history(
        self,
        user_id: str,
        interaction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[UserInteraction]:
        """Get user interaction history."""
        interactions = [
            i for i in self.user_interactions
            if i.user_id == user_id
        ]

        if interaction_type:
            interactions = [
                i for i in interactions
                if i.interaction_type == interaction_type
            ]

        # Sort by timestamp (most recent first)
        interactions.sort(key=lambda x: x.timestamp, reverse=True)

        return interactions[:limit]

    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics."""
        interactions = [i for i in self.user_interactions if i.user_id == user_id]

        stats = {
            "total_interactions": len(interactions),
            "views": len([i for i in interactions if i.interaction_type == "view"]),
            "clicks": len([i for i in interactions if i.interaction_type == "click"]),
            "add_to_cart": len([i for i in interactions if i.interaction_type == "add_to_cart"]),
            "purchases": len([i for i in interactions if i.interaction_type == "purchase"]),
            "first_interaction": min([i.timestamp for i in interactions]) if interactions else None,
            "last_interaction": max([i.timestamp for i in interactions]) if interactions else None
        }

        # Calculate CTR and conversion rate
        if stats["views"] > 0:
            stats["ctr"] = stats["clicks"] / stats["views"]
        else:
            stats["ctr"] = 0.0

        if stats["clicks"] > 0:
            stats["conversion_rate"] = stats["purchases"] / stats["clicks"]
        else:
            stats["conversion_rate"] = 0.0

        return stats


# Create global auth manager instance
auth_manager = AuthManager()

# Create demo users
auth_manager.create_user(
    username="demo_user",
    email="demo@example.com",
    password="demo123",
    full_name="Demo User"
)

auth_manager.create_user(
    username="admin",
    email="admin@example.com",
    password="admin123",
    full_name="Admin User",
    role=UserRole.ADMIN
)
