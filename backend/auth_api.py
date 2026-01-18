from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"

# Database connection
def get_db():
    conn = sqlite3.connect('dsa_tutor.db')
    conn.row_factory = sqlite3.Row
    return conn

# Data models
class UserSignup(BaseModel):
    username: str
    email: Optional[str] = None
    password: str
    full_name: Optional[str] = None
    experience_level: str = "beginner"

class UserLogin(BaseModel):
    username: str
    password: str

class UserProfile(BaseModel):
    user_id: int
    full_name: Optional[str]
    experience_level: str
    preferred_topics: list
    preferred_difficulty: str

# Authentication endpoints
@app.post("/api/signup")
async def signup(user_data: UserSignup):
    """User registration"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (user_data.username,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Hash password
    password_hash = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt())
    
    # Create user
    user_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO users (username, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
    """, (user_data.username, user_data.email, password_hash.decode(), datetime.now()))
    
    conn.commit()
    user_db_id = cursor.lastrowid
    
    # Create user profile
    cursor.execute("""
        INSERT INTO user_profiles (user_id, full_name, experience_level)
        VALUES (?, ?, ?)
    """, (user_db_id, user_data.full_name, user_data.experience_level))
    
    conn.commit()
    conn.close()
    
    # Generate JWT token
    token = create_access_token({"user_id": user_db_id, "username": user_data.username})
    
    return {"message": "User created successfully", "token": token, "user_id": user_db_id}

@app.post("/api/login")
async def login(login_data: UserLogin):
    """User login"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, password_hash FROM users 
        WHERE username = ?
    """, (login_data.username,))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.checkpw(login_data.password.encode(), user['password_hash'].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                   (datetime.now(), user['id']))
    conn.commit()
    conn.close()
    
    # Generate JWT token
    token = create_access_token({"user_id": user['id'], "username": user['username']})
    
    return {"message": "Login successful", "token": token, "user_id": user['id']}

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")