from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import time
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import bcrypt
import jwt
import resend
import shutil
from twilio.rest import Client as TwilioClient
import cloudinary
import cloudinary.uploader
import cloudinary.utils
from pywebpush import webpush, WebPushException
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
import ssl
mongo_url = os.environ['MONGO_URL']
# Handle SSL/TLS issues with MongoDB Atlas
try:
    # First try with default SSL settings
    client = AsyncIOMotorClient(
        mongo_url,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=30000
    )
    logger.info("MongoDB client initialized with TLS settings")
except Exception as e:
    logger.warning(f"Failed to connect with TLS settings, trying without: {e}")
    client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET')
if not JWT_SECRET:
    import secrets
    JWT_SECRET = secrets.token_urlsafe(32)
    print("WARNING: JWT_SECRET not set in environment. Using auto-generated secret (not persistent across restarts).")
JWT_ALGORITHM = "HS256"

# Resend Configuration
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'onboarding@resend.dev')
ADMIN_NOTIFICATION_EMAIL = os.environ.get('ADMIN_NOTIFICATION_EMAIL', 'admin@nbrents.ca')

if RESEND_API_KEY and RESEND_API_KEY != 're_your_api_key_here':
    resend.api_key = RESEND_API_KEY
    logger.info("Resend email service configured")
else:
    logger.warning("Resend API key not configured - email notifications disabled")

# Twilio SMS Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
    try:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio SMS service configured")
    except Exception as e:
        logger.warning(f"Failed to initialize Twilio client: {e}")
else:
    logger.warning("Twilio not configured - SMS notifications disabled")

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET')

if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("Cloudinary configured")
else:
    logger.warning("Cloudinary not configured - using local storage")

# VAPID Configuration for Push Notifications
VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY')
VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY')
VAPID_EMAIL = os.environ.get('VAPID_EMAIL', 'mailto:help@nbrents.ca')

if VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY:
    logger.info("VAPID keys configured for push notifications")
else:
    logger.warning("VAPID keys not configured - push notifications disabled")

# Create uploads directory (fallback for local storage)
UPLOADS_DIR = ROOT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Create the main app without a prefix
app = FastAPI(title="NB Rents API")

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer(auto_error=False)

# ==================== MODELS ====================

# Auth Models
class UserBase(BaseModel):
    email: EmailStr
    name: str
    phone: Optional[str] = None
    user_type: str = Field(default="tenant", description="owner, tenant, admin, or service")

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    phone: Optional[str] = None
    user_type: str
    sms_opt_out: bool = False
    push_opt_out: bool = False
    created_at: str

class UserPreferencesUpdate(BaseModel):
    sms_opt_out: Optional[bool] = None
    push_opt_out: Optional[bool] = None
    phone: Optional[str] = None
    name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Property Models
class PropertyBase(BaseModel):
    title: str
    description: str
    address: str
    city: str
    price: float
    bedrooms: int
    bathrooms: float
    sqft: int
    property_type: str  # apartment, house, condo, townhouse
    status: str = "available"  # available, rented, maintenance
    amenities: List[str] = []
    images: List[str] = []
    featured: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PropertyCreate(PropertyBase):
    owner_id: Optional[str] = None

class PropertyResponse(PropertyBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    owner_id: Optional[str] = None
    created_at: str

# Contact Models
class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str
    property_id: Optional[str] = None

class ContactResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    subject: str
    message: str
    property_id: Optional[str] = None
    created_at: str
    status: str = "new"

# Testimonial Models
class TestimonialBase(BaseModel):
    name: str
    role: str  # Owner or Tenant
    content: str
    rating: int = Field(ge=1, le=5)
    image: Optional[str] = None

class TestimonialResponse(TestimonialBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str
    approved: bool = True

# Maintenance Request Models
class MaintenanceRequestBase(BaseModel):
    title: str
    description: str
    priority: str = "normal"  # low, normal, high, urgent
    category: str  # plumbing, electrical, hvac, appliance, other

class MaintenanceRequestCreate(MaintenanceRequestBase):
    property_id: str
    property_address: Optional[str] = None
    tenant_name: Optional[str] = None
    tenant_phone: Optional[str] = None
    tenant_email: Optional[EmailStr] = None

class WorkUpdate(BaseModel):
    note: str
    updated_by: Optional[str] = None
    timestamp: Optional[str] = None

class MaintenanceRequestUpdate(BaseModel):
    status: Optional[str] = None  # open, pending, closed
    work_note: Optional[str] = None

class MaintenanceRequestResponse(MaintenanceRequestBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    property_id: str
    property_address: Optional[str] = None
    tenant_id: str
    tenant_name: Optional[str] = None
    tenant_phone: Optional[str] = None
    tenant_email: Optional[str] = None
    status: str = "open"  # open, pending, closed
    work_notes: List[dict] = []
    created_at: str
    updated_at: Optional[str] = None

# Push Notification Models
class PushSubscription(BaseModel):
    endpoint: str
    keys: dict  # Contains p256dh and auth keys

class PushNotificationPayload(BaseModel):
    title: str
    body: str
    url: Optional[str] = None
    icon: Optional[str] = None

# ==================== AUTH HELPERS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str, email: str, user_type: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "user_type": user_type,
        "exp": datetime.now(timezone.utc).timestamp() + (24 * 60 * 60)  # 24 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"id": payload["user_id"]}, {"_id": 0, "password": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.users.find_one({"id": payload["user_id"]}, {"_id": 0, "password": 0})
        return user
    except:
        return None

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    # Case-insensitive email check
    email_lower = user_data.email.lower()
    existing = await db.users.find_one({"email": {"$regex": f"^{email_lower}$", "$options": "i"}})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    user_dict = {
        "id": user_id,
        "email": user_data.email.lower(),  # Store email in lowercase
        "name": user_data.name,
        "phone": user_data.phone,
        "user_type": user_data.user_type,
        "password": hash_password(user_data.password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_dict)
    
    token = create_token(user_id, user_data.email.lower(), user_data.user_type)
    
    user_response = UserResponse(
        id=user_id,
        email=user_data.email.lower(),
        name=user_data.name,
        phone=user_data.phone,
        user_type=user_data.user_type,
        created_at=user_dict["created_at"]
    )
    
    return TokenResponse(access_token=token, user=user_response)

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    # Case-insensitive email lookup
    email_lower = credentials.email.lower()
    user = await db.users.find_one({"email": {"$regex": f"^{email_lower}$", "$options": "i"}})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token(user["id"], user["email"], user["user_type"])
    
    user_response = UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        phone=user.get("phone"),
        user_type=user["user_type"],
        created_at=user["created_at"]
    )
    
    return TokenResponse(access_token=token, user=user_response)

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)

@api_router.put("/auth/preferences", response_model=UserResponse)
async def update_user_preferences(
    preferences: UserPreferencesUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user preferences including SMS and push notification opt-out"""
    update_data = {}
    
    if preferences.sms_opt_out is not None:
        update_data["sms_opt_out"] = preferences.sms_opt_out
    
    if preferences.push_opt_out is not None:
        update_data["push_opt_out"] = preferences.push_opt_out
    
    if preferences.phone is not None:
        update_data["phone"] = preferences.phone
    
    if preferences.name is not None:
        update_data["name"] = preferences.name
    
    if update_data:
        await db.users.update_one(
            {"id": current_user["id"]},
            {"$set": update_data}
        )
    
    # Get updated user
    updated_user = await db.users.find_one({"id": current_user["id"]}, {"_id": 0, "password": 0})
    return UserResponse(**updated_user)

# ==================== SMS NOTIFICATION HELPER ====================

def format_phone_number(phone: str) -> str:
    """Format phone number to E.164 format for Twilio"""
    if not phone:
        return None
    
    # Remove all non-digit characters
    digits = ''.join(filter(str.isdigit, phone))
    
    # If 10 digits (North American), add +1
    if len(digits) == 10:
        return f"+1{digits}"
    # If 11 digits starting with 1, add +
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+{digits}"
    # If already has country code (11+ digits)
    elif len(digits) >= 11:
        return f"+{digits}"
    else:
        return None

async def send_sms_notification(phone_number: str, message: str, tenant_id: str = None):
    """Send SMS notification to a phone number"""
    if not twilio_client:
        logger.info("SMS notification skipped - Twilio not configured")
        return False
    
    # Check if user has opted out of SMS
    if tenant_id:
        user = await db.users.find_one({"id": tenant_id}, {"_id": 0})
        if user and user.get("sms_opt_out", False):
            logger.info(f"SMS notification skipped - user opted out: {tenant_id}")
            return False
    
    formatted_phone = format_phone_number(phone_number)
    if not formatted_phone:
        logger.warning(f"Invalid phone number format: {phone_number}")
        return False
    
    try:
        message_obj = await asyncio.to_thread(
            twilio_client.messages.create,
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=formatted_phone
        )
        logger.info(f"SMS sent successfully to {formatted_phone}, SID: {message_obj.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS to {formatted_phone}: {str(e)}")
        return False

# ==================== PASSWORD RESET ROUTES ====================

# Store password reset tokens (in production, use Redis or DB with expiration)
password_reset_tokens = {}

async def send_password_reset_email(email: str, reset_token: str, user_name: str):
    """Send password reset email to user"""
    if not RESEND_API_KEY or RESEND_API_KEY == 're_your_api_key_here':
        logger.info(f"Password reset email skipped - Resend not configured. Token: {reset_token}")
        return
    
    try:
        # Use frontend URL from environment variable
        frontend_url = os.environ.get('FRONTEND_URL', 'https://maintenance-hub-test-1.preview.emergentagent.com')
        reset_link = f"{frontend_url}/?reset_token={reset_token}"
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #4f46e5; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">Password Reset Request</h1>
            </div>
            <div style="padding: 20px; background: #f9fafb;">
                <p style="color: #374151; font-size: 16px;">Hi {user_name},</p>
                
                <p style="color: #374151; font-size: 16px;">We received a request to reset your password. Click the button below to create a new password:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}" style="background: #4f46e5; color: white; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; display: inline-block;">Reset Password</a>
                </div>
                
                <p style="color: #6b7280; font-size: 14px;">Or copy and paste this link into your browser:</p>
                <p style="color: #4f46e5; font-size: 14px; word-break: break-all;">{reset_link}</p>
                
                <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <p style="color: #92400e; margin: 0; font-size: 14px;">
                        <strong>Important:</strong> This link will expire in 1 hour. If you didn't request a password reset, you can safely ignore this email.
                    </p>
                </div>
            </div>
            <div style="background: #1f2937; color: #9ca3af; padding: 15px; text-align: center; font-size: 12px;">
                NB Rents Property Management | (506) 962-RENT(7368)
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [email],
            "subject": "Reset Your NB Rents Password",
            "html": html_content
        }
        
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Password reset email sent to {email}")
    except Exception as e:
        logger.error(f"Failed to send password reset email: {str(e)}")

@api_router.post("/auth/forgot-password")
async def forgot_password(request: PasswordResetRequest):
    """Request a password reset email"""
    email_lower = request.email.lower()
    user = await db.users.find_one({"email": {"$regex": f"^{email_lower}$", "$options": "i"}})
    
    # Always return success to prevent email enumeration
    if not user:
        return {"message": "If an account exists with this email, you will receive a password reset link."}
    
    # Generate reset token
    reset_token = str(uuid.uuid4())
    expiry = datetime.now(timezone.utc).timestamp() + 3600  # 1 hour expiry
    
    # Store token
    password_reset_tokens[reset_token] = {
        "user_id": user["id"],
        "email": user["email"],
        "expiry": expiry
    }
    
    # Send email (non-blocking)
    asyncio.create_task(send_password_reset_email(user["email"], reset_token, user.get("name", "User")))
    
    return {"message": "If an account exists with this email, you will receive a password reset link."}

@api_router.post("/auth/reset-password")
async def reset_password(request: PasswordReset):
    """Reset password using token"""
    token_data = password_reset_tokens.get(request.token)
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Check expiry
    if datetime.now(timezone.utc).timestamp() > token_data["expiry"]:
        del password_reset_tokens[request.token]
        raise HTTPException(status_code=400, detail="Reset token has expired")
    
    # Update password
    new_hashed_password = hash_password(request.new_password)
    await db.users.update_one(
        {"id": token_data["user_id"]},
        {"$set": {"password": new_hashed_password}}
    )
    
    # Remove used token
    del password_reset_tokens[request.token]
    
    return {"message": "Password has been reset successfully. You can now log in with your new password."}

@api_router.get("/auth/verify-reset-token/{token}")
async def verify_reset_token(token: str):
    """Verify if a reset token is valid"""
    token_data = password_reset_tokens.get(token)
    
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid reset token")
    
    if datetime.now(timezone.utc).timestamp() > token_data["expiry"]:
        del password_reset_tokens[token]
        raise HTTPException(status_code=400, detail="Reset token has expired")
    
    return {"valid": True, "email": token_data["email"]}

# ==================== PROPERTY ROUTES ====================

@api_router.get("/properties", response_model=List[PropertyResponse])
async def get_properties(
    status: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    featured: Optional[bool] = None,
    city: Optional[str] = None
):
    query = {}
    if status:
        query["status"] = status
    if property_type:
        query["property_type"] = property_type
    if min_price is not None:
        query["price"] = {"$gte": min_price}
    if max_price is not None:
        query.setdefault("price", {})["$lte"] = max_price
    if bedrooms is not None:
        query["bedrooms"] = bedrooms
    if featured is not None:
        query["featured"] = featured
    if city:
        query["city"] = city
    
    properties = await db.properties.find(query, {"_id": 0}).to_list(100)
    return properties

@api_router.get("/properties/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str):
    prop = await db.properties.find_one({"id": property_id}, {"_id": 0})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return prop

@api_router.post("/properties", response_model=PropertyResponse)
async def create_property(
    property_data: PropertyCreate,
    current_user: dict = Depends(get_current_user)
):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Only owners can create properties")
    
    property_id = str(uuid.uuid4())
    property_dict = property_data.model_dump()
    property_dict["id"] = property_id
    property_dict["owner_id"] = current_user["id"]
    property_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.properties.insert_one(property_dict)
    property_dict.pop("_id", None)
    return property_dict

@api_router.put("/properties/{property_id}", response_model=PropertyResponse)
async def update_property(
    property_id: str,
    property_data: PropertyBase,
    current_user: dict = Depends(get_current_user)
):
    prop = await db.properties.find_one({"id": property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    if prop.get("owner_id") and prop["owner_id"] != current_user["id"] and current_user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_dict = property_data.model_dump()
    await db.properties.update_one({"id": property_id}, {"$set": update_dict})
    
    updated = await db.properties.find_one({"id": property_id}, {"_id": 0})
    return updated

@api_router.patch("/properties/{property_id}/featured")
async def toggle_featured(
    property_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Toggle featured status of a property"""
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    prop = await db.properties.find_one({"id": property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    new_featured = not prop.get("featured", False)
    await db.properties.update_one(
        {"id": property_id}, 
        {"$set": {"featured": new_featured}}
    )
    
    return {"id": property_id, "featured": new_featured}

@api_router.patch("/properties/{property_id}/status")
async def update_property_status(
    property_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    """Update property status (available, rented, maintenance)"""
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if status not in ["available", "rented", "maintenance"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    prop = await db.properties.find_one({"id": property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    await db.properties.update_one(
        {"id": property_id}, 
        {"$set": {"status": status}}
    )
    
    return {"id": property_id, "status": status}

@api_router.delete("/properties/{property_id}")
async def delete_property(property_id: str, current_user: dict = Depends(get_current_user)):
    prop = await db.properties.find_one({"id": property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    if prop["owner_id"] != current_user["id"] and current_user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.properties.delete_one({"id": property_id})
    return {"message": "Property deleted"}

# ==================== IMAGE UPLOAD ROUTES ====================

@api_router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload an image file to Cloudinary and return its URL"""
    if current_user["user_type"] not in ["owner", "admin", "tenant"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: JPEG, PNG, WebP, GIF")
    
    # Validate file size (max 5MB)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size: 5MB")
    
    # Upload to Cloudinary if configured
    if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
        try:
            # Reset file position for upload
            await file.seek(0)
            result = await asyncio.to_thread(
                cloudinary.uploader.upload,
                contents,
                folder="nbrents/properties",
                resource_type="image"
            )
            return {
                "url": result["secure_url"],
                "public_id": result["public_id"],
                "cloudinary": True
            }
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")
            # Fall back to local storage
    
    # Fallback to local storage
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = UPLOADS_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(contents)
    
    return {
        "url": f"/uploads/{filename}",
        "filename": filename,
        "cloudinary": False
    }

@api_router.post("/upload/images")
async def upload_multiple_images(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload multiple image files to Cloudinary and return their URLs"""
    if current_user["user_type"] not in ["owner", "admin", "tenant"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    urls = []
    
    for file in files:
        if file.content_type not in allowed_types:
            continue
        
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            continue
        
        # Upload to Cloudinary if configured
        if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
            try:
                result = await asyncio.to_thread(
                    cloudinary.uploader.upload,
                    contents,
                    folder="nbrents/properties",
                    resource_type="image"
                )
                urls.append(result["secure_url"])
                continue
            except Exception as e:
                logger.error(f"Cloudinary upload failed: {e}")
        
        # Fallback to local storage
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = UPLOADS_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        urls.append(f"/uploads/{filename}")
    
    return {"urls": urls}

# ==================== CLOUDINARY SIGNATURE (for direct uploads) ====================

@api_router.get("/cloudinary/signature")
async def get_cloudinary_signature(
    folder: str = Query(default="nbrents/properties"),
    current_user: dict = Depends(get_current_user)
):
    """Generate a signature for direct Cloudinary uploads from frontend"""
    if not CLOUDINARY_API_SECRET:
        raise HTTPException(status_code=503, detail="Cloudinary not configured")
    
    # Validate folder
    allowed_folders = ("nbrents/properties", "nbrents/maintenance", "nbrents/users")
    if not folder.startswith(allowed_folders):
        raise HTTPException(status_code=400, detail="Invalid folder path")
    
    timestamp = int(time.time())
    params = {
        "timestamp": timestamp,
        "folder": folder
    }
    
    signature = cloudinary.utils.api_sign_request(params, CLOUDINARY_API_SECRET)
    
    return {
        "signature": signature,
        "timestamp": timestamp,
        "cloud_name": CLOUDINARY_CLOUD_NAME,
        "api_key": CLOUDINARY_API_KEY,
        "folder": folder
    }

# ==================== PUSH NOTIFICATIONS ====================

@api_router.get("/push/vapid-key")
async def get_vapid_public_key():
    """Get the VAPID public key for push subscription"""
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=503, detail="Push notifications not configured")
    return {"publicKey": VAPID_PUBLIC_KEY}

@api_router.post("/push/subscribe")
async def subscribe_to_push(
    subscription: PushSubscription,
    current_user: dict = Depends(get_current_user)
):
    """Subscribe a user to push notifications"""
    if not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        raise HTTPException(status_code=503, detail="Push notifications not configured")
    
    # Store subscription in database
    sub_data = {
        "user_id": current_user["id"],
        "endpoint": subscription.endpoint,
        "keys": subscription.keys,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Upsert (replace existing subscription for this user/endpoint)
    await db.push_subscriptions.update_one(
        {"user_id": current_user["id"], "endpoint": subscription.endpoint},
        {"$set": sub_data},
        upsert=True
    )
    
    return {"message": "Successfully subscribed to push notifications"}

@api_router.delete("/push/unsubscribe")
async def unsubscribe_from_push(
    endpoint: str,
    current_user: dict = Depends(get_current_user)
):
    """Unsubscribe from push notifications"""
    await db.push_subscriptions.delete_one({
        "user_id": current_user["id"],
        "endpoint": endpoint
    })
    return {"message": "Successfully unsubscribed from push notifications"}

async def send_push_notification(user_id: str, title: str, body: str, url: str = None):
    """Send push notification to all subscriptions for a user"""
    if not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        logger.info("Push notification skipped - VAPID not configured")
        return
    
    # Check if user has opted out
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if user and user.get("push_opt_out", False):
        logger.info(f"Push notification skipped - user opted out: {user_id}")
        return
    
    # Get all subscriptions for user
    subscriptions = await db.push_subscriptions.find({"user_id": user_id}, {"_id": 0}).to_list(10)
    
    if not subscriptions:
        logger.info(f"No push subscriptions found for user: {user_id}")
        return
    
    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url or "/tenant-app",
        "icon": "/icon-192.png"
    })
    
    for sub in subscriptions:
        try:
            await asyncio.to_thread(
                webpush,
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": sub["keys"]
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_EMAIL}
            )
            logger.info(f"Push notification sent to user: {user_id}")
        except WebPushException as e:
            logger.error(f"Push notification failed: {e}")
            # Remove invalid subscription
            if e.response and e.response.status_code in [404, 410]:
                await db.push_subscriptions.delete_one({"endpoint": sub["endpoint"]})
                logger.info(f"Removed invalid subscription: {sub['endpoint']}")
        except Exception as e:
            logger.error(f"Push notification error: {e}")

# ==================== CONTACT ROUTES ====================

@api_router.post("/contact", response_model=ContactResponse)
async def submit_contact(message: ContactMessage):
    contact_id = str(uuid.uuid4())
    contact_dict = message.model_dump()
    contact_dict["id"] = contact_id
    contact_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    contact_dict["status"] = "new"
    
    await db.contacts.insert_one(contact_dict)
    contact_dict.pop("_id", None)
    
    # Send email notification to admin
    asyncio.create_task(send_contact_form_notification(contact_dict))
    
    return contact_dict

async def send_contact_form_notification(contact_data: dict):
    """Send email notification to admin when contact form is submitted"""
    if not RESEND_API_KEY or RESEND_API_KEY == 're_your_api_key_here':
        logger.info("Contact form notification skipped - Resend not configured")
        return
    
    try:
        name = contact_data.get('name', 'Unknown')
        email = contact_data.get('email', 'No email provided')
        phone = contact_data.get('phone', 'Not provided')
        subject = contact_data.get('subject', 'No subject')
        message_text = contact_data.get('message', 'No message')
        property_id = contact_data.get('property_id')
        created_at = contact_data.get('created_at', '')
        
        # Format the date nicely
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%B %d, %Y at %I:%M %p')
        except:
            formatted_date = created_at
        
        property_section = ""
        if property_id:
            property_section = f"""
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Property Interest</td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">ID: {property_id}</td>
                </tr>
            """
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0; font-size: 24px;">New Contact Form Submission</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 14px;">Someone reached out through your website</p>
            </div>
            
            <div style="padding: 30px; background: #ffffff; border: 1px solid #e5e7eb; border-top: none;">
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <p style="margin: 0; color: #166534; font-size: 14px;">
                        <strong>📬 New inquiry received</strong> on {formatted_date}
                    </p>
                </div>
                
                <h2 style="color: #111827; font-size: 18px; margin: 0 0 15px 0; border-bottom: 2px solid #4f46e5; padding-bottom: 10px;">Contact Details</h2>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; width: 120px;">Name</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px; font-weight: 600;">{name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Email</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">
                            <a href="mailto:{email}" style="color: #4f46e5; text-decoration: none;">{email}</a>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Phone</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px;">
                            {f'<a href="tel:{phone}" style="color: #4f46e5; text-decoration: none;">{phone}</a>' if phone != 'Not provided' else '<span style="color: #9ca3af;">Not provided</span>'}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 14px;">Subject</td>
                        <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb; color: #111827; font-size: 14px; font-weight: 600;">{subject}</td>
                    </tr>
                    {property_section}
                </table>
                
                <h3 style="color: #111827; font-size: 16px; margin: 25px 0 10px 0;">Message</h3>
                <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px;">
                    <p style="margin: 0; color: #374151; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">{message_text}</p>
                </div>
                
                <div style="margin-top: 25px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                    <a href="mailto:{email}?subject=Re: {subject}" style="display: inline-block; background: #4f46e5; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px;">
                        Reply to {name.split()[0]}
                    </a>
                </div>
            </div>
            
            <div style="background: #1f2937; color: #9ca3af; padding: 20px; text-align: center; font-size: 12px; border-radius: 0 0 8px 8px;">
                <p style="margin: 0;">NB Rents Property Management</p>
                <p style="margin: 5px 0 0 0;">(506) 962-RENT(7368) | <a href="mailto:Help@NBRents.ca" style="color: #9ca3af;">Help@NBRents.ca</a></p>
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [ADMIN_NOTIFICATION_EMAIL],
            "reply_to": email,
            "subject": f"📬 New Contact: {subject} - from {name}",
            "html": html_content
        }
        
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Contact form notification sent to {ADMIN_NOTIFICATION_EMAIL}")
        
        # Also send confirmation email to the person who submitted
        await send_contact_confirmation_email(contact_data)
        
    except Exception as e:
        logger.error(f"Failed to send contact form notification: {str(e)}")

async def send_contact_confirmation_email(contact_data: dict):
    """Send confirmation email to the person who submitted the contact form"""
    try:
        name = contact_data.get('name', 'there')
        email = contact_data.get('email')
        subject = contact_data.get('subject', 'your inquiry')
        
        if not email:
            return
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0; font-size: 24px;">Thank You for Contacting Us!</h1>
            </div>
            
            <div style="padding: 30px; background: #ffffff; border: 1px solid #e5e7eb; border-top: none;">
                <p style="color: #374151; font-size: 16px; line-height: 1.6;">
                    Hi {name.split()[0]},
                </p>
                
                <p style="color: #374151; font-size: 16px; line-height: 1.6;">
                    Thank you for reaching out to NB Rents! We've received your message regarding "<strong>{subject}</strong>" and our team will get back to you as soon as possible.
                </p>
                
                <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <p style="margin: 0; color: #0369a1; font-size: 14px;">
                        <strong>⏰ Response Time:</strong> We typically respond within 24-48 business hours.
                    </p>
                </div>
                
                <p style="color: #374151; font-size: 16px; line-height: 1.6;">
                    In the meantime, feel free to:
                </p>
                
                <ul style="color: #374151; font-size: 14px; line-height: 1.8;">
                    <li>Browse our <a href="{os.environ.get('FRONTEND_URL', '')}/properties" style="color: #4f46e5;">available properties</a></li>
                    <li>Learn more about our <a href="{os.environ.get('FRONTEND_URL', '')}/services" style="color: #4f46e5;">property management services</a></li>
                    <li>Call us directly at <a href="tel:5069627368" style="color: #4f46e5;">(506) 962-RENT(7368)</a> for urgent matters</li>
                </ul>
                
                <p style="color: #374151; font-size: 16px; line-height: 1.6; margin-top: 25px;">
                    Best regards,<br>
                    <strong>The NB Rents Team</strong>
                </p>
            </div>
            
            <div style="background: #1f2937; color: #9ca3af; padding: 20px; text-align: center; font-size: 12px; border-radius: 0 0 8px 8px;">
                <p style="margin: 0;">NB Rents Property Management</p>
                <p style="margin: 5px 0 0 0;">72 Elizabeth St. Unit 85, Miramichi, NB E1V 1W1</p>
                <p style="margin: 5px 0 0 0;">(506) 962-RENT(7368) | <a href="mailto:Help@NBRents.ca" style="color: #9ca3af;">Help@NBRents.ca</a></p>
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [email],
            "subject": f"Thank you for contacting NB Rents!",
            "html": html_content
        }
        
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Contact confirmation email sent to {email}")
        
    except Exception as e:
        logger.error(f"Failed to send contact confirmation email: {str(e)}")

@api_router.get("/contacts", response_model=List[ContactResponse])
async def get_contacts(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    contacts = await db.contacts.find({}, {"_id": 0}).to_list(100)
    return contacts

# ==================== TESTIMONIAL ROUTES ====================

@api_router.get("/testimonials", response_model=List[TestimonialResponse])
async def get_testimonials():
    testimonials = await db.testimonials.find({"approved": True}, {"_id": 0}).to_list(50)
    return testimonials

@api_router.post("/testimonials", response_model=TestimonialResponse)
async def create_testimonial(testimonial: TestimonialBase, current_user: dict = Depends(get_current_user)):
    testimonial_id = str(uuid.uuid4())
    testimonial_dict = testimonial.model_dump()
    testimonial_dict["id"] = testimonial_id
    testimonial_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    testimonial_dict["approved"] = True
    
    await db.testimonials.insert_one(testimonial_dict)
    testimonial_dict.pop("_id", None)
    return testimonial_dict

# ==================== MAINTENANCE REQUEST ROUTES ====================

async def send_admin_notification(request_data: dict, tenant_name: str):
    """Send email notification to Help@NBRents.ca when a maintenance request is submitted"""
    if not RESEND_API_KEY or RESEND_API_KEY == 're_your_api_key_here':
        logger.info("Email notification skipped - Resend not configured")
        return
    
    try:
        # Get tenant contact info from request
        contact_name = request_data.get('tenant_name') or tenant_name
        contact_phone = request_data.get('tenant_phone', 'Not provided')
        contact_email = request_data.get('tenant_email', 'Not provided')
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #4f46e5; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">New Maintenance Request</h1>
            </div>
            <div style="padding: 20px; background: #f9fafb;">
                <p style="color: #374151; font-size: 16px;">A new maintenance request has been submitted:</p>
                
                <h3 style="color: #4f46e5; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #e5e7eb; padding-bottom: 5px;">Tenant Contact Information</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280; width: 140px;"><strong>Name:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{contact_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;"><strong>Phone:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{contact_phone}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;"><strong>Email:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{contact_email}</td>
                    </tr>
                </table>
                
                <h3 style="color: #4f46e5; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #e5e7eb; padding-bottom: 5px;">Request Details</h3>
                <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280; width: 140px;"><strong>Property:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{request_data.get('property_address', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;"><strong>Issue:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{request_data.get('title', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;"><strong>Category:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #111827;">{request_data.get('category', 'other').capitalize()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #6b7280;"><strong>Priority:</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: {'#dc2626' if request_data.get('priority') == 'urgent' else '#f59e0b' if request_data.get('priority') == 'high' else '#111827'}; font-weight: {'bold' if request_data.get('priority') in ['urgent', 'high'] else 'normal'};">{request_data.get('priority', 'normal').upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #6b7280; vertical-align: top;"><strong>Description:</strong></td>
                        <td style="padding: 10px; color: #111827;">{request_data.get('description', 'No description provided')}</td>
                    </tr>
                </table>
                
                <p style="color: #6b7280; font-size: 14px; margin-top: 20px;">
                    Log in to the <a href="#" style="color: #4f46e5;">Owner Portal</a> to manage this request.
                </p>
            </div>
            <div style="background: #1f2937; color: #9ca3af; padding: 15px; text-align: center; font-size: 12px;">
                NB Rents Property Management | (506) 962-RENT(7368)
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [ADMIN_NOTIFICATION_EMAIL],
            "subject": f"🔧 New Maintenance Request: {request_data.get('title', 'Maintenance Issue')}",
            "html": html_content
        }
        
        # Run sync SDK in thread to keep FastAPI non-blocking
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Admin notification email sent for maintenance request")
    except Exception as e:
        logger.error(f"Failed to send admin notification email: {str(e)}")

async def send_tenant_confirmation(request_data: dict):
    """Send confirmation email to tenant when they submit a maintenance request"""
    tenant_email = request_data.get('tenant_email')
    
    if not tenant_email:
        logger.info("Tenant confirmation skipped - no email provided")
        return
        
    if not RESEND_API_KEY or RESEND_API_KEY == 're_your_api_key_here':
        logger.info("Tenant confirmation skipped - Resend not configured")
        return
    
    try:
        tenant_name = request_data.get('tenant_name', 'Valued Tenant')
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #4f46e5; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">Maintenance Request Received</h1>
            </div>
            <div style="padding: 20px; background: #f9fafb;">
                <p style="color: #374151; font-size: 16px;">Hi {tenant_name},</p>
                
                <p style="color: #374151; font-size: 16px;">Thank you for submitting your maintenance request. We have received it and our team will be in touch soon.</p>
                
                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #4f46e5; margin-top: 0; margin-bottom: 15px;">Request Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280; width: 120px;"><strong>Reference #:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('id', 'N/A')[:8].upper()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280;"><strong>Property:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('property_address', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280;"><strong>Issue:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('title', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280;"><strong>Category:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('category', 'other').capitalize()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280;"><strong>Priority:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('priority', 'normal').capitalize()}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; color: #6b7280; vertical-align: top;"><strong>Description:</strong></td>
                            <td style="padding: 8px 0; color: #111827;">{request_data.get('description', 'N/A')}</td>
                        </tr>
                    </table>
                </div>
                
                <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <p style="color: #92400e; margin: 0; font-size: 14px;">
                        <strong>What happens next?</strong><br>
                        Our maintenance team will review your request and contact you within 24-48 hours. For urgent issues, please call us directly.
                    </p>
                </div>
                
                <p style="color: #6b7280; font-size: 14px;">
                    If you have any questions, feel free to contact us at <a href="tel:5069627368" style="color: #4f46e5;">(506) 962-RENT(7368)</a> or reply to this email.
                </p>
            </div>
            <div style="background: #1f2937; color: #9ca3af; padding: 15px; text-align: center; font-size: 12px;">
                NB Rents Property Management | (506) 962-RENT(7368)<br>
                <a href="mailto:Help@NBRents.ca" style="color: #9ca3af;">Help@NBRents.ca</a>
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [tenant_email],
            "subject": f"✅ Maintenance Request Received - {request_data.get('title', 'Your Request')}",
            "html": html_content
        }
        
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Tenant confirmation email sent to {tenant_email}")
    except Exception as e:
        logger.error(f"Failed to send tenant confirmation email: {str(e)}")

@api_router.post("/maintenance-requests", response_model=MaintenanceRequestResponse)
async def create_maintenance_request(
    request: MaintenanceRequestCreate,
    current_user: dict = Depends(get_current_user)
):
    request_id = str(uuid.uuid4())
    request_dict = request.model_dump()
    request_dict["id"] = request_id
    request_dict["tenant_id"] = current_user["id"]
    request_dict["status"] = "open"
    request_dict["work_notes"] = []
    request_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.maintenance_requests.insert_one(request_dict)
    request_dict.pop("_id", None)
    
    # Send email notifications (non-blocking)
    asyncio.create_task(send_admin_notification(request_dict, current_user.get("name", "Unknown Tenant")))
    asyncio.create_task(send_tenant_confirmation(request_dict))
    
    return request_dict

@api_router.get("/maintenance-requests", response_model=List[MaintenanceRequestResponse])
async def get_maintenance_requests(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] == "tenant":
        requests = await db.maintenance_requests.find(
            {"tenant_id": current_user["id"]}, {"_id": 0}
        ).to_list(100)
    else:
        # Owners see requests for their properties
        properties = await db.properties.find(
            {"owner_id": current_user["id"]}, {"id": 1, "_id": 0}
        ).to_list(100)
        property_ids = [p["id"] for p in properties]
        requests = await db.maintenance_requests.find(
            {"property_id": {"$in": property_ids}}, {"_id": 0}
        ).to_list(100)
    return requests

@api_router.put("/maintenance-requests/{request_id}/status")
async def update_maintenance_status(
    request_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.maintenance_requests.update_one(
        {"id": request_id},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return {"message": "Status updated"}

# ==================== SERVICE PORTAL ROUTES ====================

async def send_tenant_update_notification(request_data: dict, update_note: str, new_status: str = None, updated_by: str = "NB Rents Team"):
    """Send email, SMS, and push notification to tenant when their maintenance request is updated"""
    tenant_email = request_data.get('tenant_email')
    tenant_phone = request_data.get('tenant_phone')
    tenant_id = request_data.get('tenant_id')
    tenant_name = request_data.get('tenant_name', 'Valued Tenant')
    status_display = new_status.capitalize() if new_status else request_data.get('status', 'Open').capitalize()
    request_title = request_data.get('title', 'Your Request')
    
    # Send push notification
    if tenant_id:
        push_title = f"Maintenance Update: {status_display}"
        push_body = f"{request_title} - {update_note[:100]}{'...' if len(update_note) > 100 else ''}"
        asyncio.create_task(send_push_notification(tenant_id, push_title, push_body, "/tenant-app"))
    
    # Send SMS notification if phone is available
    if tenant_phone:
        sms_message = f"NB Rents: Update on your maintenance request '{request_title}'. Status: {status_display}. {update_note[:100]}{'...' if len(update_note) > 100 else ''} Reply STOP to opt out."
        asyncio.create_task(send_sms_notification(tenant_phone, sms_message, tenant_id))
    
    # Send email notification
    if not tenant_email:
        logger.info("Tenant email notification skipped - no email provided")
        return
        
    if not RESEND_API_KEY or RESEND_API_KEY == 're_your_api_key_here':
        logger.info("Tenant email notification skipped - Resend not configured")
        return
    
    try:
        # Status color coding
        status_colors = {
            'open': '#3b82f6',      # blue
            'pending': '#f59e0b',   # amber
            'closed': '#10b981'     # green
        }
        status_color = status_colors.get(new_status or request_data.get('status', 'open'), '#6b7280')
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #4f46e5; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">Maintenance Request Update</h1>
            </div>
            <div style="padding: 20px; background: #f9fafb;">
                <p style="color: #374151; font-size: 16px;">Hi {tenant_name},</p>
                
                <p style="color: #374151; font-size: 16px;">There's an update on your maintenance request:</p>
                
                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="color: #111827; margin: 0;">{request_data.get('title', 'Maintenance Request')}</h3>
                        <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">{status_display}</span>
                    </div>
                    <p style="color: #6b7280; font-size: 14px; margin-bottom: 10px;">
                        <strong>Property:</strong> {request_data.get('property_address', 'N/A')}
                    </p>
                    <p style="color: #6b7280; font-size: 14px; margin-bottom: 0;">
                        <strong>Reference #:</strong> {request_data.get('id', 'N/A')[:8].upper()}
                    </p>
                </div>
                
                <div style="background: #ecfdf5; border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin: 20px 0;">
                    <h4 style="color: #065f46; margin: 0 0 10px 0;">Latest Update</h4>
                    <p style="color: #047857; margin: 0; font-size: 14px;">{update_note}</p>
                    <p style="color: #6b7280; font-size: 12px; margin-top: 10px; margin-bottom: 0;">— {updated_by}</p>
                </div>
                
                <p style="color: #6b7280; font-size: 14px;">
                    If you have any questions, feel free to contact us at <a href="tel:5069627368" style="color: #4f46e5;">(506) 962-RENT(7368)</a> or reply to this email.
                </p>
            </div>
            <div style="background: #1f2937; color: #9ca3af; padding: 15px; text-align: center; font-size: 12px;">
                NB Rents Property Management | (506) 962-RENT(7368)<br>
                <a href="mailto:Help@NBRents.ca" style="color: #9ca3af;">Help@NBRents.ca</a>
            </div>
        </div>
        """
        
        params = {
            "from": SENDER_EMAIL,
            "to": [tenant_email],
            "subject": f"📋 Update on Your Maintenance Request - {request_data.get('title', 'Your Request')}",
            "html": html_content
        }
        
        await asyncio.to_thread(resend.Emails.send, params)
        logger.info(f"Tenant update notification sent to {tenant_email}")
    except Exception as e:
        logger.error(f"Failed to send tenant update notification: {str(e)}")

@api_router.get("/service/requests", response_model=List[MaintenanceRequestResponse])
async def get_all_maintenance_requests(current_user: dict = Depends(get_current_user)):
    """Get all maintenance requests for the service portal (admin/service only)"""
    if current_user["user_type"] not in ["service", "admin"]:
        raise HTTPException(status_code=403, detail="Service portal access required")
    
    # Admin and service users see all requests
    requests = await db.maintenance_requests.find({}, {"_id": 0}).sort("created_at", -1).to_list(500)
    return requests

@api_router.get("/service/requests/{request_id}")
async def get_maintenance_request_detail(request_id: str, current_user: dict = Depends(get_current_user)):
    """Get detailed maintenance request with full history"""
    if current_user["user_type"] not in ["service", "admin"]:
        raise HTTPException(status_code=403, detail="Service portal access required")
    
    request = await db.maintenance_requests.find_one({"id": request_id}, {"_id": 0})
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    return request

@api_router.put("/service/requests/{request_id}")
async def update_maintenance_request(
    request_id: str,
    update: MaintenanceRequestUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a maintenance request with status and/or work note"""
    if current_user["user_type"] not in ["service", "admin"]:
        raise HTTPException(status_code=403, detail="Service portal access required")
    
    # Get the existing request
    existing = await db.maintenance_requests.find_one({"id": request_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Request not found")
    
    update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}
    
    # Update status if provided
    if update.status:
        if update.status not in ["open", "pending", "closed"]:
            raise HTTPException(status_code=400, detail="Invalid status. Must be: open, pending, or closed")
        update_data["status"] = update.status
    
    # Add work note if provided
    if update.work_note:
        work_note_entry = {
            "note": update.work_note,
            "updated_by": current_user.get("name", "Staff"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_at_time": update.status or existing.get("status", "open")
        }
        # Get existing notes or initialize empty list
        existing_notes = existing.get("work_notes", [])
        existing_notes.append(work_note_entry)
        update_data["work_notes"] = existing_notes
    
    await db.maintenance_requests.update_one(
        {"id": request_id},
        {"$set": update_data}
    )
    
    # Get the updated request
    updated_request = await db.maintenance_requests.find_one({"id": request_id}, {"_id": 0})
    
    # Send email notification to tenant (non-blocking)
    if update.work_note or update.status:
        notification_message = update.work_note or f"Status updated to: {update.status.capitalize()}"
        asyncio.create_task(send_tenant_update_notification(
            updated_request, 
            notification_message, 
            update.status,
            current_user.get("name", "NB Rents Team")
        ))
    
    return updated_request

@api_router.get("/service/stats")
async def get_service_stats(current_user: dict = Depends(get_current_user)):
    """Get statistics for the service portal"""
    if current_user["user_type"] not in ["service", "admin"]:
        raise HTTPException(status_code=403, detail="Service portal access required")
    
    all_requests = await db.maintenance_requests.find({}, {"_id": 0}).to_list(500)
    
    open_count = sum(1 for r in all_requests if r.get("status") == "open")
    pending_count = sum(1 for r in all_requests if r.get("status") == "pending")
    closed_count = sum(1 for r in all_requests if r.get("status") == "closed")
    
    # Count by priority
    urgent_count = sum(1 for r in all_requests if r.get("priority") == "urgent" and r.get("status") != "closed")
    high_count = sum(1 for r in all_requests if r.get("priority") == "high" and r.get("status") != "closed")
    
    return {
        "total": len(all_requests),
        "open": open_count,
        "pending": pending_count,
        "closed": closed_count,
        "urgent_active": urgent_count,
        "high_priority_active": high_count
    }

# ==================== OWNER PORTAL ROUTES ====================

@api_router.get("/owner/properties", response_model=List[PropertyResponse])
async def get_owner_properties(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] == "admin":
        # Admin sees all properties
        properties = await db.properties.find({}, {"_id": 0}).to_list(100)
    elif current_user["user_type"] == "owner":
        # Owners only see their own properties
        properties = await db.properties.find(
            {"owner_id": current_user["id"]}, {"_id": 0}
        ).to_list(100)
    else:
        raise HTTPException(status_code=403, detail="Owner or admin access required")
    return properties

@api_router.get("/owner/stats")
async def get_owner_stats(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] == "admin":
        # Admin sees stats for all properties
        properties = await db.properties.find({}).to_list(100)
    elif current_user["user_type"] == "owner":
        # Owners see stats for their own properties only
        properties = await db.properties.find({"owner_id": current_user["id"]}).to_list(100)
    else:
        raise HTTPException(status_code=403, detail="Owner or admin access required")
    
    total_properties = len(properties)
    rented = sum(1 for p in properties if p.get("status") == "rented")
    available = sum(1 for p in properties if p.get("status") == "available")
    total_rent = sum(p.get("price", 0) for p in properties if p.get("status") == "rented")
    
    return {
        "total_properties": total_properties,
        "rented": rented,
        "available": available,
        "monthly_revenue": total_rent,
        "occupancy_rate": (rented / total_properties * 100) if total_properties > 0 else 0
    }

# ==================== TENANT PORTAL ROUTES ====================

@api_router.get("/tenant/dashboard")
async def get_tenant_dashboard(current_user: dict = Depends(get_current_user)):
    # Allow both tenants and admins
    if current_user["user_type"] not in ["tenant", "admin"]:
        raise HTTPException(status_code=403, detail="Tenant or admin access required")
    
    # Admins see ALL maintenance requests, tenants see only their own
    if current_user["user_type"] == "admin":
        requests = await db.maintenance_requests.find({}, {"_id": 0}).to_list(100)
    else:
        requests = await db.maintenance_requests.find(
            {"tenant_id": current_user["id"]}, {"_id": 0}
        ).to_list(100)
    
    pending = sum(1 for r in requests if r.get("status") == "pending")
    in_progress = sum(1 for r in requests if r.get("status") == "in_progress")
    completed = sum(1 for r in requests if r.get("status") == "completed")
    
    return {
        "maintenance_requests": {
            "total": len(requests),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed
        },
        "recent_requests": requests[:10] if current_user["user_type"] == "admin" else requests[:5]
    }

# ==================== SEED DATA ====================

@api_router.post("/seed")
async def seed_data():
    """Seed the database with sample data"""
    
    # Clear existing data
    await db.properties.delete_many({})
    await db.testimonials.delete_many({})
    
    # Sample properties with coordinates for New Brunswick cities
    properties = [
        {
            "id": str(uuid.uuid4()),
            "title": "Modern Downtown Apartment",
            "description": "Beautiful 2-bedroom apartment in the heart of downtown. Recently renovated with modern finishes, stainless steel appliances, and stunning city views.",
            "address": "123 Main Street, Unit 4B",
            "city": "Moncton",
            "price": 1850,
            "bedrooms": 2,
            "bathrooms": 1,
            "sqft": 950,
            "property_type": "apartment",
            "status": "available",
            "amenities": ["In-unit laundry", "Parking", "Gym access", "Rooftop deck"],
            "images": ["https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=800"],
            "featured": True,
            "owner_id": None,
            "latitude": 46.0878,
            "longitude": -64.7782,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Cozy Family Home",
            "description": "Spacious 4-bedroom family home with large backyard, updated kitchen, and finished basement. Perfect for families!",
            "address": "456 Oak Avenue",
            "city": "Moncton",
            "price": 2400,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "sqft": 2200,
            "property_type": "house",
            "status": "available",
            "amenities": ["Garage", "Backyard", "Central AC", "Fireplace"],
            "images": ["https://images.unsplash.com/photo-1627141234469-24711efb373c?w=800"],
            "featured": True,
            "owner_id": None,
            "latitude": 46.0950,
            "longitude": -64.8010,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Luxury Waterfront Condo",
            "description": "Stunning waterfront condo with panoramic views, high-end finishes, and resort-style amenities.",
            "address": "789 Harbor Drive, Unit 12",
            "city": "Saint John",
            "price": 3200,
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft": 1800,
            "property_type": "condo",
            "status": "rented",
            "amenities": ["Pool", "Concierge", "Fitness center", "Waterfront views"],
            "images": ["https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=800"],
            "featured": True,
            "owner_id": None,
            "latitude": 45.2733,
            "longitude": -66.0633,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Charming Townhouse",
            "description": "End-unit townhouse with private patio, updated bathrooms, and open floor plan. Close to schools and shopping.",
            "address": "321 Maple Lane",
            "city": "Fredericton",
            "price": 1950,
            "bedrooms": 3,
            "bathrooms": 1.5,
            "sqft": 1400,
            "property_type": "townhouse",
            "status": "available",
            "amenities": ["Patio", "Storage unit", "Parking", "Pet-friendly"],
            "images": ["https://images.unsplash.com/photo-1605276374104-dee2a0ed3cd6?w=800"],
            "featured": False,
            "owner_id": None,
            "latitude": 45.9636,
            "longitude": -66.6431,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Studio Loft",
            "description": "Bright studio loft with exposed brick, high ceilings, and modern amenities. Perfect for young professionals.",
            "address": "555 Arts District Way",
            "city": "Moncton",
            "price": 1200,
            "bedrooms": 0,
            "bathrooms": 1,
            "sqft": 550,
            "property_type": "apartment",
            "status": "available",
            "amenities": ["High ceilings", "Exposed brick", "In-unit laundry", "Bike storage"],
            "images": ["https://images.unsplash.com/photo-1536376072261-38c75010e6c9?w=800"],
            "featured": False,
            "owner_id": None,
            "latitude": 46.0920,
            "longitude": -64.7650,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Suburban Retreat",
            "description": "Beautiful 3-bedroom home in quiet suburban neighborhood. Large lot, renovated kitchen, and two-car garage.",
            "address": "888 Willow Court",
            "city": "Dieppe",
            "price": 2100,
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft": 1650,
            "property_type": "house",
            "status": "available",
            "amenities": ["Two-car garage", "Large yard", "Deck", "Central AC"],
            "images": ["https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=800"],
            "featured": True,
            "owner_id": None,
            "latitude": 46.0980,
            "longitude": -64.7240,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    await db.properties.insert_many(properties)
    
    # Sample testimonials
    testimonials = [
        {
            "id": str(uuid.uuid4()),
            "name": "Sarah Mitchell",
            "role": "Property Owner",
            "content": "NB Rents has been managing my properties for 3 years now. They've increased my rental income by 20% and their maintenance team is incredibly responsive. Highly recommend!",
            "rating": 5,
            "image": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved": True
        },
        {
            "id": str(uuid.uuid4()),
            "name": "James Thompson",
            "role": "Tenant",
            "content": "Best rental experience I've ever had. The online portal makes everything easy, and any maintenance issues are handled within 24 hours. The team truly cares!",
            "rating": 5,
            "image": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved": True
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Emily Rodriguez",
            "role": "Property Owner",
            "content": "The renovation crew transformed my outdated property into a modern gem. Now I'm getting top dollar for rent and the property practically manages itself!",
            "rating": 5,
            "image": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=200",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved": True
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Michael Chen",
            "role": "Tenant",
            "content": "Moving to a new city was stressful, but NB Rents made finding my perfect home so easy. Professional, friendly, and always available when I need them.",
            "rating": 4,
            "image": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=200",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approved": True
        }
    ]
    
    await db.testimonials.insert_many(testimonials)
    
    return {"message": "Database seeded successfully", "properties": len(properties), "testimonials": len(testimonials)}

# ==================== ADMIN SETUP ====================

@api_router.post("/seed-users")
async def seed_default_users():
    """Seed default admin and service accounts"""
    default_password = "NBRents2024!"
    hashed = hash_password(default_password)
    
    default_users = [
        {
            "id": str(uuid.uuid4()),
            "email": "admin@nbrents.ca",
            "name": "Admin",
            "phone": "(506) 962-RENT(7368)",
            "user_type": "admin",
            "password": hashed,
            "created_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "email": "service@nbrents.ca",
            "name": "Service Staff",
            "phone": "(506) 962-RENT(7368)",
            "user_type": "service",
            "password": hashed,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    results = {"created": [], "existing": []}
    
    for user in default_users:
        existing = await db.users.find_one({"email": user["email"]})
        if existing:
            # Update password to ensure it's correct
            await db.users.update_one(
                {"email": user["email"]},
                {"$set": {"password": hashed}}
            )
            results["existing"].append(user["email"])
        else:
            await db.users.insert_one(user)
            results["created"].append(user["email"])
    
    return {
        "message": "Default users seeded/updated",
        "password": default_password,
        "results": results
    }

class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    admin_secret: str

@api_router.post("/admin/create")
async def create_admin(admin_data: AdminCreate):
    """Create an admin account (requires admin secret)"""
    # Simple admin secret check - in production, use environment variable
    ADMIN_SECRET = os.environ.get('ADMIN_SECRET', 'nbrents-admin-2024')
    
    if admin_data.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    # Check if admin already exists
    existing = await db.users.find_one({"email": admin_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    admin_id = str(uuid.uuid4())
    admin_dict = {
        "id": admin_id,
        "email": admin_data.email,
        "name": "Admin",
        "phone": None,
        "user_type": "admin",
        "password": hash_password(admin_data.password),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(admin_dict)
    
    return {"message": "Admin account created successfully", "email": admin_data.email}

# ==================== HEALTH CHECK ====================

@api_router.get("/")
async def root():
    return {"message": "NB Rents API is running", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
