from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import bcrypt
import jwt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'nbrents_db')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"

UPLOADS_DIR = ROOT_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NB Rents API")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

class UserCreate(BaseModel):
    email: EmailStr
    name: str
    phone: Optional[str] = None
    password: str
    user_type: str = "tenant"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    phone: Optional[str] = None
    user_type: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class PropertyBase(BaseModel):
    title: str
    description: str
    address: str
    city: str
    price: float
    bedrooms: int
    bathrooms: float
    sqft: int
    property_type: str
    status: str = "available"
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

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    subject: str
    message: str

class ContactResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    email: str
    subject: str
    message: str
    created_at: str

class MaintenanceRequest(BaseModel):
    property_id: str
    title: str
    description: str
    priority: str = "medium"
    tenant_name: Optional[str] = None
    tenant_phone: Optional[str] = None
    tenant_email: Optional[str] = None

class MaintenanceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    property_id: str
    title: str
    description: str
    priority: str
    status: str
    tenant_id: str
    tenant_name: Optional[str] = None
    tenant_phone: Optional[str] = None
    tenant_email: Optional[str] = None
    created_at: str

class TestimonialBase(BaseModel):
    name: str
    role: str
    content: str
    rating: int = 5
    image: Optional[str] = None

class TestimonialResponse(TestimonialBase):
    model_config = ConfigDict(extra="ignore")
    id: str
    created_at: str

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_token(user_id: str) -> str:
    return jwt.encode({"user_id": user_id, "exp": datetime.now(timezone.utc).timestamp() + 86400}, JWT_SECRET, algorithm=JWT_ALGORITHM)

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
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user: UserCreate):
    if await db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_dict = user.model_dump()
    user_dict["id"] = str(uuid.uuid4())
    user_dict["password"] = hash_password(user.password)
    user_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    await db.users.insert_one(user_dict)
    token = create_token(user_dict["id"])
    user_dict.pop("password")
    user_dict.pop("_id", None)
    return TokenResponse(access_token=token, user=UserResponse(**user_dict))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"])
    user.pop("password")
    user.pop("_id", None)
    return TokenResponse(access_token=token, user=UserResponse(**user))

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)

@api_router.get("/properties", response_model=List[PropertyResponse])
async def get_properties(status: Optional[str] = None, city: Optional[str] = None, property_type: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None, bedrooms: Optional[int] = None):
    query = {}
    if status:
        query["status"] = status
    if city:
        query["city"] = {"$regex": city, "$options": "i"}
    if property_type:
        query["property_type"] = property_type
    if min_price is not None:
        query["price"] = {"$gte": min_price}
    if max_price is not None:
        query.setdefault("price", {})["$lte"] = max_price
    if bedrooms is not None:
        query["bedrooms"] = {"$gte": bedrooms}
    properties = await db.properties.find(query, {"_id": 0}).to_list(100)
    return properties

@api_router.get("/properties/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str):
    prop = await db.properties.find_one({"id": property_id}, {"_id": 0})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return prop

@api_router.post("/properties", response_model=PropertyResponse)
async def create_property(property: PropertyCreate, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    prop_dict = property.model_dump()
    prop_dict["id"] = str(uuid.uuid4())
    prop_dict["owner_id"] = current_user["id"]
    prop_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    await db.properties.insert_one(prop_dict)
    prop_dict.pop("_id", None)
    return prop_dict

@api_router.put("/properties/{property_id}", response_model=PropertyResponse)
async def update_property(property_id: str, property: PropertyBase, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    existing = await db.properties.find_one({"id": property_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Property not found")
    update_data = property.model_dump()
    await db.properties.update_one({"id": property_id}, {"$set": update_data})
    updated = await db.properties.find_one({"id": property_id}, {"_id": 0})
    return updated

@api_router.delete("/properties/{property_id}")
async def delete_property(property_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    result = await db.properties.delete_one({"id": property_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Property not found")
    return {"message": "Property deleted"}

@api_router.patch("/properties/{property_id}/featured")
async def toggle_featured(property_id: str, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    prop = await db.properties.find_one({"id": property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    await db.properties.update_one({"id": property_id}, {"$set": {"featured": not prop.get("featured", False)}})
    return {"featured": not prop.get("featured", False)}

@api_router.patch("/properties/{property_id}/status")
async def update_status(property_id: str, status: str, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    if status not in ["available", "rented", "maintenance"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    result = await db.properties.update_one({"id": property_id}, {"$set": {"status": status}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Property not found")
    return {"status": status}

@api_router.post("/upload/image")
async def upload_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin", "tenant"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{ext}"
    with open(UPLOADS_DIR / filename, "wb") as f:
        f.write(contents)
    return {"url": f"/uploads/{filename}"}

@api_router.post("/upload/images")
async def upload_images(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["owner", "admin", "tenant"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    urls = []
    for file in files:
        if file.content_type in ["image/jpeg", "image/png", "image/webp"]:
            contents = await file.read()
            if len(contents) <= 5 * 1024 * 1024:
                ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
                filename = f"{uuid.uuid4()}.{ext}"
                with open(UPLOADS_DIR / filename, "wb") as f:
                    f.write(contents)
                urls.append(f"/uploads/{filename}")
    return {"urls": urls}

@api_router.post("/contact", response_model=ContactResponse)
async def submit_contact(message: ContactMessage):
    contact_dict = message.model_dump()
    contact_dict["id"] = str(uuid.uuid4())
    contact_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    await db.contacts.insert_one(contact_dict)
    contact_dict.pop("_id", None)
    return contact_dict

@api_router.get("/contacts", response_model=List[ContactResponse])
async def get_contacts(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return await db.contacts.find({}, {"_id": 0}).to_list(100)

@api_router.post("/maintenance-requests", response_model=MaintenanceResponse)
async def create_maintenance_request(request: MaintenanceRequest, current_user: dict = Depends(get_current_user)):
    req_dict = request.model_dump()
    req_dict["id"] = str(uuid.uuid4())
    req_dict["tenant_id"] = current_user["id"]
    req_dict["status"] = "open"
    req_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    if not req_dict.get("tenant_name"):
        req_dict["tenant_name"] = current_user.get("name")
    if not req_dict.get("tenant_email"):
        req_dict["tenant_email"] = current_user.get("email")
    await db.maintenance_requests.insert_one(req_dict)
    req_dict.pop("_id", None)
    return req_dict

@api_router.get("/maintenance-requests", response_model=List[MaintenanceResponse])
async def get_maintenance_requests(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] in ["admin", "service"]:
        requests = await db.maintenance_requests.find({}, {"_id": 0}).to_list(500)
    else:
        requests = await db.maintenance_requests.find({"tenant_id": current_user["id"]}, {"_id": 0}).to_list(100)
    return requests

@api_router.put("/maintenance-requests/{request_id}/status")
async def update_maintenance_status(request_id: str, status: str, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["admin", "service"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    if status not in ["open", "pending", "closed"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    result = await db.maintenance_requests.update_one({"id": request_id}, {"$set": {"status": status}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Request not found")
    return {"status": status}

@api_router.get("/service/requests", response_model=List[MaintenanceResponse])
async def get_service_requests(current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["admin", "service"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    return await db.maintenance_requests.find({}, {"_id": 0}).to_list(500)

@api_router.put("/service/requests/{request_id}")
async def update_service_request(request_id: str, status: Optional[str] = None, work_note: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] not in ["admin", "service"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    update = {}
    if status:
        update["status"] = status
    if work_note:
        update["$push"] = {"work_notes": {"note": work_note, "by": current_user["name"], "at": datetime.now(timezone.utc).isoformat()}}
    if update:
        await db.maintenance_requests.update_one({"id": request_id}, {"$set": update} if "$push" not in update else update)
    updated = await db.maintenance_requests.find_one({"id": request_id}, {"_id": 0})
    return updated

@api_router.get("/testimonials", response_model=List[TestimonialResponse])
async def get_testimonials():
    return await db.testimonials.find({}, {"_id": 0}).to_list(50)

@api_router.post("/testimonials", response_model=TestimonialResponse)
async def create_testimonial(testimonial: TestimonialBase, current_user: dict = Depends(get_current_user)):
    if current_user["user_type"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    t_dict = testimonial.model_dump()
    t_dict["id"] = str(uuid.uuid4())
    t_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    await db.testimonials.insert_one(t_dict)
    t_dict.pop("_id", None)
    return t_dict

@api_router.post("/seed-users")
async def seed_users():
    users = [
        {"email": "admin@nbrents.ca", "name": "Admin", "user_type": "admin", "password": "NBRents2024!"},
        {"email": "service@nbrents.ca", "name": "Service Staff", "user_type": "service", "password": "NBRents2024!"}
    ]
    for u in users:
        if not await db.users.find_one({"email": u["email"]}):
            user_dict = {"id": str(uuid.uuid4()), "email": u["email"], "name": u["name"], "user_type": u["user_type"], "password": hash_password(u["password"]), "created_at": datetime.now(timezone.utc).isoformat()}
            await db.users.insert_one(user_dict)
    return {"message": "Users seeded"}

@api_router.post("/seed-data")
async def seed_data():
    if await db.properties.count_documents({}) == 0:
        properties = [
            {"title": "Modern Downtown Apartment", "description": "Beautiful apartment", "address": "123 Main Street", "city": "Moncton", "price": 1850, "bedrooms": 2, "bathrooms": 1, "sqft": 950, "property_type": "apartment", "status": "available", "amenities": ["parking"], "images": [], "featured": True, "latitude": 46.0878, "longitude": -64.7782},
            {"title": "Cozy Family Home", "description": "Spacious home", "address": "456 Oak Avenue", "city": "Moncton", "price": 2400, "bedrooms": 4, "bathrooms": 2.5, "sqft": 2200, "property_type": "house", "status": "available", "amenities": ["parking", "garage"], "images": [], "featured": True, "latitude": 46.0950, "longitude": -64.8010},
            {"title": "Luxury Waterfront Condo", "description": "Waterfront views", "address": "789 Harbor Drive", "city": "Saint John", "price": 3200, "bedrooms": 3, "bathrooms": 2, "sqft": 1800, "property_type": "condo", "status": "rented", "amenities": ["parking", "gym"], "images": [], "featured": True, "latitude": 45.2733, "longitude": -66.0633},
            {"title": "Charming Townhouse", "description": "Starter home", "address": "321 Maple Lane", "city": "Fredericton", "price": 1950, "bedrooms": 3, "bathrooms": 1.5, "sqft": 1400, "property_type": "townhouse", "status": "available", "amenities": ["parking"], "images": [], "featured": False, "latitude": 45.9636, "longitude": -66.6431}
        ]
        for p in properties:
            p["id"] = str(uuid.uuid4())
            p["created_at"] = datetime.now(timezone.utc).isoformat()
            await db.properties.insert_one(p)
    if await db.testimonials.count_documents({}) == 0:
        testimonials = [{"name": "Sarah Mitchell", "role": "Property Owner", "content": "NB Rents transformed my rental property!", "rating": 5}, {"name": "James Thompson", "role": "Tenant", "content": "Best rental experience!", "rating": 5}]
        for t in testimonials:
            t["id"] = str(uuid.uuid4())
            t["created_at"] = datetime.now(timezone.utc).isoformat()
            await db.testimonials.insert_one(t)
    return {"message": "Data seeded"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

app.include_router(api_router)

app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)