from pydantic import BaseModel, Field
from typing import Optional, List
import datetime
from decimal import Decimal

# Base Config
class ConfigBase(BaseModel):
    class Config:
        from_attributes = True

# --- Auth Schemas ---
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    signup_type: str  # "create_tenant" or "join_tenant"
    tenant_name: Optional[str] = None
    tenant_id: Optional[int] = None
    name: str
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None
    tenant_id: Optional[int] = None
    role: Optional[str] = None

# --- Tenant Schemas ---
class TenantResponse(ConfigBase):
    id: int
    name: str
    created_at: datetime.datetime

# --- User Schemas ---
class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str = "user"  # 'admin' or 'user'

class UserResponse(ConfigBase):
    id: int
    tenant_id: int
    email: str
    name: str
    role: str
    created_at: datetime.datetime

# --- Client Schemas ---
class ClientCreate(BaseModel):
    name: str = Field(..., min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None

class ClientResponse(ConfigBase):
    id: int
    tenant_id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    created_at: datetime.datetime

# --- Lead Schemas ---
class LeadCreate(BaseModel):
    client_id: Optional[int] = None
    name_fallback: Optional[str] = None
    email_fallback: Optional[str] = None
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    status: str = "New"  # 'New', 'Contacted', 'Qualified', 'Lost', 'Won'
    value: Decimal = Field(default=Decimal("0.00"))

class LeadUpdate(BaseModel):
    client_id: Optional[int] = None
    name_fallback: Optional[str] = None
    email_fallback: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    value: Optional[Decimal] = None

class LeadResponse(ConfigBase):
    id: int
    tenant_id: int
    client_id: Optional[int] = None
    name_fallback: Optional[str] = None
    email_fallback: Optional[str] = None
    title: str
    description: Optional[str] = None
    status: str
    value: Decimal
    created_at: datetime.datetime
    client: Optional[ClientResponse] = None

# --- Opportunity Schemas ---
class OpportunityCreate(BaseModel):
    lead_id: Optional[int] = None
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    stage: str = "Discovery"  # 'Discovery', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost'
    value: Decimal = Field(default=Decimal("0.00"))
    probability: int = Field(default=10, ge=0, le=100)
    close_date: Optional[datetime.date] = None

class OpportunityUpdate(BaseModel):
    lead_id: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    stage: Optional[str] = None
    value: Optional[Decimal] = None
    probability: Optional[int] = None
    close_date: Optional[datetime.date] = None

class OpportunityResponse(ConfigBase):
    id: int
    tenant_id: int
    lead_id: Optional[int] = None
    title: str
    description: Optional[str] = None
    stage: str
    value: Decimal
    probability: int
    close_date: Optional[datetime.date] = None
    created_at: datetime.datetime
    lead: Optional[LeadResponse] = None

# --- Chat Schemas ---
class ChatMessageCreate(BaseModel):
    content: str

class ChatMessageResponse(ConfigBase):
    id: int
    role: str
    content: str
    created_at: datetime.datetime

class ChatQueryResponse(BaseModel):
    response: str
    sql_executed: Optional[str] = None
    raw_results: Optional[List[dict]] = None
