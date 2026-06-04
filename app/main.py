from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import os

from app.database import get_db, init_db, hash_password
from app.models import Tenant, User, Client, Lead, Opportunity, ChatMessage
from app.schemas import (
    LoginRequest, TokenResponse, UserResponse, UserCreate,
    ClientCreate, ClientUpdate, ClientResponse,
    LeadCreate, LeadUpdate, LeadResponse,
    OpportunityCreate, OpportunityUpdate, OpportunityResponse,
    ChatMessageCreate, ChatMessageResponse, ChatQueryResponse,
    RegisterRequest, TenantResponse
)
from app.auth import create_access_token, get_current_user, require_admin, require_user, verify_password
from app.chat_engine import CRMQueryEngine

# Initialize DB on startup
init_db()

app = FastAPI(title="Brainerhub Chat CRM", description="Multi-tenant CRM with LLM-powered Chat-with-Data")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat Engine Instance
chat_engine = CRMQueryEngine()

# --- Authentication Endpoints ---
@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    # Create token payload containing tenant_id and role
    token_data = {
        "email": user.email,
        "user_id": user.id,
        "tenant_id": user.tenant_id,
        "role": user.role
    }
    
    access_token = create_access_token(data=token_data)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/api/auth/tenants", response_model=List[TenantResponse])
def get_all_tenants(db: Session = Depends(get_db)):
    return db.query(Tenant).order_by(Tenant.name.asc()).all()

@app.post("/api/auth/register", response_model=TokenResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # Check if email is already taken
    existing_user = db.query(User).filter(User.email == request.email.lower().strip()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )
        
    if request.signup_type == "create_tenant":
        if not request.tenant_name or not request.tenant_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workspace (Tenant) name is required.",
            )
        # Create a new tenant
        new_tenant = Tenant(name=request.tenant_name.strip())
        db.add(new_tenant)
        db.commit()
        db.refresh(new_tenant)
        tenant_id = new_tenant.id
        role = "admin"  # Creator of a tenant is the Admin
    elif request.signup_type == "join_tenant":
        if not request.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant selection is required to join an existing workspace.",
            )
        # Verify tenant exists
        tenant_exists = db.query(Tenant).filter(Tenant.id == request.tenant_id).first()
        if not tenant_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The selected workspace does not exist.",
            )
        tenant_id = request.tenant_id
        role = "user"  # Joining user gets the customer/user role
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signup type. Must be 'create_tenant' or 'join_tenant'.",
        )
        
    # Create user
    db_user = User(
        tenant_id=tenant_id,
        email=request.email.lower().strip(),
        hashed_password=hash_password(request.password),
        name=request.name.strip(),
        role=role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Generate Token
    token_data = {
        "email": db_user.email,
        "user_id": db_user.id,
        "tenant_id": db_user.tenant_id,
        "role": db_user.role
    }
    access_token = create_access_token(data=token_data)
    return {"access_token": access_token, "token_type": "bearer"}


# --- Dashboard Endpoints ---
@app.get("/api/dashboard")
def get_dashboard_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Verify Admin access
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dashboard is restricted to Administrator role.",
        )
        
    tenant_id = current_user.tenant_id
    
    # 1. Total Clients
    total_clients = db.query(Client).filter(Client.tenant_id == tenant_id).count()
    
    # 2. Total Leads
    total_leads = db.query(Lead).filter(Lead.tenant_id == tenant_id).count()
    
    # 3. Total Lead Value
    lead_value_sum = db.query(func.coalesce(func.sum(Lead.value), 0)).filter(Lead.tenant_id == tenant_id).scalar()
    
    # 4. Lead Status Distribution
    status_results = db.query(Lead.status, func.count(Lead.id)).filter(Lead.tenant_id == tenant_id).group_by(Lead.status).all()
    lead_status_dist = {status: count for status, count in status_results}
    
    # 5. Total Opportunities
    total_opportunities = db.query(Opportunity).filter(Opportunity.tenant_id == tenant_id).count()
    
    # 6. Sales Pipeline Value (sum of Opportunities)
    pipeline_value = db.query(func.coalesce(func.sum(Opportunity.value), 0)).filter(Opportunity.tenant_id == tenant_id).scalar()
    
    # 7. Opportunity Stage Distribution
    stage_results = db.query(Opportunity.stage, func.count(Opportunity.id)).filter(Opportunity.tenant_id == tenant_id).group_by(Opportunity.stage).all()
    opportunity_stage_dist = {stage: count for stage, count in stage_results}

    # 8. Recent Leads
    recent_leads_query = db.query(Lead).filter(Lead.tenant_id == tenant_id).order_by(Lead.created_at.desc()).limit(5).all()
    recent_leads = [
        {
            "id": l.id,
            "title": l.title,
            "status": l.status,
            "value": float(l.value),
            "created_at": l.created_at
        } for l in recent_leads_query
    ]
    
    # 9. Recent Opportunities
    recent_opps_query = db.query(Opportunity).filter(Opportunity.tenant_id == tenant_id).order_by(Opportunity.created_at.desc()).limit(5).all()
    recent_opps = [
        {
            "id": o.id,
            "title": o.title,
            "stage": o.stage,
            "value": float(o.value),
            "probability": o.probability,
            "created_at": o.created_at
        } for o in recent_opps_query
    ]
    
    return {
        "total_clients": total_clients,
        "total_leads": total_leads,
        "total_lead_value": float(lead_value_sum),
        "lead_status_distribution": lead_status_dist,
        "total_opportunities": total_opportunities,
        "pipeline_value": float(pipeline_value),
        "opportunity_stage_distribution": opportunity_stage_dist,
        "recent_leads": recent_leads,
        "recent_opportunities": recent_opps
    }


# --- Clients Endpoints ---
@app.get("/api/clients", response_model=List[ClientResponse])
def list_clients(current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(Client).filter(Client.tenant_id == current_user.tenant_id).order_by(Client.name.asc()).all()

@app.post("/api/clients", response_model=ClientResponse)
def create_client(client_data: ClientCreate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_client = Client(
        tenant_id=current_user.tenant_id,
        name=client_data.name,
        email=client_data.email,
        phone=client_data.phone,
        company=client_data.company
    )
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client

@app.put("/api/clients/{client_id}", response_model=ClientResponse)
def update_client(client_id: int, client_data: ClientUpdate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_client = db.query(Client).filter(Client.id == client_id, Client.tenant_id == current_user.tenant_id).first()
    if not db_client:
        raise HTTPException(status_code=404, detail="Client not found")
        
    for key, value in client_data.model_dump(exclude_unset=True).items():
        setattr(db_client, key, value)
        
    db.commit()
    db.refresh(db_client)
    return db_client

@app.delete("/api/clients/{client_id}")
def delete_client(client_id: int, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_client = db.query(Client).filter(Client.id == client_id, Client.tenant_id == current_user.tenant_id).first()
    if not db_client:
        raise HTTPException(status_code=404, detail="Client not found")
        
    db.delete(db_client)
    db.commit()
    return {"detail": "Client deleted successfully"}


# --- Leads Endpoints ---
@app.get("/api/leads", response_model=List[LeadResponse])
def list_leads(current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(Lead).filter(Lead.tenant_id == current_user.tenant_id).order_by(Lead.created_at.desc()).all()

@app.post("/api/leads", response_model=LeadResponse)
def create_lead(lead_data: LeadCreate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    # Validate client if client_id is provided
    if lead_data.client_id:
        client = db.query(Client).filter(Client.id == lead_data.client_id, Client.tenant_id == current_user.tenant_id).first()
        if not client:
            raise HTTPException(status_code=400, detail="Invalid client ID specified")

    db_lead = Lead(
        tenant_id=current_user.tenant_id,
        client_id=lead_data.client_id,
        name_fallback=lead_data.name_fallback,
        email_fallback=lead_data.email_fallback,
        title=lead_data.title,
        description=lead_data.description,
        status=lead_data.status,
        value=lead_data.value
    )
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    return db_lead

@app.put("/api/leads/{lead_id}", response_model=LeadResponse)
def update_lead(lead_id: int, lead_data: LeadUpdate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_lead = db.query(Lead).filter(Lead.id == lead_id, Lead.tenant_id == current_user.tenant_id).first()
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    if lead_data.client_id:
        client = db.query(Client).filter(Client.id == lead_data.client_id, Client.tenant_id == current_user.tenant_id).first()
        if not client:
            raise HTTPException(status_code=400, detail="Invalid client ID specified")
        
    for key, value in lead_data.model_dump(exclude_unset=True).items():
        setattr(db_lead, key, value)
        
    db.commit()
    db.refresh(db_lead)
    return db_lead

@app.delete("/api/leads/{lead_id}")
def delete_lead(lead_id: int, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_lead = db.query(Lead).filter(Lead.id == lead_id, Lead.tenant_id == current_user.tenant_id).first()
    if not db_lead:
        raise HTTPException(status_code=404, detail="Lead not found")
        
    db.delete(db_lead)
    db.commit()
    return {"detail": "Lead deleted successfully"}


# --- Opportunities Endpoints ---
@app.get("/api/opportunities", response_model=List[OpportunityResponse])
def list_opportunities(current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(Opportunity).filter(Opportunity.tenant_id == current_user.tenant_id).order_by(Opportunity.created_at.desc()).all()

@app.post("/api/opportunities", response_model=OpportunityResponse)
def create_opportunity(opp_data: OpportunityCreate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    # Validate lead if lead_id is provided
    if opp_data.lead_id:
        lead = db.query(Lead).filter(Lead.id == opp_data.lead_id, Lead.tenant_id == current_user.tenant_id).first()
        if not lead:
            raise HTTPException(status_code=400, detail="Invalid lead ID specified")

    db_opp = Opportunity(
        tenant_id=current_user.tenant_id,
        lead_id=opp_data.lead_id,
        title=opp_data.title,
        description=opp_data.description,
        stage=opp_data.stage,
        value=opp_data.value,
        probability=opp_data.probability,
        close_date=opp_data.close_date
    )
    db.add(db_opp)
    db.commit()
    db.refresh(db_opp)
    return db_opp

@app.put("/api/opportunities/{opp_id}", response_model=OpportunityResponse)
def update_opportunity(opp_id: int, opp_data: OpportunityUpdate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_opp = db.query(Opportunity).filter(Opportunity.id == opp_id, Opportunity.tenant_id == current_user.tenant_id).first()
    if not db_opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    if opp_data.lead_id:
        lead = db.query(Lead).filter(Lead.id == opp_data.lead_id, Lead.tenant_id == current_user.tenant_id).first()
        if not lead:
            raise HTTPException(status_code=400, detail="Invalid lead ID specified")
        
    for key, value in opp_data.model_dump(exclude_unset=True).items():
        setattr(db_opp, key, value)
        
    db.commit()
    db.refresh(db_opp)
    return db_opp

@app.delete("/api/opportunities/{opp_id}")
def delete_opportunity(opp_id: int, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_opp = db.query(Opportunity).filter(Opportunity.id == opp_id, Opportunity.tenant_id == current_user.tenant_id).first()
    if not db_opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")
        
    db.delete(db_opp)
    db.commit()
    return {"detail": "Opportunity deleted successfully"}


# --- Chat Endpoints ---
@app.get("/api/chat/history", response_model=List[ChatMessageResponse])
def get_chat_history(current_user: User = Depends(require_user), db: Session = Depends(get_db)):
    # Chat is only for users except Admin
    return db.query(ChatMessage).filter(
        ChatMessage.tenant_id == current_user.tenant_id,
        ChatMessage.user_id == current_user.id
    ).order_by(ChatMessage.created_at.asc()).limit(50).all()

@app.post("/api/chat", response_model=ChatQueryResponse)
def send_chat_message(message_data: ChatMessageCreate, current_user: User = Depends(require_user), db: Session = Depends(get_db)):
    # 1. Save user message to database
    user_msg = ChatMessage(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        role="user",
        content=message_data.content
    )
    db.add(user_msg)
    db.commit()
    
    # 2. Process query via LLM engine
    query_result = chat_engine.ask(
        prompt=message_data.content,
        tenant_id=current_user.tenant_id,
        db=db
    )
    
    # 3. Save assistant message to database
    assistant_msg = ChatMessage(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        role="assistant",
        content=query_result["response"]
    )
    db.add(assistant_msg)
    db.commit()
    
    # 4. Return results
    return ChatQueryResponse(
        response=query_result["response"],
        sql_executed=query_result["sql_executed"],
        raw_results=query_result["raw_results"]
    )


# --- Settings / Configuration Endpoints ---
@app.get("/api/settings")
def get_settings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
    return {
        "user": {
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "role": current_user.role,
        },
        "tenant": {
            "id": tenant.id,
            "name": tenant.name,
        },
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip())
    }

@app.get("/api/settings/users", response_model=List[UserResponse])
def get_tenant_users(current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    # Admin can list all users in their tenant
    return db.query(User).filter(User.tenant_id == current_user.tenant_id).order_by(User.name.asc()).all()

@app.post("/api/settings/users", response_model=UserResponse)
def create_tenant_user(user_data: UserCreate, current_user: User = Depends(require_admin), db: Session = Depends(get_db)):
    # Check if email is already taken
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )
        
    db_user = User(
        tenant_id=current_user.tenant_id,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        name=user_data.name,
        role=user_data.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# Serve Static Front-end
# Verify that the static directory exists, if not, we create it dynamically.
static_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)
    os.makedirs(os.path.join(static_path, "css"))
    os.makedirs(os.path.join(static_path, "js"))

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
