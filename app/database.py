import os
import hashlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

# Load env variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables.")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password hashing utility for seeding
def hash_password(password: str) -> str:
    # Use PBKDF2 HMAC SHA-256 for secure hashing without external dependencies
    salt = b"crm_tenant_salt_2026"
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return pwd_hash.hex()

def seed_db(db):
    from app.models import Tenant, User, Client, Lead, Opportunity
    import datetime

    # Check if we already have tenants
    if db.query(Tenant).first():
        print("Database already seeded.")
        return

    print("Seeding database...")

    # 1. Create Tenants
    acme = Tenant(name="Acme Corp")
    globex = Tenant(name="Globex Corp")
    db.add_all([acme, globex])
    db.commit()
    db.refresh(acme)
    db.refresh(globex)

    # 2. Create Users
    hashed_admin_pw = hash_password("admin123")
    hashed_user_pw = hash_password("user123")

    users = [
        # Acme Corp
        User(tenant_id=acme.id, email="admin@acme.com", hashed_password=hashed_admin_pw, name="Alice Admin", role="admin"),
        User(tenant_id=acme.id, email="user@acme.com", hashed_password=hashed_user_pw, name="Bob User", role="user"),
        # Globex Corp
        User(tenant_id=globex.id, email="admin@globex.com", hashed_password=hashed_admin_pw, name="Charlie Admin", role="admin"),
        User(tenant_id=globex.id, email="user@globex.com", hashed_password=hashed_user_pw, name="Dave User", role="user")
    ]
    db.add_all(users)
    db.commit()

    # 3. Create Clients
    clients_acme = [
        Client(tenant_id=acme.id, name="John Doe", email="john@doe.com", phone="123-456-7890", company="Apex Industries"),
        Client(tenant_id=acme.id, name="Jane Smith", email="jane@smith.com", phone="987-654-3210", company="Stark Enterprises")
    ]
    clients_globex = [
        Client(tenant_id=globex.id, name="Bruce Wayne", email="bruce@wayne.com", phone="555-0199", company="Wayne Enterprises"),
        Client(tenant_id=globex.id, name="Clark Kent", email="clark@dailyplanet.com", phone="555-0144", company="Daily Planet")
    ]
    db.add_all(clients_acme + clients_globex)
    db.commit()
    
    # Refresh to get IDs
    for c in clients_acme + clients_globex:
        db.refresh(c)

    # 4. Create Leads
    leads_acme = [
        Lead(tenant_id=acme.id, client_id=clients_acme[0].id, title="Enterprise Software License", description="Interested in purchasing 500 licenses of our core product.", status="New", value=15000.0),
        Lead(tenant_id=acme.id, client_id=clients_acme[1].id, title="Cloud Migration Consultation", description="Wants an assessment for migrating their on-premise infrastructure.", status="Qualified", value=8500.0),
        Lead(tenant_id=acme.id, client_id=None, name_fallback="Tony Stark", email_fallback="tony@stark.com", title="Custom Iron Suit SaaS integration", description="Incoming request for high-security container orchestration.", status="Contacted", value=120000.0)
    ]
    leads_globex = [
        Lead(tenant_id=globex.id, client_id=clients_globex[0].id, title="Batcave IT Security Systems", description="Upgrade high-performance computing clusters and security firewalls.", status="New", value=250000.0),
        Lead(tenant_id=globex.id, client_id=clients_globex[1].id, title="Metropolis Newspaper Digitization", description="Wants to scan and digitize historical archives.", status="Won", value=5000.0)
    ]
    db.add_all(leads_acme + leads_globex)
    db.commit()

    for l in leads_acme + leads_globex:
        db.refresh(l)

    # 5. Create Opportunities
    opportunities_acme = [
        Opportunity(
            tenant_id=acme.id, 
            lead_id=leads_acme[1].id, 
            title="Stark Cloud Migration Project", 
            description="Negotiations on cloud setup project.", 
            stage="Proposal", 
            value=8500.0, 
            probability=60, 
            close_date=datetime.date.today() + datetime.timedelta(days=30)
        ),
        Opportunity(
            tenant_id=acme.id, 
            lead_id=leads_acme[2].id, 
            title="Iron Suit SaaS Integration SLA", 
            description="Discussing contract details and service levels.", 
            stage="Negotiation", 
            value=120000.0, 
            probability=80, 
            close_date=datetime.date.today() + datetime.timedelta(days=15)
        )
    ]
    opportunities_globex = [
        Opportunity(
            tenant_id=globex.id, 
            lead_id=leads_globex[0].id, 
            title="Wayne Enterprises Datacenter Security Upgrade", 
            description="Large scale computing infrastructure contract.", 
            stage="Discovery", 
            value=250000.0, 
            probability=30, 
            close_date=datetime.date.today() + datetime.timedelta(days=90)
        )
    ]
    db.add_all(opportunities_acme + opportunities_globex)
    db.commit()

    print("Database seeding completed.")

def init_db():
    from app.models import Tenant, User, Client, Lead, Opportunity, ChatMessage
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_db(db)
    finally:
        db.close()
