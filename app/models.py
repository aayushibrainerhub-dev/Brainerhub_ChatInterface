from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Numeric, Date, Text
from sqlalchemy.orm import relationship
import datetime
from app.database import Base

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    clients = relationship("Client", back_populates="tenant", cascade="all, delete-orphan")
    leads = relationship("Lead", back_populates="tenant", cascade="all, delete-orphan")
    opportunities = relationship("Opportunity", back_populates="tenant", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="tenant", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False)  # 'admin' or 'user'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    company = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="clients")
    leads = relationship("Lead", back_populates="client")

class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    client_id = Column(Integer, ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    
    # In case there's no pre-existing client
    name_fallback = Column(String(150), nullable=True)
    email_fallback = Column(String(255), nullable=True)
    
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="New")  # 'New', 'Contacted', 'Qualified', 'Lost', 'Won'
    value = Column(Numeric(12, 2), default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="leads")
    client = relationship("Client", back_populates="leads")
    opportunities = relationship("Opportunity", back_populates="lead", cascade="all, delete-orphan")

class Opportunity(Base):
    __tablename__ = "opportunities"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    lead_id = Column(Integer, ForeignKey("leads.id", ondelete="SET NULL"), nullable=True)
    
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    stage = Column(String(50), default="Discovery")  # 'Discovery', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost'
    value = Column(Numeric(12, 2), default=0.0)
    probability = Column(Integer, default=10)  # 0 to 100
    close_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="opportunities")
    lead = relationship("Lead", back_populates="opportunities")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="chat_messages")
    user = relationship("User", back_populates="chat_messages")
