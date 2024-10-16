import uuid
from sqlalchemy import ForeignKey, Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class Roles(Base):
    __tablename__ = 'roles'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    accounts = relationship("Accounts", back_populates="role")


class ModelOpenAIs(Base):
    __tablename__ = 'modelopenais'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    accounts = relationship("Accounts", back_populates="modelopenai")


class Accounts(Base):
    __tablename__ = 'accounts'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), nullable=False)
    role = relationship("Roles", back_populates="accounts")
    image = Column(String, default='anh1.jpg', nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    stories = relationship("Stories", back_populates="account")
    openai_api_key = Column(String, unique=False, nullable=False)
    model_openai_id = Column(UUID(as_uuid=True), ForeignKey('modelopenais.id'), nullable=False)
    model_openai = relationship("ModelOpenAIs", back_populates="accounts")


class Stories(Base):
    __tablename__ = 'stories'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    account_id = Column(UUID(as_uuid=True), ForeignKey('accounts.id'), nullable=False)
    account = relationship("Accounts", back_populates="stories")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    steps = relationship("Steps", back_populates="story")


class Steps(Base):
    __tablename__ = 'steps'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    qna = Column(JSON, nullable=False)
    story_id = Column(UUID(as_uuid=True), ForeignKey('stories.id'))
    story = relationship("Stories", back_populates="steps")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


