# Import library for build API
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Annotated, List, Optional
from uuid import UUID, uuid4
import uuid
import models
from database import engine, SessionLocal
from database import *
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from auth import get_current_user, check_role, create_access_token, create_refresh_token
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import psycopg2
from psycopg2 import sql
from minio import Minio
from minio.error import S3Error
from pathlib import Path
from PIL import Image
from create_bot_k8s import create_bot_k8s
from delete_bot_k8s import delete_bot_k8s
import json

# Import library for langchain server
import bs4
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
import tempfile
from dotenv import load_dotenv


############################# Langchain #################################
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", '10.14.16.30')
CHROMA_DB_PORT = os.environ.get("CHROMA_DB_PORT", 32123)
CHUNK_SIZE = os.environ.get("CHUNK_SIZE", 400)
CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP", 80)

CHUNK_SIZE = int(CHUNK_SIZE)
CHUNK_OVERLAP = int(CHUNK_OVERLAP)
CHROMA_DB_PORT = int(CHROMA_DB_PORT)
client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)

################################ API ####################################
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "chatbotllms")
MINIO_EPT = os.environ.get("MINIO_EPT", "minio.prod.bangpdk.dev")
MINIO_EPT_DOMAIN = os.environ.get("MINIO_EPT_DOMAIN", "minio.prod.bangpdk.dev")
app = FastAPI(title="Chatbot Back End",
              description="Back End deploy chatbot using Langchain with OpenAI")
models.Base.metadata.create_all(bind=engine)

#-----------------------------CREATE ROLE, MODEL OPENAI NAME AND SUPER USER AT FIRST------------------------#
PASS_DB_TEMP = os.environ.get("PASS_DB_TEMP", "@1234")
connection = psycopg2.connect(
    host=POSTGRESQL_DB_HOST,
    port=POSTGRESQL_DB_PORT,
    database=POSTGRESQL_DB_NAME,
    user=POSTGRESQL_DB_USER,
    password=PASS_DB_TEMP
)

cursor = connection.cursor()

role_id = str(uuid4())
role_name = "superuser"

model_openai_id = str(uuid4())
model_openai_name = "gpt-4o-mini"

acc_id_sup = str(uuid4())
acc_username_sup = "superuser"
acc_email_sup = "qyangtuyennguyen0299@gmail.com"
acc_hashed_password_sup = "$argon2id$v=19$m=65536,t=3,p=4$Y8xZi1HKuZdyTgmhtNaaUw$VUtf0JcoyR5Hqk0QiERscPq/DHmlHpJn7jx2E4PZ1kM"
acc_role_sup = role_id
acc_image_sup = f"https://{MINIO_EPT}/{MINIO_BUCKET_NAME}/anh1.jpg"
acc_created_at_sup = "2024-10-05 09:15:50.463435+00"
acc_model_openai_id = model_openai_id

check_query = sql.SQL(
    """
    SELECT * FROM roles WHERE name = %s
    """
)

check_query_account = sql.SQL(
    """
    SELECT * FROM accounts where username = %s
    """
)

check_query_model_openai = sql.SQL(
    """
    SELECT * FROM modelopenais WHERE name = %s
    """
)

try:
    cursor.execute(check_query, (role_name,))
    existing_role = cursor.fetchone()  # select 1 row if it exist
    cursor.execute(check_query_model_openai, (model_openai_name,))
    existing_model_openai = cursor.fetchone()  # select 1 row if it exist
    cursor.execute(check_query_account, (role_name,))
    existing_acc = cursor.fetchone()  # select 1 row if it exist

    if existing_role:
        print(f"Vai trò '{role_name}' đã tồn tại, không cần thêm.")
    else:
        insert_query = sql.SQL(
            """
            INSERT INTO roles (id, name)
            VALUES (%s, %s)
            """
        )
        cursor.execute(insert_query, (role_id, role_name))
        connection.commit()  # Xác nhận thay đổi vào database
        print(f"Đã thêm vai trò '{role_name}' thành công.")

    if existing_model_openai:
        print(f"Tên model '{model_openai_name}' đã tồn tại, không cần thêm.")
    else:
        insert_query = sql.SQL(
            """
            INSERT INTO modelopenais (id, name)
            VALUES (%s, %s)
            """
        )
        cursor.execute(insert_query, (model_openai_id, model_openai_name))
        connection.commit()  # Xác nhận thay đổi vào database
        print(f"Đã thêm tên model openai '{model_openai_name}' thành công.")

    if existing_acc:
        print(f"Tài khoản superuser đã tồn tại, không cần thêm.")
    else:
        cursor.execute(check_query, (role_name,))
        existing_role_2 = cursor.fetchone()  # select 1 row if it exist
        cursor.execute(check_query_model_openai, (model_openai_name,))
        existing_model_openai_2 = cursor.fetchone()  # select 1 row if it exist

        insert_query_account = sql.SQL(
            """
            INSERT INTO accounts (id, username, email, hashed_password, role_id, image, created_at, openai_api_key, 
            model_openai_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        )
        cursor.execute(insert_query_account, (acc_id_sup, acc_username_sup, acc_email_sup, acc_hashed_password_sup,
                                              existing_role_2[0], acc_image_sup, acc_created_at_sup, acc_openai_api_key,
                                              existing_model_openai_2[0]))
        try:
            try:
                collection = client.create_collection(name=acc_id_sup)
                print(f"Collection '{acc_id_sup}' đã được tạo thành công.")
            except Exception as bug:
                print("bug: ", bug)
                print(f"Collection '{acc_id_sup}' đã tồn tại.")
            create_bot_k8s(acc_id_sup, acc_openai_api_key, model_openai_name)
            connection.commit()  # Xác nhận thay đổi vào database
            print(f"Đã thêm tài khoản 'superuser' và tạo bot cho 'superuser' thành công.")
        except Exception as bug:
            client.delete_collection(acc_id_sup)
            print(f"Collection '{acc_id_sup}' đã bị xóa.")
            print("Bug in create bot: ", bug)
            print("Không thể tạo account superuser do có lỗi xảy ra khi tạo bot.")


except Exception as e:
    print("Có lỗi xảy ra:", e)
    connection.rollback()


cursor.close()
connection.close()
#-----------------------------CREATE ROLE, MODEL OPENAI NAME AND SUPER USER AT FIRST------------------------#

# Config MinIO client
minio_client = Minio(
    endpoint=os.environ.get("MINIO_ENDPOINT", "minio.prod.bangpdk.dev"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", ""),
    secret_key=os.environ.get("MINIO_SECRET_KEY", ""),
    secure=False  # set True if MinIO server use HTTPS
)


# create bucket if not exist
if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
    minio_client.make_bucket(MINIO_BUCKET_NAME)


def check_permissions(account_role: str, required_role: str):
    if account_role != required_role:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập.")


class StoryCreate(BaseModel):
    name: str
    account_id: UUID


class StoryUpdate(BaseModel):
    name: str


class StoryResponse(BaseModel):
    id: UUID
    name: str
    account_id: UUID
    created_at: datetime


class StepCreate(BaseModel):
    qna: List[str]
    story: UUID


class StepResponse(BaseModel):
    id: UUID
    qna: List[str]
    created_at: datetime
    story: UUID


class InputData(BaseModel):
    text: str
    story: UUID


class AccountCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UUID


class AccountResponse(BaseModel):
    id: UUID
    username: str
    email: str
    role: UUID
    created_at: datetime


class AccountUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    openai_api_key: Optional[str] = None
    model_openai_name_id: Optional[UUID] = None
    role_id: Optional[UUID] = None


class LoginModel(BaseModel):
    username: str
    password: str


class LoginRefreshToken(BaseModel):
    refresh_token: str


class RoleCreate(BaseModel):
    name: str


class ModelOpenAICreate(BaseModel):
    name: str


class LinkWeb(BaseModel):
    link: str


class BucketCreate(BaseModel):
    bucket_name: str
    bucket_description: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (nếu muốn giới hạn, thay "*" bằng danh sách các URL cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE,...)
    allow_headers=["*"],  # Cho phép tất cả các headers
)


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class Hasher():
    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)


@app.post("/account/", tags=["User Management"])
async def create_account(username: str,
                         email: EmailStr,
                         password: str,
                         role: UUID,
                         openai_api_key: str,
                         name_model_openai_id: UUID,
                         db: db_dependency,
                         image:  Optional[UploadFile] = File(None),
                         current_user: models.Accounts = Depends(get_current_user)):

    check_role(current_user, ["superuser", "admin"])

    existing_account = db.query(models.Accounts).filter(models.Accounts.username == username).first()
    if existing_account:
        raise HTTPException(status_code=400, detail="Tên tài khoản đã tồn tại.")

    hashed_password = Hasher.get_password_hash(password)

    ept = os.environ.get("MINIO_ENDPOINT", "minio.prod.bangpdk.dev")

    if image:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="File phải ở định dạng JPEG hoặc PNG")

        try:
            img = Image.open(image.file)
            img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail="File ảnh không hợp lệ.")

        file_extension = image.filename.split(".")[-1]
        new_filename = f"{uuid.uuid4()}.{file_extension}"

        image.file.seek(0, os.SEEK_END)
        file_size = image.file.tell()

        image.file.seek(0)

        # Upload file to MinIO
        try:
            minio_client.put_object(
                bucket_name=MINIO_BUCKET_NAME,
                object_name=new_filename,
                data=image.file,
                length=file_size,  # use -1 if not identify file length
                content_type=image.content_type
            )
        except S3Error as e:
            raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi upload ảnh lên MinIO")

        image_url = f"https://{ept}/{MINIO_BUCKET_NAME}/{new_filename}"
    else:
        image_url = f"https://{ept}/{MINIO_BUCKET_NAME}/anh1.jpg"
    # print("image url: ", image_url)
    new_account = models.Accounts(
        id=uuid.uuid4(),
        username=username,
        email=email,
        hashed_password=hashed_password,
        role_id=role,
        openai_api_key=openai_api_key,
        model_openai_id=name_model_openai_id,
        image=image_url,
        created_at=datetime.now()
    )

    try:
        model_openai = db.query(models.ModelOpenAIs).filter(models.ModelOpenAIs.id == name_model_openai_id).first()
        if not model_openai:
            raise HTTPException(status_code=404, detail="Không tìm thấy model openai với ID đã cho.")
        try:
            try:
                collection_name = client.create_collection(name=new_account.id)
                print(f"Collection '{new_account.id}' đã được tạo thành công.")
            except Exception as bug:
                print("bug: ", bug)
                print(f"Collection '{new_account.id}' đã tồn tại.")
                raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi tạo collection do đó không thể tạo account")
            db.add(new_account)
            db.commit()
            db.refresh(new_account)
            create_bot_k8s(new_account.id, openai_api_key, model_openai.name)
        except Exception as bug:
            client.delete_collection(str(new_account.id))
            print(f"Collection '{new_account.id}' đã bị xóa.")
            print("Bug in create bot: ", bug)
            raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi tạo bot do đó không thể tạo account.")
        return new_account
    except Exception as e:
        db.rollback()
        print("Bug in create account: ", e)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo tài khoản.")


@app.get("/accounts/", tags=["User Management"])
def get_accounts(db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    try:
        accounts = db.query(models.Accounts).order_by(models.Accounts.created_at.asc()).all()
        return accounts
    except Exception as e:
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi lấy accounts")


@app.get("/account/{account_id}", tags=["User Management"])
async def get_account_detail(account_id: UUID, db: db_dependency,
                             current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    account = db.query(models.Accounts).filter(models.Accounts.id == account_id).first()

    if account is None:
        raise HTTPException(status_code=404, detail="Account không tồn tại.")

    response = {
            "id": account.id,
            "username": account.username,
            "email": account.email,
            "role": account.role.name,
            "model OpenAI": account.model_openai.name,
            "OpenAI API key": account.openai_api_key,
            "image": account.image
    }
    return response


@app.patch("/account/{account_id}", tags=["User Management"])
def update_account(account_id: UUID, account_data: AccountUpdate, db: db_dependency,
                   current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    account_to_update = db.query(models.Accounts).filter(models.Accounts.id == str(account_id)).first()
    if account_to_update is None:
        raise HTTPException(status_code=404, detail="Account không tồn tại.")

    existing_account = db.query(models.Accounts).filter(models.Accounts.username == account_data.username).first()
    if existing_account and existing_account.username != account_data.username:
        raise HTTPException(status_code=400, detail="Tên account đã tồn tại. Vui lòng chọn tên khác.")

    # account_to_update.username = account_data.username
    # account_to_update.email = account_data.email
    # account_to_update.hashed_password = hashed_password
    # account_to_update.openai_api_key = account_data.openai_api_key
    # account_to_update.model_openai_id = account_data.model_openai_name_id
    # account_to_update.role_id = account_data.role_id

    if account_data.username:
        account_to_update.username = account_data.username
    if account_data.email:
        account_to_update.email = account_data.email
    if account_data.password:
        hashed_password = Hasher.get_password_hash(account_data.password)
        account_to_update.hashed_password = hashed_password
    if account_data.openai_api_key:
        account_to_update.openai_api_key = account_data.openai_api_key
    if account_data.model_openai_name_id:
        account_to_update.model_openai_id = account_data.model_openai_name_id
    if account_data.role_id:
        account_to_update.role_id = account_data.role_id

    try:
        db.commit()
        db.refresh(account_to_update)
        return account_to_update
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi cập nhật account.")


@app.delete("/account/{account_id}", tags=["User Management"])
async def delete_account(account_id: UUID, db: db_dependency,
                         current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])

    account_to_delete = db.query(models.Accounts).filter(models.Accounts.id == account_id).first()
    if account_to_delete is None:
        raise HTTPException(status_code=404, detail="Account không tồn tại.")

    if account_to_delete.role.name == "superuser":
        raise HTTPException(status_code=404, detail="Không thể xóa tài khoản superuser.")

    related_stories = db.query(models.Stories).filter(models.Stories.account_id == account_id).all()
    try:
        try:
            client.delete_collection(str(account_id))
            delete_bot_k8s(account_id)
        except Exception as bug:
            print("Bug in delete Bot when delete account: ", bug)
            raise HTTPException(status_code=404, detail="Không thể xóa tài khoản do không thể xóa bot tương ứng.")

        # delete stories in acccount
        for story in related_stories:
            related_steps = db.query(models.Steps).filter(models.Steps.story_id == story.id).all()
            for step in related_steps:
                db.delete(step)
            db.delete(story)
        # delete account
        db.delete(account_to_delete)
        db.commit()
        return {"message": "Account và tất cả các stories liên quan đã được xóa thành công."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi xóa account và các stories.")


# @app.post("/login", tags=["Login Management"])
# def login(user: LoginModel, db: Session = Depends(get_db)):
#     account = db.query(models.Accounts).filter(models.Accounts.username == user.username).first()
#     if not account or not Hasher.verify_password(user.password, account.hashed_password):
#         raise HTTPException(status_code=401, detail="Tên tài khoản hoặc mật khẩu không đúng.")
#     access_token = create_access_token(str(account.id))
#     return {"access_token": access_token,
#             "account_id": account.id,
#             "role": account.role.name,
#             "image": account.image,
#             "user_name": account.username}


@app.post("/login", tags=["Login Management"])
def login(user: LoginModel, db: Session = Depends(get_db)):
    account = db.query(models.Accounts).filter(models.Accounts.username == user.username).first()
    if not account or not Hasher.verify_password(user.password, account.hashed_password):
        raise HTTPException(status_code=401, detail="Tên tài khoản hoặc mật khẩu không đúng.")

    access_token = create_access_token(str(account.id))
    refresh_token_create = create_refresh_token(str(account.id))

    access_token_expiry = datetime.utcnow() + timedelta(minutes=5)
    refresh_token_expiry = datetime.utcnow() + timedelta(days=7)

    token_entry = models.Tokens(
        account_id=account.id,
        access_token=access_token,
        refresh_token=refresh_token_create,
        access_token_expiry=access_token_expiry,
        refresh_token_expiry=refresh_token_expiry
    )
    db.add(token_entry)
    db.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token_create,
        "account_id": account.id,
        "role": account.role.name,
        "image": account.image,
        "user_name": account.username
    }


@app.post("/refresh-token", tags=["Login Management"])
def refresh_token(refresh: LoginRefreshToken, db: Session = Depends(get_db)):
    token_entry = db.query(models.Tokens).filter(models.Tokens.refresh_token == refresh.refresh_token).first()

    if not token_entry:
        raise HTTPException(status_code=401, detail="Refresh token không hợp lệ hoặc đã bị thu hồi.")

    if token_entry.refresh_token_expiry < datetime.utcnow().replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token đã hết hạn.")

    account_id = token_entry.account_id

    new_access_token = create_access_token(str(token_entry.account_id))
    access_token_expiry = datetime.utcnow() + timedelta(minutes=5)

    new_refresh_token = create_refresh_token(str(token_entry.account_id))
    refresh_token_expiry = datetime.utcnow() + timedelta(days=7)

    # token_entry.access_token = new_access_token
    # token_entry.access_token_expiry = access_token_expiry
    # token_entry.refresh_token = new_refresh_token
    # token_entry.refresh_token_expiry = refresh_token_expiry

    new_token = models.Tokens(
        id=uuid4(),
        account_id=account_id,
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        access_token_expiry=access_token_expiry,
        refresh_token_expiry=refresh_token_expiry
    )
    db.add(new_token)
    db.delete(token_entry)
    db.commit()

    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
    }


@app.get("/roles/", tags=["Role Management"])
def get_roles(db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])
    try:
        roles = db.query(models.Roles).all()
        return roles
    except Exception as e:
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi lấy roles")


@app.post("/role", tags=["Role Management"])
def create_role(add_role: RoleCreate, db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])
    existing_roles = db.query(models.Roles).filter(models.Roles.name == add_role.name).first()
    if existing_roles:
        raise HTTPException(status_code=400, detail="Tên role đã tồn tại. Vui lòng chọn tên khác.")

    new_role = models.Roles(
        id=uuid4(),
        name=add_role.name
    )
    try:
        db.add(new_role)
        db.commit()
        db.refresh(new_role)
        return new_role
    except Exception as e:
        db.rollback()
        print("---------------------------")
        print(e)
        print("---------------------------")
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo role")


@app.delete("/role/{role_id_delete}", tags=["Role Management"])
def delete_role(role_id_delete: UUID, db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])

    role_to_delete = db.query(models.Roles).filter(models.Roles.id == role_id_delete).first()
    if role_to_delete is None:
        raise HTTPException(status_code=404, detail="Role không tồn tại.")

    related_accounts = db.query(models.Accounts).filter(models.Accounts.role_id == role_id_delete)

    for account in related_accounts:
        if account.id is not None:
            raise HTTPException(status_code=404, detail="Role đã được thiết lập trong account nên không thể xóa.")

    if role_to_delete.name == "superuser":
        raise HTTPException(status_code=404, detail="Không thể xóa role superuser.")

    try:
        db.delete(role_to_delete)
        db.commit()
        return {"message": "Role đã được xóa thành công."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi xóa role.")


@app.get("/modelopenais/", tags=["Model OpenAI Management"])
def get_models_openai(db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])
    try:
        models_openai = db.query(models.ModelOpenAIs).all()
        return models_openai
    except Exception as e:
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi lấy danh sách models openAI")


@app.post("/modelopenai", tags=["Model OpenAI Management"])
def create_model_openai(add_model_openai: ModelOpenAICreate, db: db_dependency,
                        current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])
    existing_models_openai = db.query(models.ModelOpenAIs).filter(models.ModelOpenAIs.name == add_model_openai.name).first()
    if existing_models_openai:
        raise HTTPException(status_code=400, detail="Tên model openai đã tồn tại. Vui lòng chọn tên khác.")

    new_model_openai = models.ModelOpenAIs(
        id=uuid4(),
        name=add_model_openai.name
    )
    try:
        db.add(new_model_openai)
        db.commit()
        db.refresh(new_model_openai)
        return new_model_openai
    except Exception as e:
        db.rollback()
        print("---------------------------")
        print(e)
        print("---------------------------")
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo tên model openai.")


@app.delete("/modelopenai/{model_openai_id_delete}", tags=["Model OpenAI Management"])
def delete_modelopenai(model_openai_id_delete: UUID, db: db_dependency,
                       current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser"])

    model_openai_to_delete = db.query(models.ModelOpenAIs).filter(models.ModelOpenAIs.id == model_openai_id_delete).first()
    if model_openai_to_delete is None:
        raise HTTPException(status_code=404, detail="Tên model openAI không tồn tại.")

    related_accounts = db.query(models.Accounts).filter(models.Accounts.model_openai_id == model_openai_id_delete)

    for account in related_accounts:
        if account.id is not None:
            raise HTTPException(status_code=404, detail="Tên model openai đã được thiết "
                                                        "lập trong account nên không thể xóa.")
    try:
        db.delete(model_openai_to_delete)
        db.commit()
        return {"message": "Tên model OpenAI đã được xóa thành công."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi xóa tên model OpenAI.")


@app.post("/story", tags=["Stories Management"])
def create_stories(story: StoryCreate, db: db_dependency,
                   current_user: models.Accounts = Depends(get_current_user)):

    check_role(current_user, ["superuser", "admin", "user"])

    existing_story = db.query(models.Stories).filter(models.Stories.name == story.name).first()
    if existing_story:
        raise HTTPException(status_code=400, detail="Tên story đã tồn tại. Vui lòng chọn tên khác.")

    new_story = models.Stories(
        id=uuid4(),
        name=story.name,
        account_id=story.account_id,
        created_at=datetime.now()
    )
    try:
        db.add(new_story)
        db.commit()
        db.refresh(new_story)
        return new_story
    except Exception as e:
        db.rollback()
        print("-----------------------")
        print(e)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo story")


@app.get("/stories/", response_model=List[StoryResponse], tags=["Stories Management"])
def get_stories(db: db_dependency, current_user: models.Accounts = Depends(get_current_user),
                account_id: Optional[UUID] = None):
    check_role(current_user, ["superuser", "admin", "user"])

    try:
        stories = db.query(models.Stories).filter(models.Stories.account_id == account_id).all()
        return stories
    except Exception as e:
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi lấy stories")


@app.get("/stories/{story_id}", tags=["Stories Management"])
async def get_story_detail(story_id: UUID, db: db_dependency,
                           current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    story = db.query(models.Stories).filter(models.Stories.id == story_id).first()

    if story is None:
        raise HTTPException(status_code=404, detail="Story không tồn tại.")

    steps = db.query(models.Steps).filter(models.Steps.story_id == story_id)\
        .order_by(models.Steps.created_at.asc()).all()

    response = {
        "story": {
            "id": story.id,
            "name": story.name,
            "account": story.account_id,
            "created_at": story.created_at,
        },
        "steps": [
            {
                "id": step.id,
                "qna": step.qna,
                "created_at": step.created_at,
                "story_id": step.story_id
            }
            for step in steps
        ]
    }
    return response


@app.delete("/story/{story_id}", tags=["Stories Management"])
async def delete_story(story_id: UUID, db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    story_to_delete = db.query(models.Stories).filter(models.Stories.id == story_id).first()
    if story_to_delete is None:
        raise HTTPException(status_code=404, detail="Story không tồn tại.")
    related_steps = db.query(models.Steps).filter(models.Steps.story_id == story_id).all()
    try:
        # delete steps in stories
        for step in related_steps:
            db.delete(step)
        # delete story
        db.delete(story_to_delete)
        db.commit()
        return {"message": "Story và tất cả các steps liên quan đã được xóa thành công."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi xóa story và các steps.")


@app.put("/story/{story_id}", response_model=StoryResponse, tags=["Stories Management"])
def update_story(story_id: UUID, story_data: StoryUpdate, db: db_dependency,
                 current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    story_to_update = db.query(models.Stories).filter(models.Stories.id == story_id).first()
    if story_to_update is None:
        raise HTTPException(status_code=404, detail="Story không tồn tại.")

    existing_story = db.query(models.Stories).filter(models.Stories.name == story_data.name).first()
    if existing_story and existing_story.name != story_data.name:
        raise HTTPException(status_code=400, detail="Tên story đã tồn tại. Vui lòng chọn tên khác.")

    story_to_update.name = story_data.name

    try:
        db.commit()
        db.refresh(story_to_update)
        return story_to_update
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi cập nhật story.")


@app.post("/step", tags=["Steps Management"])
async def create_step(step: StepCreate, db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    new_step = models.Steps(
        id=uuid4(),
        qna=step.qna,
        created_at=datetime.now(),
        story_id=step.story
    )
    try:
        db.add(new_step)
        db.commit()
        db.refresh(new_step)
        return new_step
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo step")


@app.post("/uploadfiletxt/", tags=["Upload Data Management"])
async def upload_file_txt(file: UploadFile = File(...), current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    if file.content_type != 'text/plain':
        return {"error": "File phải ở định dạng .txt"}
    try:
        content = await file.read()
        file_content = content.decode("utf-8")

        # save temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(file_content.encode('utf-8'))
            temp_file_path = temp_file.name

        loader = TextLoader(temp_file_path, encoding='utf-8')
        docs = loader.load()
        print("Đã load file txt")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        print("Đã tách từ tài liệu mới")

        new_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(openai_api_key=current_user.openai_api_key),
            collection_name=str(current_user.id),
            client=client
        )
        print("Vector store đã được tải từ server chroma")

        new_vectorstore.add_documents(documents=splits)

        print("Embedding mới đã được thêm và lưu trữ vào chroma server")

        return {"message": "Embedding mới đã được thêm và lưu trữ vào chroma server."}
    except Exception as bug:
        print("Lỗi khi thêm mới file txt: ", bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra trong quá trình embedd file txt.")


@app.post("/uploadfilePDF/", tags=["Upload Data Management"])
async def upload_file_pdf(file: UploadFile = File(...), current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])

    if file.content_type != 'application/pdf':
        return {"error": "File phải ở định dạng .pdf"}

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(file_path=temp_file_path)
        docs = loader.load()

        if len(docs) == 0:
            return HTTPException(status_code=400, detail="File pdf không hợp lệ!")

        print("Đã load file PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        print("Đã tách từ tài liệu PDF")

        new_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(openai_api_key=current_user.openai_api_key),
            collection_name=str(current_user.id),
            client=client
        )
        print("Vector store đã được tải từ server Chroma")

        new_vectorstore.add_documents(documents=splits)

        print("Embedding mới đã được thêm và lưu trữ vào Chroma server")

        return {"message": "Embedding mới đã được thêm và lưu trữ vào Chroma server."}
    except Exception as bug:
        print("Có lỗi xảy ra khi embedd file pdf: ", bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra trong quá trình embedd file pdf.")


@app.post("/uploadlinkweb/", tags=["Upload Data Management"])
def upload_link_web(link: LinkWeb, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    try:
        loader = WebBaseLoader(
            web_paths=(f"{link.link}",),
            encoding='utf-8',
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        docs = loader.load()
        if len(docs) == 0:
            return HTTPException(status_code=400, detail="Link không hợp lệ!")

        print("Đã load link")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        print("Splits:", splits)

        print("Đã tách từ tài liệu mới")

        new_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(openai_api_key=current_user.openai_api_key),
            collection_name=str(current_user.id),
            client=client
        )
        print("Vector store đã được tải từ server chroma")

        new_vectorstore.add_documents(documents=splits)

        print("Embedding mới đã được thêm và lưu trữ vào chroma server")

        return {"message": "Embedding mới đã được thêm và lưu trữ vào chroma server."}
    except Exception as bug:
        print("Lỗi khi embedd link web: ", bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra trong quá trình embedd link web.")


@app.post("/delete_data_train/", tags=["Upload Data Management"])
def delete_data(current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    try:
        # client.delete_collection(str(current_user.id))
        collection_vecto = client.get_collection(str(current_user.id))
        vectors = collection_vecto.get()
        ids = vectors['ids']
        collection_vecto.delete(ids=ids)
        return {"message": "Đã xóa toàn bộ vector embedding trong collection."}
    except Exception as bug:
        print(bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra trong quá trình xóa.")


@app.post("/bucket", tags=["Bucket Management"])
async def create_bucket(bucket: BucketCreate,
                        db: db_dependency,
                        current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])

    existing_bucket_name = db.query(models.MinioFileUpload) \
        .filter(models.MinioFileUpload.bucket_name == bucket.bucket_name).first()
    if existing_bucket_name:
        raise HTTPException(status_code=400, detail="Tên bucket đã tồn tại.")

    existing_bucket_description = db.query(models.MinioFileUpload)\
        .filter(models.MinioFileUpload.bucket_description == bucket.bucket_description).first()
    if existing_bucket_description:
        raise HTTPException(status_code=400, detail="Bucket description đã tồn tại.")

    new_bucket = models.MinioFileUpload(
        id=uuid4(),
        bucket_name=bucket.bucket_name,
        bucket_description=bucket.bucket_description
    )
    try:
        if not minio_client.bucket_exists(bucket.bucket_name):
            minio_client.make_bucket(bucket.bucket_name)
        public_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket.bucket_name}/*"]
                }
            ]
        }
        minio_client.set_bucket_policy(bucket.bucket_name, json.dumps(public_policy))
    except Exception as bug:
        print("bug in create bucket minio: ", bug)
        raise HTTPException(status_code=400, detail="Không thể tạo bucket trên Minio.")

    try:
        db.add(new_bucket)
        db.commit()
        db.refresh(new_bucket)
        return new_bucket
    except Exception as bug:
        db.rollback()
        print("Bug in create bucket DB: ", bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo bucket trên DB")


@app.get("/buckets", tags=["Bucket Management"])
def get_buckets(db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])

    try:
        buckets = db.query(models.MinioFileUpload).all()
        return buckets
    except Exception as bug:
        print("Bug in get buckets: ", bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi lấy buckets")


@app.get("/bucket/{bucket_id}", tags=["Bucket Management"])
async def get_bucket_detail(bucket_id: UUID, db: db_dependency,
                            current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    bucket = db.query(models.MinioFileUpload).filter(models.MinioFileUpload.id == bucket_id).first()

    if bucket is None:
        raise HTTPException(status_code=404, detail="Bucket không tồn tại.")

    bucket_name = bucket.bucket_name
    files = []
    if minio_client.bucket_exists(bucket_name):
        objects = minio_client.list_objects(bucket_name)
        for obj in objects:
            # print(obj.object_name)
            files.append(f"https://{MINIO_EPT_DOMAIN}/{bucket_name}/{obj.object_name}")
        response = {"id": bucket.id,
                    "bucket_name": bucket.bucket_name,
                    "bucket_description": bucket.bucket_description,
                    "files": files}
        return response
    else:
        print(f"Bucket '{bucket_name}' does not exist.")
        raise HTTPException(status_code=404, detail="Bucket không tồn tại.")


@app.post("/delete_file_in_bucket", tags=["Bucket Management"])
async def delete_file_in_bucket(bucket_name: str,
                                file_name: str, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    if minio_client.bucket_exists(bucket_name):
        try:
            minio_client.remove_object(bucket_name, file_name)
            print(f"File '{file_name}' đã được xóa khỏi bucket '{bucket_name}'.")
            return f"File '{file_name}' đã được xóa khỏi bucket '{bucket_name}'. "
        except Exception as bug:
            print(f"Lỗi khi xóa file khỏi bucket: {bug}")
            raise HTTPException(status_code=400, detail="Có lỗi khi xóa file trong bucket.")
    else:
        raise HTTPException(status_code=404, detail="Bucket không tồn tại.")


@app.delete("/bucket/{bucket_id}", tags=["Bucket Management"])
async def delete_bucket(bucket_id: UUID, db: db_dependency,
                        current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    bucket_to_delete = db.query(models.MinioFileUpload).filter(models.MinioFileUpload.id == bucket_id).first()
    if bucket_to_delete is None:
        raise HTTPException(status_code=404, detail="Bucket không tồn tại.")
    try:
        objects = minio_client.list_objects(bucket_to_delete.bucket_name, recursive=True)
        for obj in objects:
            minio_client.remove_object(bucket_to_delete.bucket_name, obj.object_name)
        minio_client.remove_bucket(bucket_to_delete.bucket_name)
        db.delete(bucket_to_delete)
        db.commit()
        print(f"Bucket '{bucket_to_delete.bucket_name}' đã được xóa.")
        return "Đã xóa bucket và các file trong bucket."
    except Exception as bug:
        print("Bug khi xóa bucket: ", bug)
        db.rollback()
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi xóa bucket và các files.")


@app.post("/upload_form/", tags=["Bucket Management"])
async def up_form(bucket_name: str,
                  file: UploadFile = File(...),
                  current_user: models.Accounts = Depends(get_current_user)):

    check_role(current_user, ["superuser", "admin"])

    object_name = file.filename

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, object_name)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        minio_client.fput_object(bucket_name, object_name, file_path)
        return {"message": f"File '{object_name}' đã được upload thành công tới bucket '{bucket_name}'."}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi upload file: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


