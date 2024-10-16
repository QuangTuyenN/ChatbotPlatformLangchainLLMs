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
from datetime import datetime
from auth import get_current_user, check_role, create_access_token
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

# Import library for langchain server
import bs4
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from list_tools import list_tools_use
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
import tempfile
import time


############################# Langchain #################################
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-proj-s5YkjN9E5jhGY8aovG5YT3BlbkFJZwa0SeTc60uRPpcRsYCF")
MODEL_OPENAI = os.environ.get("MODEL_OPENAI", "gpt-4o-mini")
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", '10.14.16.30')
CHROMA_DB_PORT = os.environ.get("CHROMA_DB_PORT", 30745)
CHROMA_DB_COLLECTION_NAME = os.environ.get("CHROMA_DB_COLLECTION_NAME", "thaco_collection3")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE", 400)
CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP", 80)

CHUNK_SIZE = int(CHUNK_SIZE)
CHUNK_OVERLAP = int(CHUNK_OVERLAP)
CHROMA_DB_PORT = int(CHROMA_DB_PORT)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model=MODEL_OPENAI)
client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
t1 = time.time()
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_DB_COLLECTION_NAME,
    client=client
)
t2 = time.time()
print("vectostore time: ", t2 - t1)
print("vectostore: ", type(vectorstore))
print("Đã kết nối tới server Chroma và sẵn sàng truy vấn")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
t4 = time.time()
print("delta retrieve: ", t4 - t1)
contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng \
có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập \
có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \
chỉ cần định dạng lại nó nếu cần và nếu không thì trả lại như cũ."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_system_prompt = """Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. \
Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. \
Nếu bạn không tìm được câu trả lời từ đoạn ngữ cảnh, hãy sử dụng dữ liệu bạn đã được huấn luyện sẵn để trả lời. \
Những câu hỏi xã giao ví dụ xin chào, tạm biệt thì không cần phải truy xuất ngữ cảnh. \
Nếu vẫn không thể trả lời được bạn cứ trả lời là xin lỗi vì bạn bị thiếu dữ liệu. \
Những câu trả lời cần truy cập vào internet để lấy thì bạn vẫn phải truy cập không được trả lời xin lỗi. \
Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn.\
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

rag_tools = create_retriever_tool(
    history_aware_retriever,
    "search_thaco_info",
    "Tìm kiếm và trả về những thông tin về đoạn ngữ cảnh cung cấp.",
)

tools = [rag_tools]

for tool in list_tools_use:
    tools.append(tool)

chat_history = []

agent = create_openai_tools_agent(llm, tools, qa_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
t3 = time.time()
print("delta 3: ", t3 - t1)
################################ API ####################################
MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "chatbotllms")
MINIO_EPT = os.environ.get("MINIO_EPT", "10.14.16.30:31003")
app = FastAPI(title="Chatbot Back End",
              description="Back End deploy chatbot using Langchain with OpenAI")
models.Base.metadata.create_all(bind=engine)

#-----------------------------CREATE ROLE AND SUPER USER AT FIRST------------------------#
connection = psycopg2.connect(
    host=POSTGRESQL_DB_HOST,
    port=POSTGRESQL_DB_PORT,
    database=POSTGRESQL_DB_NAME,
    user=POSTGRESQL_DB_USER,
    password="thaco@1234"
)

cursor = connection.cursor()

role_id = "99910203-0405-0607-0809-0a0b0c0d0e09"
role_name = "superuser"

acc_id_sup = "68f2b1a1-1206-48e5-9fd6-16e2afb21c99"
acc_username_sup = "superuser"
acc_email_sup = "teamaithacoindustries@gmail.com"
acc_hashed_password_sup = "$argon2id$v=19$m=65536,t=3,p=4$Y8xZi1HKuZdyTgmhtNaaUw$VUtf0JcoyR5Hqk0QiERscPq/DHmlHpJn7jx2E4PZ1kM"
acc_role_sup = role_id
acc_image_sup = f"http://{MINIO_EPT}/{MINIO_BUCKET_NAME}/anh1.jpg"
acc_created_at_sup = "2024-10-05 09:15:50.463435+00"

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

try:
    cursor.execute(check_query, (role_name,))
    existing_role = cursor.fetchone()  # select 1 row if it exist
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

    if existing_acc:
        print(f"Tài khoản superuser đã tồn tại, không cần thêm.")
    else:
        insert_query_account = sql.SQL(
            """
            INSERT INTO accounts (id, username, email, hashed_password, role_id, image, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
        )
        cursor.execute(insert_query_account, (acc_id_sup, acc_username_sup, acc_email_sup, acc_hashed_password_sup,
                                              acc_role_sup, acc_image_sup, acc_created_at_sup))
        connection.commit()  # Xác nhận thay đổi vào database
        print(f"Đã thêm tài khoản super user thành công.")

except Exception as e:
    print("Có lỗi xảy ra:", e)
    connection.rollback()


cursor.close()
connection.close()
#-----------------------------CREATE ROLE AND SUPER USER AT FIRST------------------------#

# Config MinIO client
minio_client = Minio(
    endpoint=os.environ.get("MINIO_ENDPOINT", "10.14.16.30:31003"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", "teamaithaco"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "thaco@1234"),
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
    username: str
    email: EmailStr
    password: str
    openai_api_key: str
    model_openai_id: UUID


class LoginModel(BaseModel):
    username: str
    password: str


class RoleCreate(BaseModel):
    name: str


class ModelOpenAICreate(BaseModel):
    name: str


class LinkWeb(BaseModel):
    link: str


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


# @app.post("/account/", tags=["User Management"])
# def create_account(account: AccountCreate, db: db_dependency,
#                    current_user: models.Accounts = Depends(get_current_user)):
#     check_role(current_user, ["superuser", "admin"])
#     existing_account = db.query(models.Accounts).filter(models.Accounts.username == account.username).first()
#     if existing_account:
#         raise HTTPException(status_code=400, detail="Tên tài khoản đã tồn tại.")
#
#     hashed_password = Hasher.get_password_hash(account.password)
#
#     new_account = models.Accounts(
#         id=uuid4(),
#         username=account.username,
#         email=account.email,
#         hashed_password=hashed_password,
#         role_id=account.role,
#         created_at=datetime.now()
#     )
#
#     try:
#         db.add(new_account)
#         db.commit()
#         db.refresh(new_account)
#         return new_account
#     except Exception as e:
#         db.rollback()
#         print("======================")
#         print(e)
#         raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo tài khoản.")

@app.post("/account/", tags=["User Management"])
async def create_account(username: str,
                         email: EmailStr,
                         password: str,
                         role: UUID,
                         openai_api_key: str,
                         model_openai_id: UUID,
                         db: db_dependency,
                         image: UploadFile = File(...),
                         current_user: models.Accounts = Depends(get_current_user)):

    check_role(current_user, ["superuser", "admin"])

    existing_account = db.query(models.Accounts).filter(models.Accounts.username == username).first()
    if existing_account:
        raise HTTPException(status_code=400, detail="Tên tài khoản đã tồn tại.")

    hashed_password = Hasher.get_password_hash(password)

    ept = os.environ.get("MINIO_ENDPOINT", "10.14.16.30:31003")

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

        image_url = f"http://{ept}/{MINIO_BUCKET_NAME}/{new_filename}"
    else:
        image_url = f"http://{ept}/{MINIO_BUCKET_NAME}/anh1.jpg"
    # print("image url: ", image_url)
    new_account = models.Accounts(
        id=uuid.uuid4(),
        username=username,
        email=email,
        hashed_password=hashed_password,
        role_id=role,
        openai_api_key=openai_api_key,
        model_openai_id=model_openai_id,
        image=image_url,
        created_at=datetime.now()
    )

    try:
        db.add(new_account)
        db.commit()
        db.refresh(new_account)
        model_openai = db.query(models.ModelOpenAIs).filter(models.ModelOpenAIs.id == model_openai_id).first()
        if not model_openai:
            raise HTTPException(status_code=404, detail="Không tìm thấy model_openai với ID đã cho.")
        try:
            create_bot_k8s(new_account.id, openai_api_key, model_openai.name)
        except Exception as bug:
            print("Bug in create bot: ", bug)
            raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi tạo bot.")
        return new_account
    except Exception as e:
        db.rollback()
        print("Bug in create account: ", e)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra khi tạo tài khoản.")


@app.get("/accounts/", tags=["User Management"])
def get_accounts(db: db_dependency, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    try:
        accounts = db.query(models.Accounts).all()
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


@app.put("/account/{account_id}", tags=["User Management"])
def update_account(account_id: UUID, account_data: AccountUpdate, db: db_dependency,
                   current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    account_to_update = db.query(models.Accounts).filter(models.Accounts.id == account_id).first()
    if account_to_update is None:
        raise HTTPException(status_code=404, detail="Account không tồn tại.")

    existing_account = db.query(models.Accounts).filter(models.Accounts.username == account_data.username).first()
    if existing_account and existing_account.username != account_data.username:
        raise HTTPException(status_code=400, detail="Tên account đã tồn tại. Vui lòng chọn tên khác.")

    hashed_password = Hasher.get_password_hash(account_data.password)

    account_to_update.username = account_data.username
    account_to_update.email = account_data.email
    account_to_update.hashed_password = hashed_password
    account_to_update.openai_api_key = account_data.openai_api_key
    account_to_update.model_openai_id = account_data.model_openai_id

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

# @app.post("/login")
# async def login(user: LoginModel, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
#     account = db.query(models.Accounts).filter(models.Accounts.username == user.username).first()
#
#     if not account or not verify_password(user.password, account.hashed_password):
#         raise HTTPException(status_code=401, detail="Tên tài khoản hoặc mật khẩu không đúng.")
#
#     # Tạo token
#     access_token = Authorize.create_access_token(subject=str(account.id), user_claims={"role": account.role.name})
#
#     return {"access_token": access_token}


@app.post("/login", tags=["Login Management"])
def login(user: LoginModel, db: Session = Depends(get_db)):
    account = db.query(models.Accounts).filter(models.Accounts.username == user.username).first()
    if not account or not Hasher.verify_password(user.password, account.hashed_password):
        raise HTTPException(status_code=401, detail="Tên tài khoản hoặc mật khẩu không đúng.")
    access_token = create_access_token(str(account.id))
    return {"access_token": access_token,
            "account_id": account.id}


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

    model_openai_to_delete = db.query(models.Roles).filter(models.ModelOpenAIs.id == model_openai_id_delete).first()
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


@app.post("/process", tags=["Process Question Management"])
async def process_data(input_data: InputData, db: db_dependency,
                       current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin", "user"])
    nhap = input_data.text
    story_id = input_data.story
    story = db.query(models.Stories).filter(models.Stories.id == story_id).first()
    if story is None:
        raise HTTPException(status_code=404, detail="Story không tồn tại.")
    steps = db.query(models.Steps).filter(models.Steps.story_id == story_id)\
        .order_by(models.Steps.created_at.asc()).all()
    list_qna = []
    if len(steps) == 0:
        list_qna = []
    elif 1 <= len(steps) <= 3:
        for step in steps:
            list_qna.append(step.qna[0])
            list_qna.append(step.qna[1])
    elif len(steps) > 3:
        for step in steps[-3:]:
            list_qna.append(step.qna[0])
            list_qna.append(step.qna[1])

    print("list qna: ", list_qna)

    try:
        retrieved_context = history_aware_retriever.invoke({"input": nhap, "chat_history": list_qna})
        input_data = {
            "input": nhap,
            "context": retrieved_context,
            "chat_history": list_qna
        }
        rep = agent_executor.invoke(input_data)
        bot_response = rep["output"]
    except Exception as bug:
        bot_response = "Xin lỗi nhưng tôi không có thông tin để trả lời câu hỏi của bạn."

    qna = [nhap, bot_response]

    new_step = models.Steps(
        id=uuid4(),
        qna=qna,
        created_at=datetime.now(),
        story_id=story_id
    )
    db.add(new_step)
    db.commit()
    db.refresh(new_step)
    return {
        "reply": bot_response
    }


@app.post("/uploadfiletxt/", tags=["Upload Data Management"])
async def upload_file_txt(file: UploadFile = File(...), current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    if file.content_type != 'text/plain':
        return {"error": "File phải ở định dạng .txt"}
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
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_DB_COLLECTION_NAME,
        client=client
    )
    print("Vector store đã được tải từ server chroma")

    new_vectorstore.add_documents(documents=splits)

    print("Embedding mới đã được thêm và lưu trữ vào chroma server")

    return {"message": "Embedding mới đã được thêm và lưu trữ vào chroma server."}


@app.post("/uploadfilePDF/", tags=["Upload Data Management"])
async def upload_file_pdf(file: UploadFile = File(...), current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])

    if file.content_type != 'application/pdf':
        return {"error": "File phải ở định dạng .pdf"}

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
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_DB_COLLECTION_NAME,
        client=client
    )
    print("Vector store đã được tải từ server Chroma")

    new_vectorstore.add_documents(documents=splits)

    print("Embedding mới đã được thêm và lưu trữ vào Chroma server")

    return {"message": "Embedding mới đã được thêm và lưu trữ vào Chroma server."}


@app.post("/uploadlinkweb/", tags=["Upload Data Management"])
def upload_link_web(link: LinkWeb, current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])

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
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_DB_COLLECTION_NAME,
        client=client
    )
    print("Vector store đã được tải từ server chroma")

    new_vectorstore.add_documents(documents=splits)

    print("Embedding mới đã được thêm và lưu trữ vào chroma server")

    return {"message": "Embedding mới đã được thêm và lưu trữ vào chroma server."}


@app.post("/delete_data_train/", tags=["Upload Data Management"])
def delete_data(current_user: models.Accounts = Depends(get_current_user)):
    check_role(current_user, ["superuser", "admin"])
    try:
        client.delete_collection(CHROMA_DB_COLLECTION_NAME)
        return {"message": "Đã xóa toàn bộ vector embedding trong collection."}
    except Exception as bug:
        print(bug)
        raise HTTPException(status_code=400, detail="Có lỗi xảy ra trong quá trình xóa.")
