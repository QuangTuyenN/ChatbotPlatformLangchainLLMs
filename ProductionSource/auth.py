import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from database import SessionLocal
from models import Accounts
from typing import List


class Settings(BaseModel):
    authjwt_secret_key: str = "bimatcuanguyenquangtuyen"
    authjwt_algorithm: str = "HS256"
    authjwt_access_token_expires: int = 60  # token will expire in 60 minutes


settings = Settings()

security = HTTPBearer()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(account_id: str):
    expires_delta = timedelta(minutes=settings.authjwt_access_token_expires)
    to_encode = {"sub": account_id, "exp": datetime.utcnow() + expires_delta}
    encoded_jwt = jwt.encode(to_encode, settings.authjwt_secret_key, algorithm=settings.authjwt_algorithm)
    return encoded_jwt


def get_current_user(db: Session = Depends(get_db), token: str = Security(security)):
    credentials_exception = HTTPException(
        status_code=401, detail="Token không hợp lệ hoặc đã hết hạn."
    )

    try:
        payload = jwt.decode(token.credentials, settings.authjwt_secret_key, algorithms=[settings.authjwt_algorithm])
        account_id: str = payload.get("sub")
        if account_id is None:
            raise credentials_exception
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token đã hết hạn.")
    except jwt.JWTError:
        raise credentials_exception

    account = db.query(Accounts).filter(Accounts.id == account_id).first()
    if account is None:
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại.")

    return account


def check_role(current_user: Accounts, allowed_roles: List[str]):
    if current_user.role.name not in allowed_roles:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập.")












# from fastapi import Depends, HTTPException
# from fastapi_jwt_auth import AuthJWT
# from fastapi_jwt_auth.exceptions import AuthJWTException
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from database import SessionLocal
# from models import Accounts
#
#
# # Config JWT
# class Settings(BaseModel):
#     authjwt_secret_key: str = "bimatcuanguyenquangtuyen"
#
#
# @AuthJWT.load_config
# def get_config():
#     return Settings()
#
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
#
# # Valid account
# def get_current_user(db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
#     try:
#         Authorize.jwt_required()
#         account_id = Authorize.get_jwt_subject()
#
#         account = db.query(Accounts).filter(Accounts.id == account_id).first()
#         if account is None:
#             raise HTTPException(status_code=404, detail="Tài khoản không tồn tại.")
#         return account
#     except AuthJWTException as e:
#         raise HTTPException(status_code=401, detail="Token không hợp lệ.")
#
#
# def check_role(current_user: Accounts, allowed_roles: list[str]):
#     if current_user.role_id.name not in allowed_roles:
#         raise HTTPException(status_code=403, detail="Không có quyền truy cập.")

