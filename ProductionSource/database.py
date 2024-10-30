from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

POSTGRESQL_DB_USER = os.environ.get("POSTGRESQL_DB_USER", "postgres")
POSTGRESQL_DB_PASS = os.environ.get("POSTGRESQL_DB_PASS", "thaco%401234")
POSTGRESQL_DB_NAME = os.environ.get("POSTGRESQL_DB_NAME", "corellms")
POSTGRESQL_DB_HOST = os.environ.get("POSTGRESQL_DB_HOST", "10.14.16.30")
POSTGRESQL_DB_PORT = os.environ.get("POSTGRESQL_DB_PORT", 30204)

POSTGRESQL_DB_PORT = int(POSTGRESQL_DB_PORT)

URL_DATABASE = f'postgresql://{POSTGRESQL_DB_USER}:{POSTGRESQL_DB_PASS}@{POSTGRESQL_DB_HOST}:{POSTGRESQL_DB_PORT}/{POSTGRESQL_DB_NAME}'

engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autoflush=False, bind=engine)

Base = declarative_base()
























