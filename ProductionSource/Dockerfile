# pull official base image
FROM python:3.9

# copy project
COPY . .

# install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install langchain langchain_community langchain_chroma
RUN pip install -qU langchain-openai
RUN pip install chromadb==0.5.3
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install psycopg2-binary
RUN pip install python-multipart
RUN pip install pyjwt
RUN pip install pydantic[email]
RUN pip install passlib[argon2]
RUN pip install bs4
RUN pip install psycopg2
RUN pip install minio
RUN pip install pillow

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1234"]
RUN chmod +x start.sh
ENTRYPOINT ["bash", "start.sh"]








