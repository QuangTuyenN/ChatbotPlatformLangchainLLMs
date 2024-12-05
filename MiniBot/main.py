from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from list_tools import list_tools_use
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb

############################# Langchain #################################
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-proj-s5YkjN9E5jhGY8aovG5YT3BlbkFJZwa0SeTc60uRPpcRsYCF")
MODEL_OPENAI = os.environ.get("MODEL_OPENAI", "gpt-4o-mini")
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", '10.14.16.30')
CHROMA_DB_PORT = os.environ.get("CHROMA_DB_PORT", 32123)
CHROMA_DB_COLLECTION_NAME = os.environ.get("CHROMA_DB_COLLECTION_NAME", "0273e9ca-df53-4c42-94e5-5548f4a5bbd2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))
CHROMA_DB_PORT = int(CHROMA_DB_PORT)

client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)

# Khởi tạo vectorstore lần đầu
vectorstore = None


def init_vectorstore():
    global vectorstore
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        collection_name=CHROMA_DB_COLLECTION_NAME,
        client=client
    )
    print("Vectorstore đã được reload.")


# Tạo retriever và agent executor
def create_retriever_and_agent():
    llm = ChatOpenAI(model=MODEL_OPENAI, openai_api_key=OPENAI_API_KEY)
    retriever = vectorstore.as_retriever()
    contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng, 
    có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập, 
    có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, 
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

    qa_system_prompt = """Bạn là trợ lý được phát triển bởi team AI thaco industries cho các nhiệm vụ trả lời câu hỏi. 
    Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. 
    Nếu bạn không tìm được câu trả lời từ đoạn ngữ cảnh, hãy sử dụng dữ liệu bạn đã được huấn luyện sẵn để trả lời. 
    Những câu hỏi xã giao ví dụ xin chào, tạm biệt thì không cần phải truy xuất ngữ cảnh. 
    Nếu vẫn không thể trả lời được bạn cứ trả lời là xin lỗi vì bạn bị thiếu dữ liệu. 
    Những câu trả lời cần truy cập vào internet để lấy thì bạn vẫn phải truy cập không được trả lời xin lỗi. 
    Sử dụng tối đa ba câu và giữ câu trả lời ngắn gọn. 
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
        "search_context_info",
        "Bạn là trợ lý được phát triển bởi team AI Thaco Industries tìm kiếm và trả về những thông tin về đoạn "
        "ngữ cảnh cung cấp.",
    )

    tools = [rag_tools] + list_tools_use
    agent = create_openai_tools_agent(llm, tools, qa_prompt)
    return AgentExecutor(agent=agent, tools=tools), history_aware_retriever


# Khởi tạo lần đầu
init_vectorstore()
agent_executor, history_aware_retriever = create_retriever_and_agent()

################################ API ####################################
app = FastAPI(title="Chatbot Back End 1 Bot",
              description="Back End deploy chatbot for 1 bot")


class InputData(BaseModel):
    text: str
    items: List[str]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process", tags=["Process Question Management"])
async def process_data(input_data: InputData):
    nhap = input_data.text

    list_qna = input_data.items
    try:
        retrieved_context = history_aware_retriever.invoke({"input": nhap, "chat_history": list_qna})
        input_data = {
            "input": nhap,
            "context": retrieved_context,
            "chat_history": list_qna
        }
        rep = agent_executor.invoke(input_data)
    except Exception as bug:
        print("bug: ", bug)
        return {"reply": "Xin lỗi nhưng tôi không có thông tin để trả lời câu hỏi của bạn."}

    bot_response = rep["output"]
    return {"reply": bot_response}


# Endpoint để reload lại vectorstore
@app.get("/reload", tags=["Vectorstore Management"])
async def reload_vectorstore():
    try:
        init_vectorstore()
        global agent_executor, history_aware_retriever
        agent_executor, history_aware_retriever = create_retriever_and_agent()
        return {"message": "Vectorstore và agent executor đã được reload thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi reload: {str(e)}")

