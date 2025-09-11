from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os, json
from model import ChatBot  
from pipeline import Pipeline 
from langchain.schema.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware   # 추가
load_dotenv("../.env")

json_path = "../config.json"
with open(json_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 벡터DB 준비
db_path = os.path.join(config['path'], "vectorDB/chroma_eng")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ChatBot 인스턴스
chatbot = ChatBot(retriever, session_id="default")

# Pipeline 인스턴스
pipeline = Pipeline(dbname="chatbot_db", user="jacode_blog", password="141592", host="localhost", port="5432")

# FastAPI 앱
app = FastAPI(title="JaeSik ChatBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델 정의
class Register(BaseModel):
    session_id: str
    name: str
    job: str
    company: str
    country: str

class Question(BaseModel):
    session_id: str
    question: str

class Answer(BaseModel):
    answer: str


# 사용자 등록
@app.post("/register")
def register_user(user: Register):
    pipeline.register_user(user.session_id, user.name, user.job, user.company, user.country)
    return {"status": "ok"}


# 질문/응답 처리
@app.post("/ask", response_model=Answer)
def ask_question(payload: Question):
    if not pipeline.is_active(payload.session_id):
        return Answer(answer="⚠️ Your chatbot session is not active. Please register first.")

    if not pipeline.can_ask(payload.session_id):
        return Answer(answer="⚠️ You have reached the maximum of 5 questions. Session ended.")

    pipeline.save_message(payload.session_id, "user", payload.question)

    response = chatbot.ask(payload.question, session_id=payload.session_id)

    pipeline.save_message(payload.session_id, "bot", response)
    pipeline.record_question(payload.session_id)

    return Answer(answer=response)
