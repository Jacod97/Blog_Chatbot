from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os, json

from model import ChatBot  
from langchain.schema.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware   # ✅ 추가

load_dotenv("../.env")

json_path = "../config.json"
with open(json_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 벡터DB 로드
db_path = os.path.join(config['path'], "vectorDB/chroma_eng")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Retriever 준비
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ChatBot 인스턴스 생성
chatbot = ChatBot(retriever, session_id="default")

# FastAPI 앱
app = FastAPI(title="JaeSik ChatBot API")

# ✅ CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 개발 테스트에서는 전체 허용
    allow_credentials=True,
    allow_methods=["*"],       # OPTIONS 포함 모든 메서드 허용
    allow_headers=["*"],       # 모든 헤더 허용
)

# 요청/응답 모델 정의
class Question(BaseModel):
    session_id: str
    question: str

class Answer(BaseModel):
    answer: str

# 라우트
@app.post("/ask", response_model=Answer)
def ask_question(payload: Question):
    response = chatbot.ask(payload.question, session_id=payload.session_id)
    print(response)
    return Answer(answer=response)
