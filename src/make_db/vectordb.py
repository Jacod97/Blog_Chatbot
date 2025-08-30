# vectordb.py (DB 생성은 최초 1회만 실행)
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os, json
from dotenv import load_dotenv

load_dotenv("../../.env")

json_path = "../../config.json"
with open(json_path, "r", encoding="utf-8") as f:
    config = json.load(f)

db_path = os.path.join(config['path'], "vectorDB/chroma_eng")
diary_path = os.path.join(config['path'], "meta/Diary(eng).txt")

loader = TextLoader(diary_path, encoding="utf-8")
docs = loader.load()

text = docs[0].page_content
paragraphs = text.split("\n\n")

chunk_docs = [
    Document(page_content=p.strip(), metadata={"source": diary_path, "chunk": i})
    for i, p in enumerate(paragraphs) if p.strip()
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(chunk_docs, embeddings, persist_directory=db_path)
db.persist()

print(f"✅ DB 생성 완료: {db_path}, 총 {len(chunk_docs)}개 문단 저장됨")
