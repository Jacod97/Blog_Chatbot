import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv("../.env")

# ===== 사용자 지정 경로 =====
input_pdf_path = "../data/pdf/test/pdf"     # 입력 PDF 경로
output_db_path = "../data/chroma_db"           # 출력 Chroma DB 저장 경로

# ===== 1. PDF 읽기 =====
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# ===== 2. PDF 내용 → Q&A 분리 (간단 파서) =====
def parse_qa(text):
    qa_list = []
    blocks = text.split("**Q.")  # 질문 패턴 기준 분리
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            q, a = block.split("**", 1)   # 질문 / 답변 분리
            qa_list.append({
                "question": q.strip(),
                "answer": a.strip()
            })
        except ValueError:
            continue
    return qa_list

# ===== 3. 벡터 DB 구축 =====
def build_chroma(qa_list, persist_path):
    embeddings = OpenAIEmbeddings()  # OpenAI Embedding 사용
    docs = []
    for qa in qa_list:
        content = f"Q: {qa['question']}\nA: {qa['answer']}"
        docs.append(Document(page_content=content, metadata={"question": qa['question']}))

    # Chroma DB에 저장
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_path)
    vectordb.persist()
    print(f"Chroma DB 저장 완료: {persist_path}")

# ===== 메인 실행 =====
if __name__ == "__main__":
    text = extract_text_from_pdf(input_pdf_path)
    qa_list = parse_qa(text)
    build_chroma(qa_list, output_db_path)
