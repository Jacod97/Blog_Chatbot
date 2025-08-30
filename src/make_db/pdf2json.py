import re
import json
import os
from pathlib import Path
from pypdf import PdfReader

# 입력 / 출력 경로
path = r"C:\Project\gitpage\Blog_Chatbot\data\meta"
input_path = os.path.join(path, "korea_qd.pdf")
output_path = os.path.join(path, "korea_qd.json")

def read_pdf_text(path: str) -> str:
    """PDF 전체 텍스트 추출"""
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def parse_qa(text: str):
    """
    PDF 텍스트에서 Q/A 쌍 추출
    - "Q." 로 시작하는 질문을 찾아 그 뒤 텍스트를 답변으로 묶음
    """
    qa_list = []

    # 'Q.' 로 시작하는 구간들을 인덱스로 잡음
    pattern = re.compile(r"Q\.\s*(.+)")
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        question = m.group(1).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        answer_raw = text[start:end].strip()

        # A. 제거
        answer = re.sub(r"^A\.\s*", "", answer_raw, flags=re.M).strip()

        if question and answer:
            qa_list.append({"question": question, "answer": answer})

    return qa_list

def main():
    text = read_pdf_text(input_path)
    qa_data = parse_qa(text)

    # JSON 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 변환 완료: {len(qa_data)}개 Q/A → {output_path}")

if __name__ == "main":
    main()
