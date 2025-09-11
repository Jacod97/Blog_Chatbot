# pipeline.py

import psycopg2, os
from datetime import datetime

class Pipeline:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        self.cur = self.conn.cursor()

        # 메모리에서 질문 횟수 관리
        self.session_state = {}

    # 사용자 등록
    def register_user(self, session_id, name, job, company, country):
        # users 테이블에 저장
        self.cur.execute("""
            INSERT INTO users (session_id, name, job, company, country)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET name=EXCLUDED.name, job=EXCLUDED.job, company=EXCLUDED.company, country=EXCLUDED.country
        """, (session_id, name, job, company, country))
        self.conn.commit()

        # chatbot_state 테이블에 세션 활성화 기록
        self.cur.execute("""
            INSERT INTO chatbot_state (session_id, active, activated_at)
            VALUES (%s, TRUE, NOW())
            ON CONFLICT (session_id) DO UPDATE
            SET active=TRUE, activated_at=NOW()
        """, (session_id,))
        self.conn.commit()

        # 메모리 카운터 초기화
        self.session_state[session_id] = {"question_count": 0}

    # 세션 활성 여부 확인
    def is_active(self, session_id):
        self.cur.execute("SELECT active FROM chatbot_state WHERE session_id=%s", (session_id,))
        row = self.cur.fetchone()
        return row[0] if row else False

    # 질문 가능 여부 (메모리 기준 5회 제한)
    def can_ask(self, session_id):
        if session_id not in self.session_state:
            return False
        return self.session_state[session_id]["question_count"] < 5

    # 질문 횟수 증가
    def record_question(self, session_id):
        if session_id in self.session_state:
            self.session_state[session_id]["question_count"] += 1

            # 5회 도달 시 DB에서 비활성화
            if self.session_state[session_id]["question_count"] >= 5:
                self.cur.execute("UPDATE chatbot_state SET active=FALSE WHERE session_id=%s", (session_id,))
                self.conn.commit()

    # 대화 기록 저장
    def save_message(self, session_id, sender, message):
        self.cur.execute("""
            INSERT INTO chat_history (session_id, sender, message)
            VALUES (%s, %s, %s)
        """, (session_id, sender, message))
        self.conn.commit()

    # 현재 질문 횟수 반환 (디버깅용)
    def get_question_count(self, session_id):
        if session_id in self.session_state:
            return self.session_state[session_id]["question_count"]
        return 0
