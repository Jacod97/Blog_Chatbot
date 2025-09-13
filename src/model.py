from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv("../.env")


class ChatBot:
    def __init__(self, retriever, session_id="default"):
        # retriever 래핑
        self.retriever = self._make_retriever(retriever)

        # LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        # 프롬프트 로드
        self.prompt = self._set_prompt()

        # 대화 기록 저장소
        self.history_store = {}
        self.session_id = session_id

        # 체인 생성
        self.chain = self._make_chain()
        self.history_chain = self._wrap_with_history()

    # ==============================
    # Prompt 로딩
    # ==============================
    def _set_prompt(self):
        prompt_path = "../prompts/chatbot.prompt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        return ChatPromptTemplate.from_template(template)

    # ==============================
    # Retriever wrapper (alias 처리)
    # ==============================
    def _make_retriever(self, base_retriever):
        def retrieve(query: str):
            aliases = {
                "jacode": "Jung Jae-sik",
                "jaesik": "Jung Jae-sik",
                "jae-sik": "Jung Jae-sik",
                "정재식": "Jung Jae-sik",
            }
            q_lower = query.lower()
            for k, v in aliases.items():
                if k in q_lower:
                    query = query.replace(k, v)
            return base_retriever.get_relevant_documents(query)

        return retrieve

    # ==============================
    # Chain 생성
    # ==============================
    def _make_chain(self):
        def build_context(x):
            # retriever에서 문서 가져오기
            retrieved_docs = self.retriever(x["question"].content)
            retrieved = "\n".join(d.page_content for d in retrieved_docs)

            # 대화 기록 정리
            history_text = ""
            for msg in x["chat_history"]:
                role = "Human" if msg.type == "human" else "AI"
                history_text += f"{role}: {msg.content}\n"

            return f"[History]\n{history_text}\n[Retrieved]\n{retrieved}"

        chain = (
            {
                "context": build_context,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    # ==============================
    # 세션별 대화 기록
    # ==============================
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def _wrap_with_history(self):
        return RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    # ==============================
    # 사용자 질문 처리
    # ==============================
    def ask(self, question: str, session_id=None):
        sid = session_id or self.session_id
        response = self.history_chain.invoke(
            {"question": HumanMessage(content=question)},
            config={"configurable": {"session_id": sid}},
        )
        return response
