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
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        self.prompt = self._set_prompt()
        self.history_store = {}
        self.session_id = session_id
        self.chain = self._make_chain()
        self.history_chain = self._wrap_with_history()

    def _set_prompt(self):
        prompt_path = "../prompts/chatbot.prompt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        return ChatPromptTemplate.from_template(template)

    def _make_chain(self):
        def build_context(x):
            retrieved = "\n".join(
                d.page_content for d in self.retriever.get_relevant_documents(x["user_word"].content)
            )

            history_text = ""
            for msg in x["chat_history"]:
                role = "Human" if msg.type == "human" else "AI"
                history_text += f"{role}: {msg.content}\n"

            return f"[History]\n{history_text}\n[Retrieved]\n{retrieved}"

        chain = (
            {
                "context": build_context,
                "user_word": itemgetter("user_word"),
                "chat_history": itemgetter("chat_history"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain


    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def _wrap_with_history(self):
        return RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="user_word",
            history_messages_key="chat_history",
        )

    def ask(self, question: str, session_id=None):
        sid = session_id or self.session_id
        response = self.history_chain.invoke(
            {"user_word": HumanMessage(content=question)},
            config={"configurable": {"session_id": sid}},
        )
        return response
    