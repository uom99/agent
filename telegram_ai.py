"""
LangGraph

- state : 노드 간 공유되는 상태 정보
- Node: 실제 작업을 수행하는 함수
- Edge: 노드 간의 흐름을 정의
- Graph: 전체 워크플로우 구조
"""
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from typing_extensions import TypedDict

from env import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class AgentState(TypedDict):

    user_query: str
    messages: list

def crate_workflow():
    """
    START -> analyze_query -> generate_response -> END
    :return:
    """

    # 1. LLM
    model = ChatOpenAI(model="gpt-4o-mini")

    # 2. 노드 함수들 정의 (각 노드는 state를 입력받아 업데이트된 state를 반환)
    def analyze_query_node(state: AgentState) -> AgentState:
        user_query = state["user_query"]

        system_msg= SystemMessage(
            content="""
                당신은 전문 AI 어시스턴트입니다.
                사용자의 질문에 대해 정확하고 친절한 한국어 답변을 제공하세요.
            """)

        return {
            "messages": [system_msg, HumanMessage(content=user_query)],
            "user_query": user_query
        }

    def generate_response_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = model.invoke(messages)

        return {
            "messages": [response],
            "user_query": state["user_query"],
        }

    # 3. 그래프 생성 및 구성
    workflow = StateGraph(AgentState)

    # 4. 노드 추가
    workflow.add_node("analyze_query_node", analyze_query_node)
    workflow.add_node("generate_response_node", generate_response_node)

    # 5. 엣지 추가
    workflow.add_edge(START, "analyze_query_node")
    workflow.add_edge("analyze_query_node", "generate_response_node")
    workflow.add_edge("generate_response_node", END)

    # 6. 그래프 컴파일
    return workflow.compile()

class ChatBot:
    def __init__(self):
        self.workflow = crate_workflow()

    def process_message(self, user_message: str) -> str:
        inital_state : AgentState = {
            "messages": [],
            "user_query": user_message
        }

        result = self.workflow.invoke(inital_state)
        messages = result["messages"]

        return messages[0].content

async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    if update.message is None or update.message.text is None:
        return

    user_message = update.message.text

    result = ChatBot().process_message(user_message)

    await update.message.reply_text(result)

    # await update.message.reply_text(f'Hello {update.effective_user.first_name}')


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT, handler))

app.run_polling()