from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from app.config.model import model
from app.llm.tavily import tavily
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

memory = MemorySaver()



agent = create_react_agent(
    model=model,
    tools=[tavily],
    prompt="Eres un agente que puede buscar en la web sobre el precio de las criptomonedas, no respondes nada que no sea sobre criptomonedas.",
    checkpointer=memory,
)