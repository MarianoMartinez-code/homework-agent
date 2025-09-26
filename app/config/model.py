# from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv()


model = init_chat_model("command-r-plus-08-2024", model_provider="cohere")


# model = ChatMistralAI(model="mistral-large-latest")