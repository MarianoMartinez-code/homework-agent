from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from app.config.model import model

# 1. Inicializar embeddings y vector store
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
vector_store = InMemoryVectorStore(embedding=embeddings)


# 2. Función para cargar PDF y almacenarlo
async def load_pdf_to_store(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    fragmentos = text_splitter.split_documents(docs)
    vector_store.add_documents(fragmentos)

# 3. Prompt para RAG
prompt = ChatPromptTemplate.from_template("""Eres un asistente experto que responde preguntas utilizando información contenida en los documentos proporcionados.
Si no encuentras la respuesta en los documentos, responde: "La información no está disponible en los documentos proporcionados."

Pregunta: {user_question}
Documentos recuperados: {docs}

Respuesta:
""")

rag_chain = prompt | model

# 4. Función para responder preguntas sobre documentos
async def ask_docs(question: str):
    docs = vector_store.similarity_search(question, k=5)
    response = rag_chain.invoke({"user_question": question, "docs": docs})
    return response.content
