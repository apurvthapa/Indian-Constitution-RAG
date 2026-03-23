from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_final"

_embeddings = None
_vector_store = None
_llm = None
_retriever = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISS.load_local(
            str(FAISS_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vector_store


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8},
        )
    return _retriever
