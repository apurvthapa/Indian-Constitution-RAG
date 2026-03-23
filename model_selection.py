from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = FAISS.load_local(
    "faiss_final",
    embeddings,
    allow_dangerous_deserialization=True
)



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

retriever = vector_store.as_retriever(
    search_type="similarity",   # or "mmr"
    search_kwargs={
        "k": 8
    }
)