import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# Carrega as variáveis do arquivo .env
load_dotenv()

# Chat Model
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# Export variables
__all__ = ['llm', 'embeddings', 'vector_store']