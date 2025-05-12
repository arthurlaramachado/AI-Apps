from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated
from config import llm, vector_store
from prompt_template import Prompt
from typing import Literal

class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str
    
prompt = Prompt("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.")

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search, strict=True)
    query = structured_llm.invoke(state["question"]) # this transforms the question into a Search object
    return {"query": query}

def retrieve(state: State):
    print("Query info:")
    print(state["query"])
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(state["question"], docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}

__all__ = ['State', 'analyze_query', 'retrieve', 'generate']


