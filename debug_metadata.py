from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import config

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=config.CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)

docs = vectordb.similarity_search_with_score("RTOS", k=3)
for doc, score in docs:
    print("---")
    print("Metadata:", doc.metadata)
    print("Content preview:", doc.page_content[:100])