from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import config

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=config.CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)

print("Total documents in Chroma collection:", vectordb._collection.count())