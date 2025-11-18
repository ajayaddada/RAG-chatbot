import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreManager:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(self, docs):
        self.store.add_documents(docs)

    def get_retriever(self, k=4):
        return self.store.as_retriever(search_kwargs={"k": k})
