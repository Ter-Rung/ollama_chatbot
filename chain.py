# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# # from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_ollama import OllamaEmbeddings


# import os

# class RAGEngine:
#     def __init__(self, data_folder="./data"):
#         self.data_folder = data_folder
#         self.llm = OllamaLLM(model="mistral", temperature=0.2)
#         self.embeddings = OllamaEmbeddings(model="mistral")
#         self.retriever = self._load_data()
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             retriever=self.retriever,
#             return_source_documents=False
#         )

#     def _load_data(self):
#         # Load và split toàn bộ file txt trong /data
#         docs = []
#         for filename in os.listdir(self.data_folder):
#             if filename.endswith(".txt"):
#                 loader = TextLoader(os.path.join(self.data_folder, filename))
#                 docs.extend(loader.load())

#         text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#         texts = text_splitter.split_documents(docs)

#         # Tạo FAISS index
#         db = FAISS.from_documents(texts, self.embeddings)
#         return db.as_retriever(search_kwargs={"k": 3})

#     def ask(self, question: str) -> str:
#         return self.qa_chain.run(question)
