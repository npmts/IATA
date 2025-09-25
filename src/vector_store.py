import chromadb
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ms-marco-MiniLM-L6-v2"

class Vector_stores:
    def __init__(self, provider, embedding_model):
        self.embedding_model = embedding_model
        self.provider = provider
        # self.reranker = CrossEncoder( "cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
        # self.reranker = CrossEncoder(f"../models/ms-marco-MiniLM-L6-v2", device="cpu")
        self.reranker = CrossEncoder(str(MODEL_DIR), device="cpu")

        if self.provider == "Chroma_DB":
            self.client = chromadb.Client()
            try:
                self.vec_store = self.client.create_collection(name="documents")
            except:
                self.client.delete_collection("documents")
                self.vec_store = self.client.create_collection(name="documents")
        elif self.provider == "FAISS_DB":
            index = faiss.IndexFlatL2(len(self.embedding_model.encode("hello world")))
            self.vec_store = FAISS(embedding_function=self.embedding_model, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
        else:
            raise ValueError(f"Unknown vector store provider: {self.provider}")


    def add_embeddings(self, ids, texts, embeddings, metadatas):

        if self.provider == "Chroma_DB":
            self.vec_store.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        elif self.provider == "FAISS_DB":
            text_embeddings = list(zip(texts, embeddings))
            self.vec_store.add_embeddings(text_embeddings=text_embeddings, metadatas=metadatas, ids=ids)
        else:
            raise ValueError("Vector store is not initialized.")


    def _rerank_retrieved_docs(self, query, retrieved_docs, top_n):
        if not retrieved_docs:
            return []
        pairs = [(query, doc["content"]) for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:top_n]]
        return top_docs


    def retrieve_documents(self, query_embeddings, query, n_results):
        if self.provider == "Chroma_DB":
            results = self.vec_store.query(
                query_embeddings=query_embeddings,
                n_results=20,
                include=["documents", "metadatas"],
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            retrieved = [
                {"content": doc, "metadata": metadata or {}}
                for doc, metadata in zip(documents, metadatas)
                if doc
            ]
            reranked_results = self._rerank_retrieved_docs(query, retrieved, top_n=n_results)

            return reranked_results

        elif self.provider == "FAISS_DB":
            docs = self.vec_store.similarity_search_by_vector(query_embeddings, k=20)
            retrieved = [
                {
                    "content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
                for doc in docs
                if doc.page_content
            ]
            reranked_results = self._rerank_retrieved_docs(query, retrieved, top_n=n_results)

            return reranked_results
        
        else:
            raise ValueError("Vector store is not initialized.")
        

    def delete_embeddings(self, filter):
        if self.provider == "Chroma_DB":
            self.vec_store.delete(where=filter)
        elif self.provider == "FAISS_DB":
            print("Delete operation is not supported for FAISS_DB.")
        else:
            raise ValueError("Vector store is not initialized.")
        

    def delete_vector_store(self):
        if self.provider == "Chroma_DB":
            self.client.delete_collection("documents")
        else:
            raise ValueError("Vector store is not initialized.")
