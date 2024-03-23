from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

class Retriever(BaseRetriever):
    model:SentenceTransformer
    collection:Collection
    n_results:int
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        
        query_embedding = self._embed_query(query)
        results = self._retrieve_from_collection(query_embedding)
        documents = results['documents']

        return documents
    
    def _embed_query(self, query):
        return self.model. \
            encode(query). \
            tolist()
    
    def _retrieve_from_collection(self, query_embedding):

        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.n_results, 
            include=['documents',]
        )