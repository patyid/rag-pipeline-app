import os
import faiss
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from src.embeddings.openai_embedder import OpenAIEmbedder

class FAISSVectorStore:
    def __init__(self, db_name: str, embedder: OpenAIEmbedder):
        self.db_name = db_name
        self.embeddings = embedder.get_embeddings()
        self.vector_store: Optional[FAISS] = None
        self._initialize_index()
    
    @property
    def index(self):
        """Expõe o índice FAISS subjacente."""
        return self.vector_store.index
    
    @property
    def ntotal(self):
        """Retorna o número total de vetores no índice."""
        return self.vector_store.index.ntotal
    
    def _initialize_index(self):
        """Inicializa índice FAISS vazio."""
        sample_vector = self.embeddings.embed_query("test")
        dimension = len(sample_vector)
        
        index = faiss.IndexFlatL2(dimension)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        print(f"✓ Índice FAISS criado - Dimensão: {dimension}")
    
    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        """Adiciona textos com embeddings pré-calculados ao índice."""
        from langchain_core.documents import Document
        
        ids = self.vector_store.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=metadatas
        )
        print(f"✓ {len(ids)} vetores adicionados ao índice FAISS")
        return ids
    
    def save(self, path: Optional[str] = None):
        """Salva localmente."""
        save_path = path or self.db_name
        os.makedirs(save_path, exist_ok=True)
        self.vector_store.save_local(save_path)
        print(f"✓ Vector store salvo em: {save_path}")
        return save_path
    
    def load(self, path: Optional[str] = None):
        """Carrega de diretório local."""
        load_path = path or self.db_name
        self.vector_store = FAISS.load_local(
            load_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Vector store carregado de: {load_path}")
        return self.vector_store
    
    def search(self, query: str, k: int = 5):
        """Busca documentos similares."""
        return self.vector_store.similarity_search(query, k=k)