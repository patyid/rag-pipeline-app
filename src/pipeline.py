#!/usr/bin/env python3
from typing import Optional, List
from tqdm import tqdm
from src.loaders.pdf_loader import PDFLoader
from src.processors.chunker import DocumentChunker
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.vectorstore.s3_storage import S3Storage
from config.settings import settings

class IngestionPipeline:
    def __init__(
        self,
        data_dir: str = None,
        db_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        save_to_s3: bool = True,
        batch_size: int = 100  # Tamanho do batch para embeddings
    ):
        self.data_dir = data_dir or settings.data_dir
        self.db_name = db_name or settings.vector_db_name
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.save_to_s3 = save_to_s3
        self.batch_size = batch_size
        
        # Inicializa componentes
        self.loader = PDFLoader(self.data_dir)
        self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
        self.embedder = OpenAIEmbedder()
        self.vector_store = FAISSVectorStore(self.db_name, self.embedder)
        self.s3_storage = S3Storage() if save_to_s3 else None
   
    
    
    def _generate_embeddings(self, chunks: List, texts: List[str]) -> List[List[float]]:
        """Gera embeddings dos chunks em batches com progresso."""
        print(f"\n🔍 Gerando embeddings para {len(texts)} chunks...")
        
        all_embeddings = []
        
        # Processa em batches para ser mais eficiente e mostrar progresso
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embedder.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        print(f"✓ {len(all_embeddings)} embeddings gerados (dimensão: {len(all_embeddings[0])})")
        return all_embeddings
    
    def run(self):
        """Executa o pipeline completo."""
        print("🚀 Iniciando pipeline de ingestão...\n")
        
        # 1. Carrega PDFs
        print("📄 Carregando PDFs...")
        documents = self.loader.load()
        
        # 2. Cria chunks
        print("\n✂️ Criando chunks...")
        chunks = self.chunker.split(documents)
        
        # 3. Prepara textos dos chunks
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 4. GERA EMBEDDINGS DOS CHUNKS 
        embeddings = self._generate_embeddings(chunks, texts)
        
        # 5. Adiciona ao vector store com embeddings pré-calculados
        print("\n💾 Adicionando ao vector store...")
        self.vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # 6. Salva localmente
        print("\n💿 Salvando localmente...")
        local_path = self.vector_store.save(f"data/processed/{self.db_name}")
        
        # 7. Upload S3 (opcional)
        if self.save_to_s3 and self.s3_storage:
            print("\n☁️ Enviando para S3...")
            self.s3_storage.upload_directory(local_path, self.db_name)
        
        print(f"\n✅ Pipeline concluído! DB: {self.db_name}")
        print(f"   Total de vetores: {self.vector_store.index.ntotal}")
        return self.vector_store
    
    def query(self, question: str, k: int = 5):
        """Consulta o vector store."""
        return self.vector_store.search(question, k=k)