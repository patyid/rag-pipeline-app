#!/usr/bin/env python3


from src.loaders.pdf_loader import PDFLoader
from src.processors.chunker import DocumentChunker
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore
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
        self.embedder = OpenAIEmbedder(batch_size=self.batch_size)
        self.vector_store = FAISSVectorStore(self.db_name, self.embedder)
        if save_to_s3:
            from src.vectorstore.s3_storage import S3Storage
            self.s3_storage = S3Storage()
        else:
            self.s3_storage = None
   
    

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
        embeddings = self.embedder._generate_embeddings(chunks, texts)
        
        # 5. Adiciona ao vector store com embeddings pré-calculados
        print("\n💾 Adicionando ao vector store...")
        self.vector_store.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # 6. Persistencia (S3-only quando habilitado)
        if self.save_to_s3 and self.s3_storage:
            print("\n☁️ Salvando somente no S3...")
            self.s3_storage.upload_faiss_vectorstore(self.vector_store.langchain_store, self.db_name)
        else:
            print("\n💿 Salvando localmente...")
            self.vector_store.save(f"data/processed/{self.db_name}")
        
        print(f"\n✅ Pipeline concluído! DB: {self.db_name}")
        print(f"   Total de vetores: {self.vector_store.index.ntotal}")
        return self.vector_store
    
    def query(self, question: str, k: int = 5):
        """Consulta o vector store."""
        return self.vector_store.search(question, k=k)
