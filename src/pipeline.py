#!/usr/bin/env python3

from pathlib import Path

from src.loaders.pdf_loader import PDFLoader
from src.processors.chunker import DocumentChunker
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore
from config.settings import PROJECT_ROOT, settings

class IngestionPipeline:
    def __init__(
        self,
        data_dir: str = None,
        db_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        batch_size: int = 100,  # Tamanho do batch para embeddings
        ocr_dpi: int = None,
        ocr_workers: int = None,
    ):
        raw_data_dir = data_dir or settings.data_dir
        # Em modo local, caminhos relativos são resolvidos a partir da raiz do projeto.
        data_path = Path(raw_data_dir)
        self.data_dir = str(data_path if data_path.is_absolute() else (PROJECT_ROOT / data_path))
        self.db_name = db_name or settings.vector_db_name
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.batch_size = batch_size
        self.ocr_dpi = ocr_dpi if ocr_dpi is not None else settings.ocr_dpi
        self.ocr_workers = ocr_workers if ocr_workers is not None else settings.ocr_workers
        
        # Inicializa componentes
        self.loader = PDFLoader(
            self.data_dir,
            ocr_dpi=self.ocr_dpi,
            ocr_workers=self.ocr_workers,
        )
        self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
        self.embedder = OpenAIEmbedder(batch_size=self.batch_size)
        self.vector_store = FAISSVectorStore(self.db_name, self.embedder)
   
    

    def run(self):
        """Executa o pipeline completo."""
        print("🚀 Iniciando pipeline de ingestão...\n")
        
        # 1. Carrega PDFs
        print("📄 Carregando PDFs...")
        documents = self.loader.load()
        
        # 2. Cria chunks
        print("\n✂️ Criando chunks...")
        chunks = self.chunker.split(documents)
        if not chunks:
            raise RuntimeError(
                "Nenhum chunk foi gerado. Verifique se o OCR extraiu texto dos PDFs "
                "(tesseract + poppler instalados e idioma 'por' disponível)."
            )
        
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

        # 6. Persistencia local
        print("\n💿 Salvando localmente...")
        output_dir = PROJECT_ROOT / "data" / "processed" / self.db_name
        self.vector_store.save(str(output_dir))
        
        print(f"\n✅ Pipeline concluído! DB: {self.db_name}")
        print(f"   Total de vetores: {self.vector_store.index.ntotal}")
        return self.vector_store
    
    def query(self, question: str, k: int = 5):
        """Consulta o vector store."""
        return self.vector_store.search(question, k=k)
