import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, directory: str, use_ocr: bool = True):
        """
        Inicializa o carregador de PDFs.

        Parameters:             
        directory (str): Diretório onde estão os PDFs.
        use_ocr (bool, optional): Se True, utiliza ocr para extrair texto de PDFs de imagem. Defaults to True.
        """
        self.directory = directory
        self.use_ocr = use_ocr
        self.has_unstructured = False
        
        if use_ocr:
            try:            
                from unstructured.partition.pdf import partition_pdf
                self.has_unstructured = True
            except ImportError:
                print("⚠️ unstructured não instalado. Instale: pip install unstructured[all-docs]")
                self.has_unstructured = False
    
    def load(self) -> List[Document]:
        """Carrega todos os PDFs do diretório recursivamente."""
        pdfs = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".pdf"):
                    pdfs.append(os.path.join(root, file))
        
        docs = []
        for pdf in pdfs:
            print(f"📄 Processando: {os.path.basename(pdf)}")
            
            # Tenta PyMuPDF primeiro
            loader = PyMuPDFLoader(pdf)
            temp_docs = loader.load()
            
            # Verifica se extraiu texto significativo
            total_text = sum(len(d.page_content.strip()) for d in temp_docs)
            
            if total_text < 100 and self.use_ocr and self.has_unstructured:
                # Se pouco texto, usa OCR
                print(f"   🖼️  Detectado PDF de imagem, aplicando OCR...")
                temp_docs = self._load_with_ocr(pdf)
            
            docs.extend(temp_docs)
            print(f"   ✓ {len(temp_docs)} páginas processadas")
        
        print(f"\n✓ Total: {len(docs)} páginas de {len(pdfs)} PDFs")
        return docs
    
    def _load_with_ocr(self, pdf_path: str) -> List[Document]:
        """Usa Unstructured com OCR para PDFs de imagem."""
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",  # Usa OCR
            languages=["por"],   # Português - mude se necessário
        )
        
        # Agrupa por página
        pages = {}
        for element in elements:
            page_num = element.metadata.page_number or 1
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(str(element))
        
        # Cria documentos do LangChain
        from langchain_core.documents import Document
        docs = []
        for page_num, texts in sorted(pages.items()):
            content = "\n".join(texts)
            if content.strip():  # Só adiciona se tiver conteúdo
                docs.append(Document(
                    page_content=content,
                    metadata={"source": pdf_path, "page": page_num}
                ))
        
        return docs
