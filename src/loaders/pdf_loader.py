import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, directory: str):
        self.directory = directory
    
    def load(self) -> List[Document]:
        """Carrega todos os PDFs do diretório recursivamente."""
        pdfs = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".pdf"):
                    pdfs.append(os.path.join(root, file))
        
        docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(pdf)
            docs.extend(loader.load())
        
        print(f"✓ {len(docs)} páginas carregadas de {len(pdfs)} PDFs")
        return docs