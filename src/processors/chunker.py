from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """Divide documentos em chunks menores."""
        chunks = self.splitter.split_documents(documents)
        print(f"✓ {len(chunks)} chunks criados de {len(documents)} documentos")
        return chunks