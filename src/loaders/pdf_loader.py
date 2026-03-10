import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, directory: str, use_ocr: bool = True, s3_bucket: str = None):
        """
        Inicializa o carregador de PDFs.

        Parameters:             
        directory (str): Diretório local (modo local) ou prefixo S3 (modo S3).
        use_ocr (bool, optional): Se True, utiliza ocr para extrair texto de PDFs de imagem. Defaults to True.
        s3_bucket (str, optional): Se informado, lê PDFs de `s3://{s3_bucket}/{directory}`.
        """
        self.directory = directory
        self.use_ocr = use_ocr
        self.has_unstructured = False
        self.s3_bucket = s3_bucket
        
        if use_ocr:
            try:            
                from unstructured.partition.pdf import partition_pdf
                self.has_unstructured = True
            except ImportError:
                print("⚠️ unstructured não instalado. Instale: pip install unstructured[all-docs]")
                self.has_unstructured = False

    def _list_s3_pdfs(self) -> List[str]:
        import boto3
        from config.settings import settings

        prefix = self.directory or ""
        s3_client = boto3.client("s3", region_name=settings.aws_region)
        paginator = s3_client.get_paginator("list_objects_v2")

        pdf_keys: List[str] = []
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not key or key.endswith("/"):
                    continue
                if key.lower().endswith(".pdf"):
                    pdf_keys.append(key)

        return pdf_keys
    
    def load(self) -> List[Document]:
        """Carrega todos os PDFs do diretório recursivamente."""
        if self.s3_bucket:
            pdfs = self._list_s3_pdfs()
        else:
            pdfs = []
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.endswith(".pdf"):
                        pdfs.append(os.path.join(root, file))
        
        docs = []
        if self.s3_bucket:
            import boto3
            from config.settings import settings

            s3_client = boto3.client("s3", region_name=settings.aws_region)
            with tempfile.TemporaryDirectory(prefix="pdf-loader-") as tmpdir:
                for i, key in enumerate(pdfs, 1):
                    filename = os.path.basename(key) or f"document_{i}.pdf"
                    local_pdf = os.path.join(tmpdir, f"{i:05d}_{filename}")

                    print(f"📄 Baixando: s3://{self.s3_bucket}/{key}")
                    s3_client.download_file(self.s3_bucket, key, local_pdf)

                    print(f"📄 Processando: {os.path.basename(local_pdf)}")
                    loader = PyMuPDFLoader(local_pdf)
                    temp_docs = loader.load()

                    total_text = sum(len(d.page_content.strip()) for d in temp_docs)
                    if total_text < 100 and self.use_ocr and self.has_unstructured:
                        print(f"   🖼️  Detectado PDF de imagem, aplicando OCR...")
                        temp_docs = self._load_with_ocr(local_pdf)

                    # Preserva a origem S3 no metadata para rastreabilidade
                    for d in temp_docs:
                        d.metadata = dict(d.metadata or {})
                        d.metadata.setdefault("source", f"s3://{self.s3_bucket}/{key}")

                    docs.extend(temp_docs)
                    print(f"   ✓ {len(temp_docs)} páginas processadas")
        else:
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
