import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    def __init__(
        self,
        directory: str,
        use_ocr: bool = True,
        s3_bucket: str = None,
        ocr_dpi: int = 300,
        ocr_workers: int = 1,
    ):
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
        self.has_tesseract_ocr = False
        self.s3_bucket = s3_bucket
        self.ocr_dpi = max(100, int(ocr_dpi))
        self.ocr_workers = max(1, int(ocr_workers))
        
        if use_ocr:
            try:            
                from unstructured.partition.pdf import partition_pdf
                self.has_unstructured = True
            except ImportError:
                self.has_unstructured = False

            try:
                import pytesseract
                from pdf2image import convert_from_path
                self.has_tesseract_ocr = shutil.which("tesseract") is not None
            except ImportError:
                self.has_tesseract_ocr = False

            if use_ocr and not self.has_tesseract_ocr:
                print("⚠️ Binário 'tesseract' não encontrado no PATH.")

            if not self.has_unstructured and not self.has_tesseract_ocr:
                print("⚠️ OCR indisponível. Instale dependências de OCR (unstructured ou pdf2image+pytesseract).")

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
        skipped_pdfs: List[str] = []
        if self.s3_bucket:
            pdfs = self._list_s3_pdfs()
        else:
            if not os.path.isdir(self.directory):
                raise FileNotFoundError(
                    f"Diretório de PDFs não encontrado: '{self.directory}'. "
                    "Use --data-dir com o caminho correto."
                )
            pdfs = []
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.endswith(".pdf"):
                        pdfs.append(os.path.join(root, file))
            if not pdfs:
                raise FileNotFoundError(
                    f"Nenhum PDF encontrado em '{self.directory}'. "
                    "Verifique o caminho informado em --data-dir."
                )
        
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
                    if total_text < 100 and self.use_ocr:
                        print(f"   🖼️  Detectado PDF de imagem, aplicando OCR...")
                        try:
                            ocr_docs = self._load_with_ocr(local_pdf)
                            if ocr_docs:
                                temp_docs = ocr_docs
                        except Exception as exc:
                            print(f"   ⚠️ OCR falhou ({exc}). Seguindo com extração padrão.")

                    if not temp_docs:
                        print(
                            "   ⚠️ Nenhuma página extraída deste PDF "
                            "(arquivo pode estar corrompido ou sem páginas)."
                        )
                        skipped_pdfs.append(f"s3://{self.s3_bucket}/{key}")

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
                
                if total_text < 100 and self.use_ocr:
                    # Se pouco texto, usa OCR
                    print(f"   🖼️  Detectado PDF de imagem, aplicando OCR...")
                    try:
                        ocr_docs = self._load_with_ocr(pdf)
                        if ocr_docs:
                            temp_docs = ocr_docs
                    except Exception as exc:
                        print(f"   ⚠️ OCR falhou ({exc}). Seguindo com extração padrão.")

                if not temp_docs:
                    print(
                        "   ⚠️ Nenhuma página extraída deste PDF "
                        "(arquivo pode estar corrompido ou sem páginas)."
                    )
                    skipped_pdfs.append(pdf)
                
                docs.extend(temp_docs)
                print(f"   ✓ {len(temp_docs)} páginas processadas")

        if skipped_pdfs:
            print("\n⚠️ PDFs ignorados por falha de leitura/OCR:")
            for skipped in skipped_pdfs:
                print(f"  - {skipped}")

        if not docs:
            raise RuntimeError(
                "Nenhuma página foi extraída dos PDFs informados. "
                "Verifique os arquivos em data/raw (alguns podem estar corrompidos ou sem páginas)."
            )

        print(f"\n✓ Total: {len(docs)} páginas de {len(pdfs)} PDFs")
        return docs
    
    def _load_with_ocr(self, pdf_path: str) -> List[Document]:
        """Aplica OCR usando Unstructured (preferencial) ou fallback Tesseract."""
        if self.has_unstructured:
            return self._load_with_unstructured_ocr(pdf_path)
        if self.has_tesseract_ocr:
            return self._load_with_tesseract_ocr(pdf_path)
        return []

    def _load_with_unstructured_ocr(self, pdf_path: str) -> List[Document]:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",
            languages=["por"],
        )

        pages = {}
        for element in elements:
            page_num = element.metadata.page_number or 1
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(str(element))

        docs = []
        for page_num, texts in sorted(pages.items()):
            content = "\n".join(texts)
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={"source": pdf_path, "page": page_num}
                ))
        return docs

    def _load_with_tesseract_ocr(self, pdf_path: str) -> List[Document]:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFPageCountError
        import pytesseract
        from pytesseract import TesseractNotFoundError
        import fitz
        from PIL import Image

        def _render_with_fitz() -> List[Image.Image]:
            images_local = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
                    mode = "RGB" if pix.n < 4 else "RGBA"
                    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    images_local.append(image)
            return images_local

        try:
            images = convert_from_path(pdf_path, dpi=self.ocr_dpi)
        except PDFPageCountError:
            # Fallback para PDFs com xref/trailer problemáticos no poppler.
            images = _render_with_fitz()

        # Alguns PDFs retornam lista vazia no poppler sem lançar exceção.
        if not images:
            print("   ⚠️ pdf2image retornou 0 páginas; tentando fallback com PyMuPDF...")
            images = _render_with_fitz()

        docs = []
        total_pages = len(images)
        if total_pages == 0:
            print(
                "   ⚠️ OCR não conseguiu renderizar páginas deste PDF. "
                "Pulando OCR e seguindo com extração padrão."
            )
            return []
        print(
            f"   🔎 OCR em andamento: {total_pages} páginas "
            f"(dpi={self.ocr_dpi}, workers={self.ocr_workers})..."
        )
        try:
            if self.ocr_workers == 1:
                for page_num, image in enumerate(images, start=1):
                    content = pytesseract.image_to_string(image, lang="por")
                    if content.strip():
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": pdf_path, "page": page_num}
                        ))
                    if page_num == 1 or page_num % 5 == 0 or page_num == total_pages:
                        print(f"   ⏳ OCR: página {page_num}/{total_pages}")
            else:
                page_texts = {}
                with ThreadPoolExecutor(max_workers=self.ocr_workers) as executor:
                    futures = {
                        executor.submit(pytesseract.image_to_string, image, lang="por"): page_num
                        for page_num, image in enumerate(images, start=1)
                    }
                    for done_count, future in enumerate(as_completed(futures), start=1):
                        page_num = futures[future]
                        page_texts[page_num] = future.result() or ""
                        if done_count == 1 or done_count % 5 == 0 or done_count == total_pages:
                            print(f"   ⏳ OCR: página {done_count}/{total_pages}")

                for page_num in sorted(page_texts):
                    content = page_texts[page_num]
                    if content.strip():
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": pdf_path, "page": page_num}
                        ))
        except TesseractNotFoundError as exc:
            raise RuntimeError(
                "OCR indisponível: binário 'tesseract' não instalado ou fora do PATH. "
                "Instale no sistema: sudo apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils"
            ) from exc
        return docs
