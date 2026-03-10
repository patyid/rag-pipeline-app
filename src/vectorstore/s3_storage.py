import os
import tempfile
import boto3
from botocore.exceptions import ClientError
from config.settings import settings

class S3Storage:
    def __init__(self, bucket: str = None, prefix: str = None):
        self.bucket = bucket or settings.s3_bucket
        raw_prefix = prefix or settings.s3_prefix or ""
        # Normaliza para evitar chaves tipo "vector-storesdb_name".
        self.prefix = raw_prefix if (not raw_prefix or raw_prefix.endswith("/")) else f"{raw_prefix}/"

        client_kwargs = {"region_name": settings.aws_region}
        # Permite rodar local (via .env) ou em AWS (IAM role / default chain).
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs.update(
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )

        self.s3_client = boto3.client("s3", **client_kwargs)

    def upload_faiss_vectorstore(self, faiss_vectorstore, db_name: str):
        """
        Serializa e envia um FAISS (LangChain) direto para o S3.

        Observacao: o LangChain FAISS salva em arquivos, entao usamos um
        diretório temporario (nao persistente) apenas como staging.
        """
        with tempfile.TemporaryDirectory(prefix="faiss-vs-") as tmpdir:
            faiss_vectorstore.save_local(tmpdir)
            return self.upload_directory(tmpdir, db_name)
    
    def upload_directory(self, local_dir: str, db_name: str):
        """Faz upload de todo o diretório do vector store para S3."""
        if not self.bucket:
            print("⚠ S3_BUCKET não configurado, pulando upload")
            return None
        
        s3_path = f"{self.prefix}{db_name}"
        uploaded_files = []
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_path}/{relative_path}"
                
                try:
                    self.s3_client.upload_file(local_path, self.bucket, s3_key)
                    uploaded_files.append(s3_key)
                    print(f"  ↑ {relative_path} → s3://{self.bucket}/{s3_key}")
                except ClientError as e:
                    print(f"  ✗ Erro ao upload {file}: {e}")
        
        print(f"✓ {len(uploaded_files)} arquivos enviados para S3")
        return s3_path
    
    def download_directory(self, db_name: str, local_dir: str):
        """Baixa vector store do S3."""
        if not self.bucket:
            print("⚠ S3_BUCKET não configurado, pulando download")
            return False

        s3_path = f"{self.prefix}{db_name}"
        
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")

            found_any = False
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_path):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                found_any = True
                for obj in contents:
                    s3_key = obj["Key"]
                    relative_path = os.path.relpath(s3_key, s3_path)
                    if relative_path in (".", "") or relative_path.endswith("/"):
                        continue

                    local_path = os.path.join(local_dir, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.s3_client.download_file(self.bucket, s3_key, local_path)
                    print(f"  ↓ {relative_path}")

            if not found_any:
                print(f"⚠ Nenhum arquivo encontrado em s3://{self.bucket}/{s3_path}")
                return False
            
            print(f"✓ Vector store baixado do S3 para: {local_dir}")
            return True
            
        except ClientError as e:
            print(f"✗ Erro ao baixar do S3: {e}")
            return False
