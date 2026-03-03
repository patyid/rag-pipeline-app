import os
import boto3
from botocore.exceptions import ClientError
from config.settings import settings

class S3Storage:
    def __init__(self, bucket: str = None, prefix: str = None):
        self.bucket = bucket or settings.s3_bucket
        self.prefix = prefix or settings.s3_prefix
        
        if not all([settings.aws_access_key_id, settings.aws_secret_access_key]):
            raise ValueError("Credenciais AWS não configuradas no .env")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
    
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
        s3_path = f"{self.prefix}{db_name}"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket, 
                Prefix=s3_path
            )
            
            if 'Contents' not in response:
                print(f"⚠ Nenhum arquivo encontrado em s3://{self.bucket}/{s3_path}")
                return False
            
            for obj in response['Contents']:
                s3_key = obj['Key']
                relative_path = os.path.relpath(s3_key, s3_path)
                local_path = os.path.join(local_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3_client.download_file(self.bucket, s3_key, local_path)
                print(f"  ↓ {relative_path}")
            
            print(f"✓ Vector store baixado do S3 para: {local_dir}")
            return True
            
        except ClientError as e:
            print(f"✗ Erro ao baixar do S3: {e}")
            return False