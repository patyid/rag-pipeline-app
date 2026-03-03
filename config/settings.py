from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    s3_prefix: str = "vector-stores/"
    
    # Pipeline
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    vector_db_name: str = "health_supplements"
    data_dir: str = "data/raw"
    
    class Config:
        env_file = ".env"

settings = Settings()