# rag-pipeline-app
Ingestão de PDFs para RAG usando LangChain + OpenAI Embeddings + FAISS, com persistência no S3 (pensado para rodar em uma instância EC2 da AWS).


# 1. Criar ambiente virtual (na pasta do projeto)
python -m venv venv

# 2. Ativar o ambiente
# No Linux/Mac:
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Verificar instalação
pip list

# 5. Quando terminar, desativar
deactivate

# 2. Configurar .env
cp .env.example .env
# Edite .env com suas chaves

# 3. Colocar PDFs em data/raw/

# 4. Executar pipeline completo
python main.py

# Com parâmetros personalizados
python main.py --db-name="meu_banco" --chunk-size=500 --no-s3

# Ler PDFs do S3 e salvar localmente (EC2/IAM role)
# `--data-dir` vira o prefixo dentro do bucket de PDFs.
python main.py --pdf-bucket="meu-bucket-de-pdfs" --data-dir="data/raw/"

# Ler PDFs localmente e salvar o vectorstore no S3
python main.py --vector-bucket="meu-bucket-de-vetores"

# Ler PDFs do S3 e salvar o vectorstore no S3 (buckets podem ser diferentes)
python main.py --pdf-bucket="meu-bucket-de-pdfs" --vector-bucket="meu-bucket-de-vetores" --data-dir="data/raw/"

# (Opcional) Rodar uma consulta de teste no fim (custo/latência extra)
python main.py --test-query

# Ajustar performance/custo da geração de embeddings
python main.py --batch-size=200
