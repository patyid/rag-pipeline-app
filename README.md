# rag-pipeline-app
Ingestão de PDFs e análise de documentos com tecnologia LLM usando Docling + LangChain + Ollama (nomic-embed-text + llama3.2) + FAISS. Implementado como um aplicativo Streamlit em uma instância EC2 da AWS.


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
pip install -r requirements.txt

# 2. Configurar .env
cp .env.example .env
# Edite .env com suas chaves

# 3. Colocar PDFs em data/raw/

# 4. Executar pipeline completo
python main.py

# Com parâmetros personalizados
python main.py --db-name="meu_banco" --chunk-size=500 --no-s3