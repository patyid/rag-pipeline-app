# rag-pipeline-app

Ingestao de PDFs para RAG usando LangChain + OpenAI Embeddings + FAISS, com persistencia local.

## Setup

### 1) Criar e ativar ambiente virtual

```bash
python -m venv chatbot
source chatbot/bin/activate
```

### 2) Instalar dependencias

```bash
pip install -r requirements.txt
pip list
```

### 3) Configurar credenciais OpenAI

- Local: crie `.env` na raiz com `OPENAI_API_KEY=...`

## Estrutura de entrada

Coloque os PDFs em `data/raw/` (padrao local).

## Execucao

### Pipeline completo (local)

```bash
python main.py
```

### Com parametros personalizados

```bash
python main.py --db-name "meu_banco" --chunk-size 500
```

### Consulta de teste ao final

```bash
python main.py --test-query
```

### Ajuste de batch de embeddings

```bash
python main.py --batch-size 200
```

### Interface Streamlit (chat RAG)

Depois de gerar o vector store com `python main.py`, rode:

```bash
streamlit run streamlit_app.py
```

Chamada do chatbot (local):
- Abra `http://localhost:8501`
- Digite sua pergunta no campo `Pergunte sobre os documentos...`

Variáveis úteis:
- `VECTOR_DB_NAME` (default: `vector_db`)
- `CHAT_MODEL` para escolher o modelo de chat (default: `gpt-4o-mini`)

## OCR (PDF escaneado/imagem)

### Dependencias de sistema (Debian/Ubuntu/Lux)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils
```

### Ajuste de velocidade/qualidade

- Menor `--ocr-dpi` => mais rapido
- Maior `--ocr-workers` => mais paralelismo por pagina

```bash
python main.py --ocr-dpi 200 --ocr-workers 4
```

### Melhor qualidade (mais lento)

```bash
python main.py --ocr-dpi 300 --ocr-workers 2
```

## Troubleshooting OCR

- `tesseract is not installed`: instale `tesseract-ocr` e confirme no `PATH`
- `Unable to get page count`: o loader aplica fallback automatico com PyMuPDF

## Encerrar ambiente virtual

```bash
deactivate
```

streamlit run streamlit_app.py