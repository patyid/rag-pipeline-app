from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from typing import Optional, List
from tqdm import tqdm
#Classe para gerar embeddings usando a API da OpenAI, com suporte a processamento em batch e feedback visual.
class OpenAIEmbedder:
    def __init__(self, model: str = None,batch_size: int = 100):
        """Inicializa o cliente de embeddings da OpenAI.

        Args:
            model: Nome do modelo de embeddings a ser usado. Se None, usa
                o valor padrão em `settings.embedding_model`.
        """
        # Cria a instância de embeddings usando a chave e modelo configurados
        self.batch_size = batch_size
        self.embeddings = OpenAIEmbeddings(
            model=model or settings.embedding_model,
            openai_api_key=settings.openai_api_key
           
        )
    
    def get_embeddings(self):
        """Retorna a instância pronta de embeddings.

        Útil para reutilizar o cliente em outras partes do código.
        """
        return self.embeddings
    
    def test_embedding(self):
        """Testa a conexão com a API de embeddings gerando um vetor de teste.

        Faz uma chamada simples com a string "test" para garantir que a
        autenticação e o modelo estão funcionando. Retorna a dimensão do
        vetor gerado para validação rápida.
        """
        # Gera um embedding de exemplo para validar a integração
        vector = self.embeddings.embed_query("test")
        # Log simples para feedback durante execução
        print(f"✓ OpenAI Embeddings OK - Dimensão: {len(vector)}")
        return len(vector)

    
    def _generate_embeddings(self, chunks: List, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para uma lista de textos em batches com barra de progresso.

        Args:
            chunks: Lista original de 'chunks' (mantida para compatibilidade).
            texts: Lista de strings a serem transformadas em embeddings.

        Returns:
            all_embeddings: Lista de vetores (cada um é uma lista de floats).
        """
        print(f"\n🔍 Gerando embeddings para {len(texts)} chunks...")

        all_embeddings = []

        # Processa em batches para ser mais eficiente e mostrar progresso
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            # Extrai os textos deste batch
            batch_texts = texts[i:i + self.batch_size]
            # Chama o método de embedding em lote (retorna lista de vetores)
            # Usa a instância `self.embeddings` criada no __init__ (corrigido)
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            # Acrescenta os embeddings processados à lista final
            all_embeddings.extend(batch_embeddings)

        # Informações finais: quantos embeddings e dimensão do primeiro vetor
        if all_embeddings:
            print(f"✓ {len(all_embeddings)} embeddings gerados (dimensão: {len(all_embeddings[0])})")
        else:
            print("✓ Nenhum embedding gerado")

        return all_embeddings
            