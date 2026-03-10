#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
import argparse
from src.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão RAG")
    parser.add_argument("--db-name", default="health_supplements", help="Nome do vector DB")
    parser.add_argument("--data-dir", default="data/raw", help="Diretório dos PDFs")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Tamanho dos chunks")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Sobreposição")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamanho do batch para embeddings")
    parser.add_argument("--no-s3", action="store_true", help="Não enviar para S3")
    parser.add_argument("--test-query", action="store_true", help="Executa uma consulta de teste ao final")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        db_name=args.db_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        save_to_s3=not args.no_s3
    )
    
    # Executa ingestão
    pipeline.run()
    
    # Teste de consulta
    if args.test_query:
        print("\n🧪 Testando consulta...")
        results = pipeline.query("do que se trata este documento?", k=3)
        print(f"\nTop 3 resultados:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()
