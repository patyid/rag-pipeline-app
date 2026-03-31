#!/usr/bin/env python3
import argparse
from src.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão RAG")
    parser.add_argument("--db-name", default="db_vector", help="Nome do vector DB")
    parser.add_argument("--data-dir", default="data/raw", help="Diretório dos PDFs")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Tamanho dos chunks")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Sobreposição")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamanho do batch para embeddings")
    parser.add_argument("--ocr-dpi", type=int, default=300, help="DPI para OCR (menor = mais rápido)")
    parser.add_argument("--ocr-workers", type=int, default=1, help="Paralelismo do OCR por página")
    parser.add_argument("--test-query", action="store_true", help="Executa uma consulta de teste ao final")
    
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        db_name=args.db_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        ocr_dpi=args.ocr_dpi,
        ocr_workers=args.ocr_workers
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
