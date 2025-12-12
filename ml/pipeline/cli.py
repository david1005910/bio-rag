#!/usr/bin/env python3
"""CLI for running the embedding pipeline"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.pipeline.embedding_pipeline import embedding_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_single(pmid: str, title: str, abstract: str | None = None) -> None:
    """Process a single paper"""
    logger.info(f"Processing paper: {pmid}")

    chunks = embedding_pipeline.process_paper(
        pmid=pmid,
        title=title,
        abstract=abstract,
    )

    logger.info(f"Generated {len(chunks)} chunks")
    for chunk in chunks:
        logger.info(f"  - {chunk.chunk_id}: {len(chunk.embedding)} dims")


def process_file(input_file: str, output_file: str | None = None) -> None:
    """Process papers from JSON file"""
    logger.info(f"Loading papers from: {input_file}")

    with open(input_file) as f:
        papers = json.load(f)

    if not isinstance(papers, list):
        papers = [papers]

    logger.info(f"Processing {len(papers)} papers")

    results = embedding_pipeline.process_papers_batch(papers)

    # Summary
    success_count = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_processed for r in results)

    logger.info(f"Processed {success_count}/{len(results)} papers")
    logger.info(f"Total chunks generated: {total_chunks}")

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(
                [
                    {
                        "doc_id": r.doc_id,
                        "chunks_processed": r.chunks_processed,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )
        logger.info(f"Results saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Single paper command
    single_parser = subparsers.add_parser("single", help="Process single paper")
    single_parser.add_argument("--pmid", required=True, help="PubMed ID")
    single_parser.add_argument("--title", required=True, help="Paper title")
    single_parser.add_argument("--abstract", help="Paper abstract")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process papers from file")
    batch_parser.add_argument("--input", required=True, help="Input JSON file")
    batch_parser.add_argument("--output", help="Output results JSON file")

    args = parser.parse_args()

    if args.command == "single":
        process_single(args.pmid, args.title, args.abstract)
    elif args.command == "batch":
        process_file(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
