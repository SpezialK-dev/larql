"""CLI entry point for larql-knowledge."""

import argparse
from pathlib import Path


def main() -> None:
    """CLI entry point for larql-knowledge commands."""
    parser = argparse.ArgumentParser(
        prog="larql-knowledge",
        description="Knowledge pipeline for LARQL",
    )
    sub = parser.add_subparsers(dest="command")

    # Assemble
    asm = sub.add_parser("assemble", help="Assemble triple files into combined JSON")
    asm.add_argument("--triples-dir", type=Path, default=Path("data/triples"))
    asm.add_argument("--output", type=Path, default=Path("data/wikidata_triples.json"))

    # Ingest DBpedia
    dbp = sub.add_parser("ingest-dbpedia", help="Pull triples from DBpedia")
    dbp.add_argument("--output-dir", type=Path, default=Path("data/triples"))
    dbp.add_argument("--limit", type=int, default=500)

    # Ingest WordNet
    wn = sub.add_parser("ingest-wordnet", help="Extract WordNet relations")
    wn.add_argument("--output", type=Path, default=Path("data/wordnet_relations.json"))

    # Coverage report
    sub.add_parser("coverage", help="Show coverage report")

    # Probe
    prb = sub.add_parser("probe", help="Run model probe")
    prb.add_argument("--vindex", type=Path, required=True)
    prb.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    prb.add_argument("--output", type=Path, default=Path("probes"))
    prb.add_argument("--triples", type=Path, default=Path("data/wikidata_triples.json"))
    prb.add_argument("--templates", type=Path, default=Path("data/probe_templates.json"))

    args = parser.parse_args()

    if args.command == "assemble":
        from .triples import assemble, stats
        combined = assemble(args.triples_dir, args.output)
        s = stats(combined)
        print(f"Assembled {s['num_relations']} relations, {s['total_pairs']} pairs → {args.output}")

    elif args.command == "ingest-dbpedia":
        from .ingest.dbpedia import ingest_dbpedia
        print(f"Ingesting from DBpedia (limit={args.limit})...")
        ingest_dbpedia(args.output_dir, limit=args.limit)

    elif args.command == "ingest-wordnet":
        from .ingest.wordnet import ingest_wordnet
        print("Extracting WordNet relations...")
        ingest_wordnet(args.output)

    elif args.command == "coverage":
        from .analysis.coverage import coverage_report
        coverage_report()

    elif args.command == "probe":
        print("Probe requires MLX. Run: python3 scripts/probe_mlx.py")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
