"""Entry point for `python -m engram`.

Supports server mode and memory management subcommands:
  - Default: Launch MCP server (stdio or sse transport)
  - dump: Export memories as markdown files
  - ingest: Import markdown files as memories

Examples:
    python -m engram                        # local stdio server (default)
    python -m engram --transport sse        # network SSE on 0.0.0.0:8788
    python -m engram dump --project my-app --output ./memory-dump
    python -m engram ingest --project my-app --directory ./memory-ingest
"""

import argparse
import logging
import os
import sys

from .server import main

logger = logging.getLogger(__name__)


def cli() -> None:
    """Parse CLI arguments and route to appropriate handler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Create main parser with subcommands
    parser = argparse.ArgumentParser(
        description="Engram MCP memory server with memory management tools.",
        prog="engram",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command (default, no subcommand needed)
    server_parser = subparsers.add_parser(
        "server",
        help="Start the MCP memory server (default if no command specified)",
    )
    server_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport mode: stdio (local), sse (network/legacy), or streamable-http (network/recommended). Default: stdio.",
    )
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for SSE transport. Default: 0.0.0.0 (all interfaces).",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8788,
        help="Port for SSE transport. Default: 8788.",
    )
    server_parser.add_argument(
        "--api-key",
        default=os.environ.get("ENGRAM_API_KEY"),
        help="Optional API key for SSE auth. Can also set ENGRAM_API_KEY env var.",
    )

    # Dump command
    dump_parser = subparsers.add_parser(
        "dump",
        help="Export all memories as markdown files",
    )
    dump_parser.add_argument(
        "--project",
        default="default",
        help="Project namespace (e.g., my-app). Default: default.",
    )
    dump_parser.add_argument(
        "--output",
        default="./memory-dump",
        help="Output directory for markdown files. Default: ./memory-dump",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Import markdown files as memories",
    )
    ingest_parser.add_argument(
        "--project",
        default="default",
        help="Project namespace to ingest into (e.g., my-app). Default: default.",
    )
    ingest_parser.add_argument(
        "--directory",
        default="./memory-ingest",
        help="Directory with markdown files to import. Default: ./memory-ingest",
    )
    ingest_parser.add_argument(
        "--type",
        default="",
        help="Filter by memory type (decision, pattern, error, context, architecture, preference).",
    )
    ingest_parser.add_argument(
        "--importance",
        type=int,
        default=2,
        help="Importance level for ingested memories (0-4). Default: 2 (medium).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "dump":
        _handle_dump(args.project, args.output)
    elif args.command == "ingest":
        _handle_ingest(args.project, args.directory, args.type, args.importance)
    elif args.command == "server" or args.command is None:
        # Default: start server
        if args.command is None:
            # No subcommand, use default args if available
            args.transport = "stdio"
            args.host = "0.0.0.0"
            args.port = 8788
            args.api_key = os.environ.get("ENGRAM_API_KEY")
        else:
            # Extract server args from namespace
            args.transport = getattr(args, "transport", "stdio")
            args.host = getattr(args, "host", "0.0.0.0")
            args.port = getattr(args, "port", 8788)
            args.api_key = getattr(args, "api_key", os.environ.get("ENGRAM_API_KEY"))

        if args.transport == "sse":
            logger.info("Starting engram SSE server on %s:%s", args.host, args.port)
            if args.api_key:
                logger.info("API key authentication enabled.")
            elif args.host in ("0.0.0.0", "::"):
                logger.warning(
                    "Binding to all interfaces WITHOUT API key authentication. "
                    "Anyone on your network can read and write memories. "
                    "Set --api-key or ENGRAM_API_KEY to secure this endpoint. "
                    "To bind to localhost only, use --host 127.0.0.1. "
                    "If exposing beyond a trusted mesh VPN (e.g. Tailscale), deploy "
                    "behind a reverse proxy with TLS (Caddy, Nginx) to prevent "
                    "plaintext credential sniffing.",
                )
            else:
                logger.info("No API key set.")

        main(
            transport=args.transport,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )


def _handle_dump(project: str, output_path: str) -> None:
    """Handle memory dump command."""
    from .server import memory_dump

    logger.info(f"Dumping memories from project '{project}' to {output_path}")
    result = memory_dump(project=project, output_path=output_path)

    if result.get("status") == "error" or "error" in result:
        logger.error(f"Error: {result.get('error', result.get('message'))}")
        sys.exit(1)

    count = result.get("count", 0)
    logger.info(f"✅ Dumped {count} memories to {output_path}")


def _handle_ingest(
    project: str, directory: str, memory_type: str = "", importance: int = 2
) -> None:
    """Handle memory ingest command."""
    from .server import memory_ingest

    logger.info(f"Ingesting memories from {directory} into project '{project}'")
    result = memory_ingest(
        project=project,
        directory=directory,
        memory_type=memory_type,
        importance=importance,
    )

    if result.get("status") == "error" or "error" in result:
        logger.error(f"Error: {result.get('error', result.get('message'))}")
        sys.exit(1)

    count = result.get("count", 0)
    failed = result.get("failed", 0)
    snapshot_zip = result.get("snapshot_zip", "")

    logger.info(f"✅ Ingested {count} memories into project '{project}'")
    if failed > 0:
        logger.warning(f"⚠️  {failed} files failed to parse")
    if snapshot_zip:
        logger.info(f"📦 Snapshot zip created: {snapshot_zip}")


if __name__ == "__main__":
    cli()
