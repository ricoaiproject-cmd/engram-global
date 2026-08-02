"""Smoke-test CLI (owner: Agent B).

Drives the engine directly through argparse subcommands. Output is
human-readable by default (recall shows a score breakdown table),
or raw JSON with --json.

    engram remember "text" --type knowledge --importance 7 --tags a,b
    engram recall "query" [--deep] [--limit 5] [--type knowledge] [--no-record]
    engram reinforce ID [ID...] [--strength 2.0]
    engram correct ID --content "corrected text" --reason "reason"
    engram forget ID / engram link SRC DST
    engram stats / engram reindex
    engram consolidation-candidates
    engram mark-consolidated NEW_ID --episodes ID1,ID2,...  record that consolidation is done
    engram skill-candidates              show episode clusters that are skill candidates
    engram surface "utterance" [--room X]     manual check of spontaneous recall (writes nothing)
    engram hook session-end|user-prompt  entry point for agent hooks (JSON on stdin)
    engram export-onnx [--force]         export the embedding model to ONNX (faster startup)

Use the --fake-embedder flag to use FakeEmbedder (for testing without a model installed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_engine(fake_embedder: bool = False):
    """Build the engine. Chooses the embedder based on the --fake-embedder flag."""
    from .config import get_settings
    from .engine import build_engine
    from .embedder import FakeEmbedder

    settings = get_settings()
    embedder = FakeEmbedder() if fake_embedder else None
    return build_engine(settings, embedder=embedder)


def _print_recall(result: dict, as_json: bool = False) -> None:
    """Print recall results, formatted."""
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    mode = result.get("mode", "?")
    auto_deepened = result.get("auto_deepened", False)
    hits = result.get("hits", [])

    header = f"[recall: mode={mode}"
    if auto_deepened:
        header += ", auto_deepened=True"
    header += f", {len(hits)} hits]"
    print(header)
    print()

    if not hits:
        print("  (no results)")
        return

    # Compute column widths
    col_id = 26
    col_score = 7
    col_rel = 7
    col_act = 7
    col_imp = 5
    col_via = 12
    col_tier = 12

    # Header row
    print(
        f"{'ID':<{col_id}} "
        f"{'score':>{col_score}} "
        f"{'relevance':>{col_rel}} "
        f"{'activation':>{col_act}} "
        f"{'importance':>{col_imp}} "
        f"{'via':<{col_via}} "
        f"{'tier':<{col_tier}}"
    )
    print("-" * (col_id + col_score + col_rel + col_act + col_imp + col_via + col_tier + 6))

    for hit in hits:
        print(
            f"{hit['id']:<{col_id}} "
            f"{hit['score']:>{col_score}.4f} "
            f"{hit['relevance']:>{col_rel}.4f} "
            f"{hit['activation']:>{col_act}.4f} "
            f"{hit['importance']:>{col_imp}.2f} "
            f"{hit['via']:<{col_via}} "
            f"{hit['tier']:<{col_tier}}"
        )
        # Content preview (first 80 characters)
        content_preview = hit.get("content", "").replace("\n", " ")[:80]
        if content_preview:
            print(f"  > {content_preview}")
        # If a note is present
        if hit.get("note"):
            print(f"  [note] {hit['note']}")
        print()


def _print_result(result: dict, as_json: bool = False) -> None:
    """Generic dict output."""
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for k, v in result.items():
            print(f"  {k}: {v}")


def main() -> None:
    """CLI entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="engram",
        description="engram memory engine CLI",
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use FakeEmbedder (for testing without a model installed)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="JSON output",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- remember ---
    p_remember = subparsers.add_parser("remember", help="Save a memory")
    p_remember.add_argument("content", help="Memory content")
    p_remember.add_argument(
        "--type",
        default="knowledge",
        choices=["knowledge", "preference", "project", "episode"],
        help="Memory type (default: knowledge)",
    )
    p_remember.add_argument(
        "--importance",
        type=int,
        default=5,
        help="Importance 1-10 (default: 5)",
    )
    p_remember.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags",
    )
    p_remember.add_argument(
        "--source",
        default="cli",
        help="Origin of the memory",
    )
    p_remember.add_argument(
        "--related",
        dest="related_ids",
        default="",
        help="Comma-separated IDs of related memories",
    )
    p_remember.add_argument(
        "--room",
        default=None,
        help="Memory room (default: auto-detected from the current directory)",
    )

    # --- recall ---
    p_recall = subparsers.add_parser("recall", help="Search memories")
    p_recall.add_argument("query", help="Search query")
    p_recall.add_argument(
        "--deep",
        action="store_true",
        help="Search in deep mode (also includes cold/superseded/episode)",
    )
    p_recall.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    p_recall.add_argument(
        "--type",
        default=None,
        help="Filter by type",
    )
    p_recall.add_argument(
        "--no-record",
        action="store_true",
        help="Do not record a recall_hit event",
    )
    p_recall.add_argument(
        "--room",
        default=None,
        help='Room to search (default: auto-detected from the current directory; "*" for all rooms)',
    )

    # --- reinforce ---
    p_reinforce = subparsers.add_parser("reinforce", help="Report that a memory was used")
    p_reinforce.add_argument("ids", nargs="+", help="List of memory IDs to reinforce")
    p_reinforce.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Reinforcement strength 0.1-3.0 (default: 1.0)",
    )

    # --- correct ---
    p_correct = subparsers.add_parser("correct", help="Correct a memory")
    p_correct.add_argument("id", help="ID of the memory to correct")
    p_correct.add_argument(
        "--content",
        required=True,
        help="Corrected content",
    )
    p_correct.add_argument(
        "--reason",
        required=True,
        help="Reason for the correction",
    )
    p_correct.add_argument(
        "--source",
        default="cli",
        help="Origin of the correction",
    )

    # --- forget ---
    p_forget = subparsers.add_parser("forget", help="Move a memory to the trash")
    p_forget.add_argument("id", help="ID of the memory to delete")

    # --- link ---
    p_link = subparsers.add_parser("link", help="Link two memories")
    p_link.add_argument("src", help="Source memory ID")
    p_link.add_argument("dst", help="Destination memory ID")

    # --- stats ---
    subparsers.add_parser("stats", help="Show statistics")

    # --- reindex ---
    subparsers.add_parser("reindex", help="Rebuild the DB from Markdown")

    # --- consolidation-candidates ---
    subparsers.add_parser(
        "consolidation-candidates",
        help="Show episode clusters that are consolidation candidates",
    )

    # --- mark-consolidated ---
    p_mark_consolidated = subparsers.add_parser(
        "mark-consolidated",
        help="Record that consolidation is done (link episode->new_memory_id + demote to cold)",
    )
    p_mark_consolidated.add_argument(
        "new_memory_id", help="ID of the memory that consolidation merged into (already created)"
    )
    p_mark_consolidated.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated IDs of the source episodes being consolidated",
    )

    # --- skill-candidates ---
    subparsers.add_parser(
        "skill-candidates",
        help="Show episode clusters that are skill candidates",
    )

    # --- setup ---
    p_setup = subparsers.add_parser("setup", help="Run the setup wizard")
    p_setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="Set up with defaults, without prompting",
    )
    p_setup.add_argument(
        "--memories-dir",
        default=None,
        help="Path to the memories folder (default: ~/.engram/memories)",
    )
    p_setup.add_argument(
        "--agents",
        default=None,
        metavar="AGENTS",
        help="Comma-separated list of agents to register (e.g. claude,codex). Default: all detected agents",
    )

    # --- doctor ---
    subparsers.add_parser("doctor", help="Show environment diagnostics")

    # --- export-onnx ---
    p_export = subparsers.add_parser(
        "export-onnx",
        help="Export the embedding model to ONNX to speed up startup (run once)",
    )
    p_export.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing ONNX model",
    )

    # --- surface ---
    p_surface = subparsers.add_parser(
        "surface",
        help="Manual check of spontaneous recall (writes no logs or state)",
    )
    p_surface.add_argument("query", help="Text standing in for the utterance (prompt)")
    p_surface.add_argument(
        "--room",
        default=None,
        help="Room (default: auto-detected from the current directory)",
    )

    # --- hook ---
    p_hook = subparsers.add_parser(
        "hook",
        help="Entry point invoked by agent hooks (reads JSON from stdin)",
    )
    p_hook.add_argument(
        "event",
        choices=["session-end", "user-prompt"],
        help="Hook event name",
    )

    args = parser.parse_args()

    # setup / doctor run without building the engine
    if args.command == "setup":
        from .setup import parse_agents, setup_main
        memories_dir = None
        if args.memories_dir:
            from pathlib import Path
            memories_dir = Path(args.memories_dir)
        selected_agents = None
        if args.agents is not None:
            try:
                selected_agents = parse_agents(args.agents)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        setup_main(
            memories_dir=memories_dir,
            non_interactive=args.non_interactive,
            agents=selected_agents,
        )
        return

    if args.command == "doctor":
        from .setup import doctor_main
        doctor_main()
        return

    if args.command == "export-onnx":
        from .config import get_settings
        from .onnx_export import export_onnx

        try:
            report = export_onnx(get_settings(), force=args.force)
        except (ImportError, FileExistsError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"ONNX model generated: {report['target']}")
            print(f"  Model: {report['model']} (dim={report['dim']}, "
                  f"{report['onnx_size_mb']} MB)")
            print(f"  Parity: min cosine = {report['min_cosine']:.6f} "
                  f"(matches the torch path)")
            print("  Future startups will be faster via the ONNX path (embed_backend=auto)")
        return

    # hook / surface run without building the engine (fast path)
    if args.command == "hook":
        from .hooks import run_session_end, run_user_prompt
        if args.event == "session-end":
            sys.exit(run_session_end())
        else:
            sys.exit(run_user_prompt())

    if args.command == "surface":
        from pathlib import Path

        from .config import get_settings, resolve_room
        from .surface import run_surface

        settings = get_settings()
        room = args.room or resolve_room(Path.cwd(), settings.room_paths)
        result = run_surface(
            args.query, settings=settings, room=room, mode="shadow",
            dry_run=True,
        )
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[surface: room={room}, "
                  f"threshold={settings.surface_threshold}]")
            print()
            if not result["candidates"]:
                print("  (no candidates)")
            for c in result["candidates"]:
                mark = "◎ surfaced" if c["id"] in result["surfaced"] else "   silent"
                print(f"{mark}  {c['id']}  score={c['score']:.4f} "
                      f"(rel={c['relevance']:.4f} act={c['activation']:.4f} "
                      f"imp={c['importance']}) [{c['type']}/{c['room']}]")
                preview = " ".join(c["content"].split())[:80]
                print(f"        > {preview}")
        return

    # Build the engine
    engine = _build_engine(fake_embedder=args.fake_embedder)

    if args.command == "remember":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        related_ids = (
            [r.strip() for r in args.related_ids.split(",") if r.strip()]
            if args.related_ids
            else None
        )
        from pathlib import Path

        from .config import resolve_room
        room = args.room or resolve_room(
            Path.cwd(), engine.settings.room_paths
        )
        result = engine.remember(
            content=args.content,
            type=args.type,
            importance=args.importance,
            tags=tags,
            source=args.source,
            related_ids=related_ids,
            room=room,
        )
        _print_result(result, as_json=args.as_json)

    elif args.command == "recall":
        mode = "deep" if args.deep else "fast"
        from pathlib import Path

        from .config import resolve_room
        room = args.room or resolve_room(
            Path.cwd(), engine.settings.room_paths
        )
        result = engine.recall(
            query=args.query,
            mode=mode,
            limit=args.limit,
            type=args.type,
            record_hits=not args.no_record,
            room=room,
        )
        _print_recall(result, as_json=args.as_json)

    elif args.command == "reinforce":
        result = engine.reinforce(ids=args.ids, strength=args.strength)
        _print_result(result, as_json=args.as_json)

    elif args.command == "correct":
        result = engine.correct(
            id=args.id,
            corrected_content=args.content,
            reason=args.reason,
            source=args.source,
        )
        _print_result(result, as_json=args.as_json)

    elif args.command == "forget":
        result = engine.forget(id=args.id)
        _print_result(result, as_json=args.as_json)

    elif args.command == "link":
        result = engine.link(src=args.src, dst=args.dst)
        _print_result(result, as_json=args.as_json)

    elif args.command == "stats":
        result = engine.stats()
        _print_result(result, as_json=args.as_json)

    elif args.command == "reindex":
        result = engine.reindex()
        _print_result(result, as_json=args.as_json)

    elif args.command == "consolidation-candidates":
        result = engine.consolidation_candidates()
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            clusters = result.get("clusters", [])
            print(f"Consolidation candidate clusters: {len(clusters)}")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nCluster {i} ({len(cluster['ids'])} items):")
                for id_, content in zip(cluster["ids"], cluster.get("contents", [])):
                    preview = content.replace("\n", " ")[:60] if content else "(no content)"
                    print(f"  [{id_}] {preview}")

    elif args.command == "mark-consolidated":
        episode_ids = [e.strip() for e in args.episodes.split(",") if e.strip()]
        result = engine.mark_consolidated(episode_ids, args.new_memory_id)
        _print_result(result, as_json=args.as_json)

    elif args.command == "skill-candidates":
        result = engine.skill_candidates()
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            clusters = result.get("clusters", [])
            print(f"Skill candidate clusters: {len(clusters)}")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nCluster {i} ({len(cluster['ids'])} items):")
                for id_, content in zip(cluster["ids"], cluster.get("contents", [])):
                    preview = content.replace("\n", " ")[:60] if content else "(no content)"
                    print(f"  [{id_}] {preview}")


if __name__ == "__main__":
    main()
