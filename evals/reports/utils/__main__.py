"""``python -m evals.reports`` — re-render, PDF, deck, baseline, compare.

Pipelines (eval_stats / eval_quality) produce reports by default; use this only to
re-render from JSON or run optional tooling.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

from evals.reports.stats import render_report_from_json
from evals.reports.utils._rebuild import rebuild_suite_report
from evals.reports.utils.layout import ReportLayout


def _cmd_stats(args: argparse.Namespace) -> int:
    layout = ReportLayout(args.source)
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: not found: {json_path}")
        return 1
    stem = args.stem or json_path.stem.replace("_stats", "")
    out = layout.stats_html(stem) if not args.output else Path(args.output)
    layout.html_dir.mkdir(parents=True, exist_ok=True)
    render_report_from_json(json_path, out)
    print(f"Stats HTML: {out}")
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    layout = ReportLayout(args.source)
    graded = Path(args.graded)
    if not graded.exists():
        print(f"Error: not found: {graded}")
        return 1
    stem = args.stem or graded.stem.replace("_quality", "").replace("_graded", "")
    out = layout.suite_html(stem) if not args.output else Path(args.output)
    rebuild_suite_report(
        graded_path=graded,
        output_path=out,
        stats_json_path=Path(args.stats_json) if args.stats_json else None,
        source_path=Path(args.source_jsonl) if args.source_jsonl else None,
    )
    return 0


def _cmd_rebuild(args: argparse.Namespace) -> int:
    if not args.graded and not args.source and not args.stats_json:
        print("Error: need at least one of --graded, --source, --stats-json")
        return 1
    rebuild_suite_report(
        graded_path=Path(args.graded) if args.graded else None,
        source_path=Path(args.source) if args.source else None,
        stats_json_path=Path(args.stats_json) if args.stats_json else None,
        output_path=Path(args.output) if args.output else None,
        pdf=args.pdf,
    )
    return 0


def _cmd_pdf(args: argparse.Namespace) -> int:
    from evals.reports.utils._pdf import html_to_pdf

    for path_str in args.files:
        p = Path(path_str)
        if not p.exists():
            print(f"Skip (not found): {p}")
            continue
        print(f"PDF: {html_to_pdf(p)}")
    return 0


def _cmd_deck(args: argparse.Namespace) -> int:
    template = Path("evals/reports/demo/template.html")
    fig_dir = Path("evals/reports/figures/shared")
    out_path = Path("evals/reports/demo/eval_framework_tabs.html")
    backup_dir = Path("evals/reports/figures/backups")

    if args.list_markers:
        if not template.exists():
            print(f"Template not found: {template}")
            return 1
        markers = re.findall(r"<!-- FIGURE:(\w+) -->", template.read_text(encoding="utf-8"))
        figs = {p.stem for p in fig_dir.glob("*.svg")}
        for name in markers:
            print(f"  {name}: {'ok' if name in figs else 'missing'}")
        return 0

    if not template.exists():
        print(f"Template not found: {template}")
        return 1

    figs: dict[str, str] = {}
    for svg_path in sorted(fig_dir.glob("*.svg")):
        raw = svg_path.read_text(encoding="utf-8")
        raw = re.sub(r"<\?xml[^?]*\?>", "", raw).strip()
        raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw).strip()
        raw = re.sub(r'(<svg[^>]*)\s+width="[^"]*"', r"\1", raw, count=1)
        raw = re.sub(r'(<svg[^>]*)\s+height="[^"]*"', r"\1", raw, count=1)
        figs[svg_path.stem] = raw

    html = template.read_text(encoding="utf-8")
    for name in re.findall(r"<!-- FIGURE:(\w+) -->", html):
        if name in figs:
            html = html.replace(
                f"<!-- FIGURE:{name} -->", f'<div class="fig-embed">{figs[name]}</div>', 1,
            )

    if args.dry_run:
        print(f"Dry run — would write {len(html):,} bytes to {out_path}")
        return 0

    if out_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(out_path, backup_dir / f"eval_framework_tabs_{ts}.html")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")
    return 0


def _cmd_baseline(_args: argparse.Namespace) -> int:
    from evals.metrics.enrichment.baseline import generate_baseline

    baseline = generate_baseline()
    print("Baseline written:", [k for k in baseline if not k.startswith("_")])
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from evals.reports.utils.compare import DEFAULT_OUT, generate_report

    generate_report(Path(args.output) if args.output else DEFAULT_OUT)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from evals.reports.validate import main as validate_main

    vargv: list[str] = []
    if args.figures:
        vargv.append("--figures")
    if args.compute_bkh:
        vargv.append("--compute-bkh")
    return validate_main(vargv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Eval reports tooling (re-render & extras)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Render stats EDA HTML from JSON")
    p_stats.add_argument("--json", required=True)
    p_stats.add_argument("--source", required=True)
    p_stats.add_argument("--stem", default=None)
    p_stats.add_argument("--output", default=None)
    p_stats.set_defaults(func=_cmd_stats)

    p_suite = sub.add_parser("suite", help="Render suite HTML from quality JSON")
    p_suite.add_argument("--graded", required=True)
    p_suite.add_argument("--source", required=True)
    p_suite.add_argument("--stem", default=None)
    p_suite.add_argument("--output", default=None)
    p_suite.add_argument("--stats-json", default=None)
    p_suite.add_argument("--source-jsonl", default=None)
    p_suite.set_defaults(func=_cmd_suite)

    p_rebuild = sub.add_parser("rebuild", help="Rebuild suite from graded/stats JSON")
    p_rebuild.add_argument("--graded", default=None)
    p_rebuild.add_argument("--source", default=None)
    p_rebuild.add_argument("--stats-json", default=None)
    p_rebuild.add_argument("--output", default=None)
    p_rebuild.add_argument("--pdf", action="store_true")
    p_rebuild.set_defaults(func=_cmd_rebuild)

    p_pdf = sub.add_parser("pdf", help="Convert HTML report(s) to PDF")
    p_pdf.add_argument("files", nargs="+")
    p_pdf.set_defaults(func=_cmd_pdf)

    p_deck = sub.add_parser("deck", help="Assemble eval_framework_tabs.html from SVGs")
    p_deck.add_argument("--dry-run", action="store_true")
    p_deck.add_argument("--list-markers", action="store_true")
    p_deck.set_defaults(func=_cmd_deck)

    sub.add_parser("baseline", help="Update BKH regression baseline JSON").set_defaults(
        func=_cmd_baseline
    )

    p_cmp = sub.add_parser("compare", help="SA multi-agent comparison HTML")
    p_cmp.add_argument("--output", default=None)
    p_cmp.set_defaults(func=_cmd_compare)

    p_val = sub.add_parser(
        "validate",
        help="Regenerate BKH + VA golden HTML from existing JSON (no LLM)",
    )
    p_val.add_argument("--figures", action="store_true")
    p_val.add_argument("--compute-bkh", action="store_true")
    p_val.set_defaults(func=_cmd_validate)

    parsed = parser.parse_args(argv)
    return parsed.func(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
