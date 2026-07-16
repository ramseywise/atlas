"""Canonical paths for eval HTML, JSON stats, and comparison figures.

JSON stats live under ``data/datasets/{source}/stats/`` (local data dir).
Published HTML/SVG live under ``evals/reports/output/{source}/``:

| Artifact | Path |
|----------|------|
| Stats HTML | ``evals/reports/output/{source}/{source}_stats.html`` |
| Suite HTML | ``evals/reports/output/{source}/{source}_suite.html`` |
| VA A/B compare + calibration | ``evals/reports/output/va/cohort_compare.html``, ``calibration.html`` |
| Figures | ``evals/reports/output/figures/{source}/*.svg`` |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPORTS_ROOT = Path("evals/reports")
OUTPUT_ROOT = REPORTS_ROOT / "output"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
DATASETS_ROOT = Path("data/datasets")

# Cross-corpus + methods HTML (VA staging eval umbrella)
VA_OUTPUT_DIR = OUTPUT_ROOT / "va"

_REPORT_HREFS: dict[str, tuple[str, str]] = {
    "calibration": ("va", "calibration.html"),
    "comparison": ("va", "cohort_compare.html"),
    "cohort_compare": ("va", "cohort_compare.html"),
    "bkh_stats": ("bkh", "bkh_stats.html"),
    "bkh_suite": ("bkh", "bkh_suite.html"),
    "va_stats": ("va", "va_stats.html"),
    "va_suite": ("va", "va_suite.html"),
}


def report_href(from_corpus: str, to: str) -> str:
    """Relative href from a report under ``output/{bkh|va}/`` to another canonical page."""
    from_key = "va" if from_corpus in ("va", "va_staging") else from_corpus
    to_dir, filename = _REPORT_HREFS[to]
    if from_key == to_dir:
        return filename
    return f"../{to_dir}/{filename}"
_REPO_MARKERS = ("pyproject.toml", "Makefile", ".git")

REFERENCE_JSONL: dict[str, str] = {
    "bkh": "eval_sets/bkh_va_overlap.jsonl",
    "intercom": "intercom_cleaned.jsonl",
    "va": "va_staging_responses/va_staging_all_responses.jsonl",
}


def repo_root(start: Path | None = None) -> Path:
    """Galactus repo root (not process CWD)."""
    path = (start or Path(__file__)).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if any((candidate / m).exists() for m in _REPO_MARKERS):
            return candidate
    return Path(__file__).resolve().parents[2]


def dataset_root(source: str) -> Path:
    """Absolute data directory for a source (va profile → va_staging on disk)."""
    key = "va_staging" if source in ("va", "va_staging") else source
    return (repo_root() / DATASETS_ROOT / key).resolve()


def reference_jsonl(source: str) -> Path:
    """Primary reference JSONL for cross-corpus URL overlap."""
    if source == "bkh":
        from evals.reports.paths import bkh_overlap_reference_path

        return bkh_overlap_reference_path()
    rel = REFERENCE_JSONL.get(source)
    if not rel:
        raise KeyError(f"No reference JSONL configured for source={source!r}")
    path = (dataset_root(source) / rel).resolve()
    if source == "bkh" and not path.exists():
        legacy = (dataset_root("bkh") / "eval_sets/all.jsonl").resolve()
        if legacy.exists():
            return legacy
    return path


@dataclass(frozen=True)
class ReportLayout:
    """Output locations for one eval source (bkh, va, intercom, …)."""

    source: str

    @property
    def html_dir(self) -> Path:
        return OUTPUT_ROOT / self.source

    @property
    def stats_json_dir(self) -> Path:
        return dataset_root(self.source) / "stats"

    @property
    def quality_json_dir(self) -> Path:
        root = dataset_root(self.source)
        if self.source in ("va", "va_staging"):
            return root / "golden"
        return root / "quality_results"

    @property
    def figures_dir(self) -> Path:
        return FIGURES_ROOT / self.source

    def stats_html(self, _stem: str | None = None) -> Path:
        return self.html_dir / f"{self.source}_stats.html"

    def suite_html(self, _stem: str | None = None) -> Path:
        return self.html_dir / f"{self.source}_suite.html"

    @staticmethod
    def calibration_html() -> Path:
        """LLM grader calibration methods (Cohen's d, KDE, thresholds)."""
        return VA_OUTPUT_DIR / "calibration.html"

    def stats_json(self, stem: str) -> Path:
        return self.stats_json_dir / f"{stem}_stats.json"

    def quality_json(self, stem: str) -> Path:
        return self.quality_json_dir / f"{stem}_quality.json"

    def ensure_dirs(self) -> None:
        """Output dirs only — never creates ``data/datasets/bkh`` or ``va_staging``."""
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def layout_for_profile(profile: str) -> ReportLayout:
    """Map eval_quality / eval_stats profile name to report source key."""
    if profile in ("va", "va_staging"):
        return ReportLayout("va")
    if profile == "intercom":
        return ReportLayout("intercom")
    if profile == "golden":
        return ReportLayout("va")
    if profile.startswith("sa") or profile == "hc":
        return ReportLayout("sa")
    if profile == "bkh":
        return ReportLayout("bkh")
    return ReportLayout(profile if profile != "generic" else "generic")
