from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install pandas to run datacheck.py "
        "(e.g. pip install pandas)"
    ) from exc

CSV_FILE = "context_kmer_results.csv"
REPORT_FILE = "context_kmer_report.txt"
REPORT_HTML = "context_kmer_report.html"

# Expected numeric columns (besides run_num, which is treated separately)
NUMERIC_COLS = [
    "kmer",
    "context_window",
    "epochs",
    "run",
    "Accuracy",
    "F1 Score",
    "Matthews Correlation",
    "Precision",
    "Recall",
    "Runtime",
    "Samples/sec",
    "Steps/sec",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_section(title: str, body: str) -> str:
    """Format a titled section for the text report."""
    lines = [
        "=" * 80,
        f"{title}",
        "-" * 80,
        body,
        "=" * 80,
        "",
    ]
    return "\n".join(lines)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def check_duplicates(df: pd.DataFrame) -> None:
    print_header("Duplicate Rows (ignoring run_num)")
    dedup_cols = [c for c in df.columns if c != "run_num"]
    dup_mask = df.duplicated(subset=dedup_cols, keep=False)
    dup_count = dup_mask.sum()
    if dup_count == 0:
        print("No duplicate rows found.")
        return
    print(f"Found {dup_count} duplicate rows (ignoring run_num). Showing first 5:")
    print(df.loc[dup_mask].head(5).to_string(index=False))


def check_missing(df: pd.DataFrame) -> None:
    print_header("Missing Values by Column")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("No missing values detected.")
    else:
        print(missing[missing > 0])


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.copy()
    for col in NUMERIC_COLS:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    return df_num


def check_numeric(df_num: pd.DataFrame) -> None:
    print_header("Non-numeric Values (after coercion)")
    non_numeric = df_num[NUMERIC_COLS].isna().sum()
    # Ignore rows that were legitimately NaN in the source (should not happen here)
    if non_numeric.sum() == 0:
        print("All expected numeric columns parsed successfully.")
    else:
        print("Columns with parsing issues (NaN after coercion):")
        print(non_numeric[non_numeric > 0])


def summary_stats(df_num: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    print_header("Summary by (kmer, context_window, epochs)")
    grouped = (
        df_num.groupby(["kmer", "context_window", "epochs"])
        [["Accuracy", "F1 Score", "Matthews Correlation", "Precision", "Recall", "Runtime"]]
        .agg(["mean", "std", "min", "max", "count"])
    )
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(grouped)
        return grouped, grouped.to_string()


def correlations(df_num: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    print_header("Correlations (metrics with runtime / throughput)")
    cols_for_corr = [
        "Accuracy",
        "F1 Score",
        "Matthews Correlation",
        "Precision",
        "Recall",
        "Runtime",
        "Samples/sec",
        "Steps/sec",
    ]
    corr = df_num[cols_for_corr].corr()
    corr_rounded = corr.round(3)
    print(corr_rounded)
    with pd.option_context("display.width", 120):
        return corr_rounded, corr_rounded.to_string()


def main() -> None:
    df = load_csv(Path(CSV_FILE))
    df = strip_whitespace(df)

    check_duplicates(df)
    check_missing(df)

    df_num = coerce_numeric(df)
    check_numeric(df_num)
    summary_df, summary_text = summary_stats(df_num)
    corr_df, corr_text = correlations(df_num)

    # Build report body
    report_header = "\n".join(
        [
            "=" * 80,
            "CONTEXT/KMER RESULTS REPORT".center(80),
            "=" * 80,
            f"Source file: {CSV_FILE}",
            f"Total rows: {len(df)}",
            "",
        ]
    )

    report_sections = [
        report_header,
        format_section("Summary by (kmer, context_window, epochs)", summary_text),
        format_section("Correlations (metrics with runtime / throughput)", corr_text),
    ]
    report_path = Path(REPORT_FILE)
    with report_path.open("w") as fh:
        fh.write("\n".join(report_sections))

    # Also write to HTML with separate tables for summary and correlations.
    html_parts = [
        "<html>",
        "<head>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }",
        "h1, h2 { color: #0f1720; }",
        ".meta { margin-bottom: 16px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 13px; }",
        "th, td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }",
        "th { background: #f6f8fa; color: #0f1720; }",
        ".section { margin-bottom: 32px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Context/Kmer Results Report</h1>",
        f"<div class='meta'>Source file: <strong>{CSV_FILE}</strong><br>Total rows: {len(df)}</div>",
        "<div class='section'>",
        "<h2>Summary by (kmer, context_window, epochs)</h2>",
        summary_df.to_html(border=0, classes="table summary"),
        "</div>",
        "<div class='section'>",
        "<h2>Correlations (metrics with runtime / throughput)</h2>",
        corr_df.to_html(border=0, classes="table correlations"),
        "</div>",
        "</body>",
        "</html>",
    ]
    Path(REPORT_HTML).write_text("\n".join(html_parts))

    print(f"Report written to {report_path} and {REPORT_HTML}")


if __name__ == "__main__":
    main()
