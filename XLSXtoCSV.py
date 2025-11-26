
from pathlib import Path
import re
from typing import Dict, List

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install pandas (and openpyxl for .xlsx support) "
        "e.g. pip install 'pandas openpyxl'"
    ) from exc

# -------- CONFIG --------
INPUT_FILE = "Experiments Log (Tokenization 11_12).xlsx"
OUTPUT_FILE = "experiments_tokenization_parsed.csv"

# Regex patterns to capture numeric values after each metric label
# Supports simple decimals or scientific notation.
METRIC_PATTERNS = {
    "eval_accuracy": r'"eval_accuracy"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    "eval_f1": r'"eval_f1"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    "eval_loss": r'"eval_loss"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    "eval_matthews_correlation": r'"eval_matthews_correlation"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    "eval_precision": r'"eval_precision"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    "eval_recall": r'"eval_recall"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
}


def extract_first_match(text: str, pattern: str) -> str:

    if not text or pd.isna(text):
        return ""
    match = re.search(pattern, str(text))
    return match.group(1) if match else ""


def parse_run_info(run_text: str) -> tuple[str, str]:
    
    if not run_text or pd.isna(run_text):
        return "", ""

    run_str = str(run_text)
    run_number_match = re.search(r"\d+", run_str)
    mers_match = re.search(r"(\d+\s*-?\s*mer)", run_str, flags=re.IGNORECASE)

    run_number = run_number_match.group(0) if run_number_match else ""
    mers = mers_match.group(1).replace(" ", "").lower() if mers_match else ""
    return run_number, mers


def parse_epochs(text: str) -> str:
    """Extract the integer epoch count from a string like '3 epochs'."""
    if not text or pd.isna(text):
        return ""
    match = re.search(r"(\d+)\s*epoch", str(text), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def finalize_run(run_buffer: List[str]) -> str:
    
    return " ".join(s for s in run_buffer if s)


def main() -> None:
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load only the first two columns.
    df_raw = pd.read_excel(input_path, usecols=[0, 1], dtype=str, header=0)
    df_raw.columns = ["col_a", "col_b"]

    runs: List[Dict[str, str]] = []
    current_run: Dict[str, str] | None = None
    text_buffer: List[str] = []

    for _, row in df_raw.iterrows():
        a_val = row["col_a"]
        b_val = row["col_b"]
        a_str = "" if pd.isna(a_val) else str(a_val)

        # New run detected when column A contains a mer marker (e.g., "6-mer")
        mer_match = re.search(r"(\d+)\s*-?\s*mer", a_str, flags=re.IGNORECASE)
        if mer_match:
            # Close previous run
            if current_run:
                current_run["text"] = finalize_run(text_buffer)
                runs.append(current_run)
                text_buffer = []

            # Start new run with sequential run number
            run_number = str(len(runs) + 1)
            current_run = {
                "run_number": run_number,
                "mers": mer_match.group(1),
                "epochs": "",
            }

        # Capture epochs if present just below the mer row or within any row
        if current_run and not current_run.get("epochs"):
            epochs = parse_epochs(a_str)
            if epochs:
                current_run["epochs"] = epochs

        # Buffer any text we might want to parse metrics from
        if current_run:
            if not pd.isna(a_val):
                text_buffer.append(a_str)
            if not pd.isna(b_val):
                text_buffer.append(str(b_val))

    # Finalize last run
    if current_run:
        current_run["text"] = finalize_run(text_buffer)
        runs.append(current_run)

    # Extract metrics for each run
    records: List[Dict[str, str]] = []
    for run in runs:
        text_blob = run.get("text", "")
        record = {
            "run_number": run.get("run_number", ""),
            "mers": run.get("mers", ""),
            "epochs": run.get("epochs", ""),
        }
        for metric, pattern in METRIC_PATTERNS.items():
            record[metric] = extract_first_match(text_blob, pattern)
        records.append(record)

    df_final = pd.DataFrame(records, columns=[
        "run_number",
        "mers",
        "epochs",
        "eval_accuracy",
        "eval_f1",
        "eval_loss",
        "eval_matthews_correlation",
        "eval_precision",
        "eval_recall",
    ])

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned CSV to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
