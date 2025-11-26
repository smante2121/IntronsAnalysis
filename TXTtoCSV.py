from pathlib import Path
import csv
import re
from typing import Dict, List

# Input files to parse
INPUT_FILES = [
    "ContextWindow&K-merExperiments1.txt",
    "ContextWindow&K-merExperiments2.txt",
    "ContextWindow&K-merExperiments3.txt",
]

# Output CSV
OUTPUT_FILE = "context_kmer_results.csv"

# Regex to capture all fields from a single experiment block.
# Allows whitespace/newlines between fields and handles decimals or scientific notation.
BLOCK_PATTERN = re.compile(
    r"Configuration:\s*kmer=([^,]+),\s*context_window=([^,]+),\s*epochs=([^,]+),\s*run=([^\n]+)"
    r".*?Accuracy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*F1 Score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*Matthews Correlation:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*Precision:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*Recall:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*Runtime:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)s"
    r"\s*Samples/sec:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*Steps/sec:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Column order for CSV
COLUMNS = [
    "run_num",
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


def parse_file(path: Path) -> List[Dict[str, str]]:
    text = path.read_text()
    rows: List[Dict[str, str]] = []

    for match in BLOCK_PATTERN.finditer(text):
        (
            kmer,
            context_window,
            epochs,
            run,
            acc,
            f1,
            mcc,
            precision,
            recall,
            runtime,
            samples_sec,
            steps_sec,
        ) = match.groups()

        rows.append(
            {
                "kmer": kmer.strip(),
                "context_window": context_window.strip(),
                "epochs": epochs.strip(),
                "run": run.strip(),
                "Accuracy": acc,
                "F1 Score": f1,
                "Matthews Correlation": mcc,
                "Precision": precision,
                "Recall": recall,
                "Runtime": runtime,
                "Samples/sec": samples_sec,
                "Steps/sec": steps_sec,
            }
        )

    return rows


def main() -> None:
    all_rows: List[Dict[str, str]] = []
    for filename in INPUT_FILES:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        all_rows.extend(parse_file(path))

    if not all_rows:
        raise SystemExit("No experiment blocks found in provided text files.")

    # Add sequential run_num as first column
    for idx, row in enumerate(all_rows, start=1):
        row["run_num"] = str(idx)

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
