

from pathlib import Path
import sys

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: pip install pandas seaborn matplotlib"
    ) from exc

DATA_FILE = "context_kmer_results.csv"
OUTPUT_DIR = Path("graphs_output")

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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Trim whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Coerce numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prep_output():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_context("talk", font_scale=1.05)


def savefig(name: str):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_accuracy_by_context(df: pd.DataFrame):
    agg = (
        df.groupby(["kmer", "context_window", "epochs"], as_index=False)["Accuracy"]
        .mean()
        .sort_values(["kmer", "context_window"])
    )
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=agg,
        x="context_window",
        y="Accuracy",
        hue="kmer",
        style="epochs",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Context window")
    plt.ylabel("Accuracy (mean)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="kmer / epochs", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("Accuracy vs Context Window by k-mer")
    savefig("accuracy_by_context_kmer.png")


def plot_f1_box(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="context_window", y="F1 Score", hue="kmer")
    plt.xlabel("Context window")
    plt.ylabel("F1 Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="kmer", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("F1 Score distribution by Context Window and k-mer")
    savefig("f1_by_context_kmer_box.png")


def plot_runtime(df: pd.DataFrame):
    agg = (
        df.groupby(["kmer", "context_window", "epochs"], as_index=False)["Runtime"]
        .mean()
        .sort_values(["kmer", "context_window"])
    )
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=agg,
        x="context_window",
        y="Runtime",
        hue="kmer",
        style="epochs",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Context window")
    plt.ylabel("Runtime (s, mean)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="kmer / epochs", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("Runtime vs Context Window by k-mer")
    savefig("runtime_by_context_kmer.png")


def plot_throughput(df: pd.DataFrame):
    agg = (
        df.groupby(["kmer", "context_window", "epochs"], as_index=False)["Samples/sec"]
        .mean()
        .sort_values(["kmer", "context_window"])
    )
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=agg,
        x="context_window",
        y="Samples/sec",
        hue="kmer",
        style="epochs",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Context window")
    plt.ylabel("Samples/sec (mean)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="kmer / epochs", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("Throughput (Samples/sec) vs Context Window by k-mer")
    savefig("throughput_by_context_kmer.png")


def plot_accuracy_vs_runtime(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="Runtime",
        y="Accuracy",
        hue="context_window",
        style="kmer",
        palette="viridis",
        s=60,
    )
    plt.xlabel("Runtime (s)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="context_window / kmer", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("Accuracy vs Runtime (colored by context window)")
    savefig("accuracy_vs_runtime.png")


def plot_precision_recall(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="Precision",
        y="Recall",
        hue="kmer",
        style="epochs",
        s=60,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="kmer / epochs", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title("Precision vs Recall by k-mer/epochs")
    savefig("precision_recall_scatter.png")


def plot_corr_heatmap(df: pd.DataFrame):
    metrics = ["Accuracy", "F1 Score", "Matthews Correlation", "Precision", "Recall", "Runtime", "Samples/sec", "Steps/sec"]
    corr = df[metrics].corr().round(3)
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        corr,
        annot=True,
        annot_kws={"size": 7},
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Heatmap")
    savefig("correlation_heatmap.png")


def main():
    df = load_data()
    prep_output()

    plot_accuracy_by_context(df)
    plot_f1_box(df)
    plot_runtime(df)
    plot_throughput(df)
    plot_accuracy_vs_runtime(df)
    plot_precision_recall(df)
    plot_corr_heatmap(df)


if __name__ == "__main__":
    main()
