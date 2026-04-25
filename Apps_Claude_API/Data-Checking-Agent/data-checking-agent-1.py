# Requirements:
#   pip install pandas openpyxl anthropic tabulate python-dotenv

import sys
import json
import os

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from tabulate import tabulate
import anthropic


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".csv", ".tsv", ".xlsx", ".xls", ".json"):
        print(f"❌  Unsupported file format '{ext}'. Supported: .csv, .tsv, .xlsx, .xls, .json")
        sys.exit(1)

    try:
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path, engine="openpyxl" if ext == ".xlsx" else None)
        if ext == ".json":
            return pd.read_json(path)

        # CSV / TSV — try encodings in order
        sep = "\t" if ext == ".tsv" else ","
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                continue
        # Last resort: ignore undecodable bytes
        return pd.read_csv(path, sep=sep, encoding="latin-1", errors="replace")
    except Exception as e:
        print(f"❌  Could not load '{path}': {e}")
        sys.exit(1)


def build_summary(df: pd.DataFrame, file_path: str) -> dict:
    column_details = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        column_details.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "pct_missing": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
            "num_unique": int(df[col].nunique()),
        })

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    categorical_stats = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        categorical_stats[col] = {
            "top_value": str(vc.index[0]) if len(vc) else None,
            "top_freq": int(vc.iloc[0]) if len(vc) else 0,
            "unique_count": int(df[col].nunique()),
        }

    return {
        "file_name": os.path.basename(file_path),
        "file_format": os.path.splitext(file_path)[1].lstrip("."),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_details": column_details,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_stats": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "categorical_stats": categorical_stats,
        "sample_rows": df.head(5).to_dict(orient="records"),
    }


def print_summary(summary: dict) -> None:
    sep = "━" * 48
    print(f"\n📊 Dataset Summary — {summary['file_name']}")
    print(sep)
    print(
        f"Rows: {summary['num_rows']:,}    "
        f"Columns: {summary['num_columns']}    "
        f"Duplicates: {summary['duplicate_rows']}"
    )

    # Column details table
    print("\nColumns:")
    rows = [
        [c["name"], c["dtype"], c["null_count"], f"{c['pct_missing']}%", c["num_unique"]]
        for c in summary["column_details"]
    ]
    print(tabulate(rows, headers=["Column", "Type", "Nulls", "Missing%", "Unique"], tablefmt="simple"))

    # Numeric stats
    if summary["numeric_stats"]:
        print("\nNumeric Stats:")
        for col, stats in summary["numeric_stats"].items():
            mn  = round(stats.get("min", 0), 2)
            mx  = round(stats.get("max", 0), 2)
            avg = round(stats.get("mean", 0), 2)
            std = round(stats.get("std", 0), 2)
            print(f"  {col} → min: {mn}, max: {mx}, mean: {avg}, std: {std}")

    # Categorical stats
    if summary["categorical_stats"]:
        print("\nCategorical Stats:")
        for col, stats in summary["categorical_stats"].items():
            print(
                f"  {col} → top: '{stats['top_value']}' ({stats['top_freq']}x), "
                f"unique: {stats['unique_count']}"
            )

    # Sample rows
    print("\nSample Rows (first 5):")
    sample = summary["sample_rows"]
    if sample:
        print(tabulate(sample, headers="keys", tablefmt="simple"))

    print(sep)
    print("💬 Ask me anything about this dataset! Type 'exit' to quit.")


# ── Claude Q&A ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a friendly, concise data analyst assistant. You have been given a summary of a dataset.
Answer the user's questions based only on this summary. Keep answers short (2-4 sentences max)
and use a warm, helpful tone. If a question cannot be answered from the summary alone
(e.g. it requires seeing all rows, running a query, or data not captured in the summary),
clearly and kindly explain why — for example: "That would need me to scan every row, which
I can't do from my summary. You could run df.query(...) in pandas to check!"
Never make up data values that aren't in the summary.
""".strip()


def ask_claude(question: str, summary: dict, client: anthropic.Anthropic) -> str:
    summary_text = json.dumps(summary, indent=2, default=str)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Dataset summary:\n{summary_text}\n\nUser question: {question}",
            }
        ],
    )
    return response.content[0].text


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python data-checking-agent.py <path-to-dataset>")
        print("Supported formats: .csv, .tsv, .xlsx, .xls, .json")
        sys.exit(0)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  ANTHROPIC_API_KEY not found.")
        print("    Add it to a .env file in this directory: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_dataset(file_path)
    summary = build_summary(df, file_path)
    print_summary(summary)

    client = anthropic.Anthropic(api_key=api_key)

    while True:
        try:
            question = input("\n🔍 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Thanks for using the data checker. Goodbye!")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("👋 Thanks for using the data checker. Goodbye!")
            break
        if not question:
            continue

        try:
            answer = ask_claude(question, summary, client)
            print(f"\n💡 {answer}")
        except anthropic.AuthenticationError:
            print("❌  Invalid API key. Check your ANTHROPIC_API_KEY.")
        except Exception as e:
            print(f"❌  Something went wrong calling the Claude API: {e}")


if __name__ == "__main__":
    main()
