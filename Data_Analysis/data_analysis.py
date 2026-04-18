import pandas as pd

input_path = input("Enter the input CSV filename and full path: ").strip()
output_filename = input("Enter the output CSV filename: ").strip()

try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File '{input_path}' not found.")
    exit(1)

required_columns = {"Duration", "Pulse", "Calories"}
missing = required_columns - set(df.columns)
if missing:
    print(f"Error: Missing column(s) in CSV: {', '.join(missing)}")
    exit(1)

result = df.groupby("Duration").agg(
    count=("Duration", "count"),
    avg_Pulse=("Pulse", "mean"),
    avg_Calories=("Calories", "mean"),
).reset_index()

result.to_csv(output_filename, index=False)
print(f"Done! Results saved to '{output_filename}'.")
