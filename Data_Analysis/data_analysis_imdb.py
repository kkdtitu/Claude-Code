import pandas as pd

input_path = input("Enter the input CSV filename and full path: ").strip()
output_filename = input("Enter the output CSV filename: ").strip()

try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File not found: {input_path}")
    exit(1)

required_columns = {"Genre", "IMDB_Rating", "Series_Title"}
missing = required_columns - set(df.columns)
if missing:
    print(f"Error: Missing columns in CSV: {', '.join(missing)}")
    exit(1)

grouped = df.groupby("Genre")

count = grouped["IMDB_Rating"].count().rename("Count")
avg_rating = grouped["IMDB_Rating"].mean().rename("Avg_IMDB_Rating")
max_rating = grouped["IMDB_Rating"].max().rename("Max_IMDB_Rating")
top_title = grouped.apply(
    lambda g: g.loc[g["IMDB_Rating"].idxmax(), "Series_Title"]
).rename("Top_Series_Title")

result = pd.concat([count, avg_rating, max_rating, top_title], axis=1).reset_index()

result.to_csv(output_filename, index=False)
print(f"Done! Results saved to: {output_filename}")
