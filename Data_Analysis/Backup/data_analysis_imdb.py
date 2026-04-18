import pandas as pd

input_path = input("Enter the input CSV filename and full path: ").strip()
output_filename = input("Enter the output CSV filename: ").strip()

try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File '{input_path}' not found.")
    exit(1)

required_columns = {"Genre", "IMDB_Rating", "Series_Title"}
missing = required_columns - set(df.columns)
if missing:
    print(f"Error: Missing column(s) in CSV: {', '.join(missing)}")
    exit(1)

# For each group, find the Series_Title corresponding to the max IMDB_Rating
top_title = df.loc[df.groupby("Genre")["IMDB_Rating"].idxmax(), ["Genre", "Series_Title"]]
top_title = top_title.rename(columns={"Series_Title": "top_rated_title"})

result = df.groupby("Genre").agg(
    count=("Series_Title", "count"),
    avg_imdb_rating=("IMDB_Rating", "mean"),
    max_imdb_rating=("IMDB_Rating", "max"),
).reset_index()

result = result.merge(top_title, on="Genre")

result.to_csv(output_filename, index=False)
print(f"Done! Results saved to '{output_filename}'.")
