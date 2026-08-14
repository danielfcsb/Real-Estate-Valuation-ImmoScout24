from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent
csv_files = sorted(base_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No .csv file found in: {base_dir}")

file_path = csv_files[0]

# Read CSV robustly (adjust sep if needed)
df = pd.read_csv(file_path, low_memory=False)  # if your CSV uses ';', add sep=";"

# Column B (second column)
col_b = df.iloc[:, 1].dropna()

# Convert to string + trim spaces
col_b_str = col_b.astype(str).str.strip()

# Build a normalized key for case-insensitive uniqueness
key = col_b_str.str.lower()

# Keep first occurrence of each key (preserves original order)
mask_first = ~key.duplicated(keep="first")

unique_values = col_b_str[mask_first]

print(f"File: {file_path.name}")
print(f"Column B header: {df.columns[1]}\n")
print("Unique values from column B (case-insensitive):\n")

for v in unique_values:
    if v != "":
        print(v)