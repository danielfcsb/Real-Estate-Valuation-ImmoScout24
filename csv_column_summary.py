from pathlib import Path
import pandas as pd

# ====== PARAMETERS YOU CHANGE ======
COLUMN_LETTER = "D"     # change to "D", "E", etc.
CSV_NAME = None         # e.g. "apr20_price - Copy.csv" or keep None to auto-pick first CSV
SEP = None              # None = auto (tries comma first). Set ";" if you know it's semicolon.
CASE_INSENSITIVE = True # True: Darmstadt == DARMSTADT
STRIP_SPACES = True     # True: " Darmstadt " == "Darmstadt"
# ===================================

def excel_col_to_index(letter: str) -> int:
    # "A"->0, "B"->1, ..., "Z"->25, "AA"->26 ...
    letter = letter.strip().upper()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1

base_dir = Path(__file__).resolve().parent

# Pick CSV
if CSV_NAME:
    file_path = base_dir / CSV_NAME
    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")
else:
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv file found in: {base_dir}")
    file_path = csv_files[0]

# Read CSV
read_kwargs = {"low_memory": False}
if SEP is not None:
    read_kwargs["sep"] = SEP

df = pd.read_csv(file_path, **read_kwargs)

# Choose column by letter
col_idx = excel_col_to_index(COLUMN_LETTER)
if col_idx < 0 or col_idx >= df.shape[1]:
    raise IndexError(f"Column {COLUMN_LETTER} is out of range. CSV has {df.shape[1]} columns.")

col = df.iloc[:, col_idx].dropna()

# Convert to string so we can normalize and compare
col_str = col.astype(str)

if STRIP_SPACES:
    col_str = col_str.str.strip()

# Build key for uniqueness
key = col_str
if CASE_INSENSITIVE:
    key = key.str.lower()

# Keep first occurrence (preserves original order)
unique_values = col_str[~key.duplicated(keep="first")]

print(f"File: {file_path.name}")
print(f"Analyzed column: {COLUMN_LETTER} (header: {df.columns[col_idx]})\n")
print("Unique values:\n")

for v in unique_values:
    if v != "":
        print(v)
