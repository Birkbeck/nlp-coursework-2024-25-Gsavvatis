import pickle
from pathlib import Path

pickle_path = Path.cwd() / "pickles" / "parsed.pickle"

with open(pickle_path, "rb") as f:
    loaded_df = pickle.load(f)


print(loaded_df.columns)  # Confirm 'parsed' column exists
print(loaded_df.head(3))  # Show first few rows
