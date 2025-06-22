print("hello")


import nltk
import spacy
from pathlib import Path
import pandas as pd

def read_novels(path=Path.cwd() / "p1-texts" / "novels"):


    rows = []
    for file in path.glob("*.txt"):
        
        title, author, year = file.stem.split("-")
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        rows.append(
            {
            "text": text,
            "title": title.strip(),
            "author": author.strip(),
            "year": int(year.strip())
            }
        )
    
    print(rows)
    print("Looking in:", path)
    print("Files found:", list(path.glob("*.txt")))
    df = pd.DataFrame(rows)
    #df = df.sort_values(by="year").reset_index(drop=True)
    return df




if __name__ == "__main__":
    df = read_novels()
    print(df.head(20))  # Show the first few rows
    print(df.columns) # Check column names