print("hello")





import nltk

import spacy
from pathlib import Path
import pandas as pd

# Download required resources (run only once)
nltk.download('punkt')
nltk.download('cmudict')


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):

    rows = []  # List to store each novel's data

    # Loop through all .txt files in the directory
    for file in path.glob("*.txt"):

        # Extract title, author, and year from the filename
        title, author, year = file.stem.split("-")

        # Open and read the contents of the file
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        # Add the data as a dictionary to the list
        rows.append(
            {
                "text": text,
                "title": title.strip(),
                "author": author.strip(),
                "year": int(year.strip())
            }
        )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    # Sort the DataFrame by year and reset the index
    df = df.sort_values(by="year").reset_index(drop=True)

    return df  # Return the final DataFrame




if __name__ == "__main__":
    df = read_novels()
    print(df.head(20))  # Show the first few rows
    print(df.columns) # Check column names




    
def read_novels(path=Path.cwd() / "p1-texts" / "novels"):

    rows = []  # List to store each novel's data

    # Loop through all .txt files in the directory
    for file in path.glob("*.txt"):

        # Extract title, author, and year from the filename
        title, author, year = file.stem.split("-")

        # Open and read the contents of the file
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        # Add the data as a dictionary to the list
        rows.append(
            {
                "text": text,
                "title": title.strip(),
                "author": author.strip(),
                "year": int(year.strip())
            }
        )

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    # Sort the DataFrame by year and reset the index
    df = df.sort_values(by="year").reset_index(drop=True)

    return df  # Return the final DataFrame


#--------------

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results

from nltk.tokenize import word_tokenize

def nltk_ttr(text):
   
    tokens = word_tokenize(text)

  
    words = []
    for token in tokens:
        if token.isalpha():       
            words.append(token.lower())  

    if not words:
        return 0

    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    return round(ttr, 3)


# Calculate TTR scores 
ttr_scores = get_ttrs(df)
for title, score in ttr_scores.items():
    print(f"{title}: {score}")

# Add scores column to DataFrame
df["ttr"] = df["title"].map(ttr_scores)

print(df.head())



def count_syl(word, d):
 
    if word in d:
        count = 0
        for sound in d[word][0]:
            if sound[-1].isdigit():
                count += 1
        return count
    else:
        return 1
    
from nltk.corpus import cmudict
d = cmudict.dict()

print(count_syl("banana", d))      # Expected: 3
print(count_syl("syllable", d))    # Expected: 3
print(count_syl("crypt", d))       # Fallback estimate: 1
print(count_syl("euphoria", d))    # Expected: 4


