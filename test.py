
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
    
'''  
from nltk.corpus import cmudict
d = cmudict.dict()

print(count_syl("banana", d))      # Expected: 3
print(count_syl("syllable", d))    # Expected: 3
print(count_syl("crypt", d))       # Fallback estimate: 1
print(count_syl("euphoria", d))    # Expected: 4
'''

def fk_level(text, d):
    from nltk.tokenize import sent_tokenize, word_tokenize

    # Separate text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Filter out non-alphabetic tokens
    
    filtered_words = []
    for w in words:
        if w.isalpha():
            filtered_words.append(w)
    words = filtered_words


    # Count total syllables using the dictionary
    syllables = 0
    for word in words:
        lower_word = word.lower()
        syllables += count_syl(lower_word, d)


    # FK formula
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    fk_score = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59

    return fk_score

def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results



#TEST
# Calculate FK scores
fk_scores = get_fks(df)
for title, score in fk_scores.items():
    print(f"{title} (FK): {score}")

# Add scores column to DataFrame
df["fk"] = df["title"].map(fk_scores)

print(df[["title", "ttr", "fk"]].head(10))  # Display key results




import spacy
import pandas as pd
from pathlib import Path
import pickle

'''
def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    nlp = spacy.load("en_core_web_sm")
    df["parsed"] = df["text"].apply(nlp)

    return df
   

    
    import pandas as pd

# Sample data
sample_df = pd.DataFrame({
    "title": ["Test Novel"],
    "text": ["This is a short sentence. And here is another."]
})

# Call the parse function
parsed_df = parse(sample_df)

# Check the new column
print(parsed_df[["title", "parsed"]])
print("\nTokens in first doc:", [token.text for token in parsed_df["parsed"][0]])


'''

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    nlp = spacy.load("en_core_web_sm")

    max_text_len = df["text"].str.len().max()
    nlp.max_length = max_text_len + 1000

    df["parsed"] = df["text"].apply(nlp)

    store_path.mkdir(parents=True, exist_ok=True)

    out_file = store_path / out_name
    with open(out_file, "wb") as f:
        pickle.dump(df, f)



    return df
   
parsed_df = parse(df)


from collections import Counter

def adjective_counts(doc):
    adjectives = []
    for token in doc:
        if token.pos_ == "ADJ":
            adjectives.append(token.text.lower())
    return Counter(adjectives).most_common(10)

# ----
from collections import Counter
parsed_df = parse(df)
def object_counts(doc):
    object_labels = {"obj", "dobj", "pobj", "dative"}
    objects = []
    for token in doc:
        if token.dep_ in object_labels:
            objects.append(token.text.lower())
    return Counter(objects).most_common(10)

for i, row in parsed_df.iterrows():
    print(row["title"])
    print(object_counts(row["parsed"]))
    print()




