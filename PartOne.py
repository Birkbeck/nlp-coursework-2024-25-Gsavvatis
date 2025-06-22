#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.



import nltk
import spacy
from pathlib import Path


# Download required resources (run only once)
nltk.download('punkt')
nltk.download('cmudict')

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass

from nltk.corpus import cmudict
d = cmudict.dict()
def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    
 
    if word in d:
        count = 0
        for sound in d[word][0]:
            if sound[-1].isdigit():
                count += 1
        return count
    else:
        return 1


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    
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

    #return df  # Return the final DataFrame
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


from nltk.tokenize import word_tokenize
def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
  
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


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """


