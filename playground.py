'''import spacy
nlp = spacy.load("en_core_web_sm")

print(nlp.get_pipe("parser").labels)

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The dog was chased by the boy in the park.")

for token in doc:
    print(f"{token.text:<10} {token.dep_:<12} {token.head.text:<10} {token.pos_:<6}")
'''


from collections import Counter
import spacy
from pathlib import Path
import pandas as pd



parsed_df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")

sample_df = parsed_df.head(1)  
'''
# Loop through each parsed document
for i, row in sample_df.iterrows():
    doc = row["parsed"]
    print(f"--- {row['title']} ---")
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == "hear":
            print(token.text, token.lemma_, token.morph)

            
for i, row in parsed_df.iterrows():
    doc = row["parsed"] 
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == "hear":
            print(f"\nVerb: {token.text}")
            for child in token.children:
                print(f"  Child: {child.text} ({child.dep_})")



for i, row in parsed_df.iterrows():
    doc = row["parsed"] 
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == "hear":
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    print("Subject:", child.text, "→", "Verb:", token.text)




for i, row in parsed_df.iterrows():
    doc = row["parsed"] 
    subjects = []

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == "hear":
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subjects.append(child.lemma_.lower())

    print(subjects)
print(Counter(subjects).most_common(10))
'''

# filter out noise
for i, row in sample_df.iterrows():
    doc = row["parsed"]
    raw_subjects = []
    clean_subjects = []

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == "hear":
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    raw_subjects.append(child.text)
                    if not child.is_stop and not child.is_punct:
                        clean_subjects.append(child.lemma_.lower())

from collections import Counter


print(f"\n--- {row['title']} ---")
print("Raw subjects:", Counter(raw_subjects).most_common(5))
print("Filtered subjects:", Counter(clean_subjects).most_common(5))



                        