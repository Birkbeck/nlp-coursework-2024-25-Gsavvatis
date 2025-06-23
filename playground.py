import spacy
nlp = spacy.load("en_core_web_sm")

print(nlp.get_pipe("parser").labels)

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The dog was chased by the boy in the park.")

for token in doc:
    print(f"{token.text:<10} {token.dep_:<12} {token.head.text:<10} {token.pos_:<6}")
