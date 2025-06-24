
#Task 2(a): Preprocess data
import pandas as pd

def preprocess_hansard_data(filepath):
    """
    Load and preprocess the hansard dataset.

    Steps:
    1. Rename 'Labour (Co-op)' to 'Labour'
    2. Keep only rows where speech_class is 'Speech'
    3. Remove speeches shorter than 1000 characters
    4. Count parties and keep top 4
    5. Remove 'Speaker' if it's in top 4

    Returns:
        Filtered DataFrame 
    """
    # Load data
    df = pd.read_csv(filepath)

    # Rename party
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

    # Keep only 'Speech'
    df = df[df['speech_class'] == 'Speech']

    # Remove short speeches
    df = df[df['speech'].str.len() >= 1000]

    # Count parties and keep top 4
    party_counts = df['party'].value_counts()
    top_parties = party_counts.head(4).index.tolist()

    # Remove 'Speaker' if it's in top 4
    if 'Speaker' in top_parties:
        top_parties.remove('Speaker')

    # Keep only top parties
    df = df[df['party'].isin(top_parties)]

    return df

df= preprocess_hansard_data("p2-texts/hansard40000.csv")

print(df.shape)



# Task 2(b): Convert speeches to TF-IDF vectors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 1: Vectorise the speeches using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf_vectorizer.fit_transform(df['speech'])

# Step 2: Create label vector (party)
y = df['party']

# Step 3: Split into train and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.25, stratify=y, random_state=26
)