
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

    # Count parties and keep top 4
    party_counts = df['party'].value_counts()
    top_parties = party_counts.head(4).index.tolist()

    # Remove 'Speaker' if it's in top 4
    if 'Speaker' in top_parties:
        top_parties.remove('Speaker')

    # Keep only 'Speech'
    df = df[df['speech_class'] == 'Speech']

    # Remove short speeches
    df = df[df['speech'].str.len() >= 1000]

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

### Task 2(c): Train and Evaluate Classifiers

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=300, random_state=26)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Print results for Random Forest
rf_f1 = f1_score(y_test, rf_preds, average='macro')
print("Random Forest Macro F1 Score:", rf_f1)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds, zero_division=0))

# Train SVM with linear kernel
svm_model = SVC(kernel='linear', random_state=26)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Print results for SVM
svm_f1 = f1_score(y_test, svm_preds, average='macro')
print("SVM Macro F1 Score:", svm_f1)
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))