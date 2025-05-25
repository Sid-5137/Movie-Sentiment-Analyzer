import nltk
import random
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download("movie_reviews")

print("Total reviews:", len(movie_reviews.fileids()))
print("Positive reviews:", len(movie_reviews.fileids('pos')))
print("Negative reviews:", len(movie_reviews.fileids('neg')))

print("Printing Categories... ")
print(" ".join(f"{i+1}. {cat}" for i, cat in enumerate(movie_reviews.categories())))

docs = [(movie_reviews.raw(fileid), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)

texts, labels = zip(*docs)
labels = [1 if label == 'pos' else 0 for label in labels]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=4000, ngram_range=(1, 2), max_df=0.95, min_df=2, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

selector = SelectKBest(score_func=chi2, k=4000)
X_train_sel = selector.fit_transform(X_train_vec, y_train)
X_test_sel = selector.transform(X_test_vec)

# Model training
model = LinearSVC()
model.fit(X_train_sel, y_train)

y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.4f}")

with open("sentiment_svc_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)