import json as j
import pandas as pd
import re  # regular expression
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


json_data = None

with open(r'C:\Users\earahtp\Downloads\text_classification-master\text_classification-master\data\yelp_academic_dataset_review.json\yelp_academic_Dataset_review') as data_file:
    #make sure that you specify the correct path to the dataset(i.e .json file)
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]" # array conversion

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data) #reading the data as dataframes for manipulation

stemmer = SnowballStemmer('english') # stemmer is used to convert words to its root ex. typing to type
words = stopwords.words("english")  # stop words are those which does not add meaning to the sentence, ex. and, the, etc
#collecting stop words to remove or ignore from the data.

data['cleaned'] = data['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
# created a new column called 'cleaned' in data to store the processed data

X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.stars, test_size=0.2)
#stars column in the data contains the stars of that perticular comment.

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
#TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator.
                     ('chi',  SelectKBest(chi2, k=10000)),
#selecting best words/features using chisquare algorithm
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])
#support vector classifier using 3000 iterations


model = pipeline.fit(X_train, y_train)
#training the processed data

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['1', '2', '3', '4', '5']
#labels / stars / classes
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))

print("accuracy score: " + str(model.score(X_test, y_test)))

print(model.predict(['that was an awesome place. Great food!']))
