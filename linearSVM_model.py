import json as j
import pandas as pd
import re # regular expression
import numpy as np
import nltk
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm


json_data = None
with open('../data/yelp_academic_dataset_review.json/yelp_academic_dataset_review.json') as data_file:
    # make sure to specify the correct path to the dataset(.json file)
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]" # array conversion

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data) # using data as dataframe for manipulation
print(data.shape)
#print(data['stars'])
import string


def get_words(text):
    '''
    Takes in a string of text, then performs the following:
    1. Performs case normalization
    2. Remove all punctuation
    3. Remove all stopwords
    4. Return the cleaned text as a list of words
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['ive', 'hadnt', 'couldnt', 'didnt', 'id']  ## more can also be added upon analysis
    stopwords.extend(newStopWords)

    # text format of b'Review_starts' is beacuse of some encoding stuff, so we will remove it to make our review a
    # string like 'sample review'
    text = text[2: len(sample_review) - 1].lower()  ##  case normalization
    # print(text)

    text = text.replace('\\n', ' ').replace('\\t', ' ')
    print(text)
    nopun = [char for char in text if char not in string.punctuation]
    nopun = ''.join(nopun)

    print(nopun)

    l = [word for word in nopun.split() if word.lower() not in stopwords]

    return l, len(l)

for i in range(1):
    sample_review = str(data.text[i])
    #display(sample_review)
    check = get_words(sample_review)
    print(check[0]) # a tuple

#result
#wife took me here on my birthday for breakfast and it was excellent.  the weather was perfect which made sitting outside overlooking their grounds an absolute pleasure.  our waitress was excellent and our food arrived quickly on the semi-busy saturday morning.  it looked like the place fills up pretty quickly so the earlier you get here the better.

#do yourself a favor and get their bloody mary.  it was phenomenal and simply the best i've ever had.  i'm pretty sure they only use ingredients from their garden and blend them fresh when you order it.  it was amazing.

#while everything on the menu looks excellent, i had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious.  it came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete.  it was the best "toast" i've ever had.

#anyway, i can't wait to go back
# wife took me here on my birthday for breakfast and it was excellent  the weather was perfect which made sitting outside overlooking their grounds an absolute pleasure  our waitress was excellent and our food arrived quickly on the semibusy saturday morning  it looked like the place fills up pretty quickly so the earlier you get here the better

#do yourself a favor and get their bloody mary  it was phenomenal and simply the best ive ever had  im pretty sure they only use ingredients from their garden and blend them fresh when you order it  it was amazing

#while everything on the menu looks excellent i had the white truffle scrambled eggs vegetable skillet and it was tasty and delicious  it came with 2 pieces of their griddled bread with was amazing and it absolutely made the meal complete  it was the best toast ive ever had

#anyway i cant wait to go back
#['wife', 'took', 'birthday', 'breakfast', 'excellent', 'weather', 'perfect', 'made', 'sitting', 'outside', 'overlooking', 'grounds', 'absolute', 'pleasure', 'waitress', 'excellent', 'food', 'arrived', 'quickly', 'semibusy', 'saturday', 'morning', 'looked', 'like', 'place', 'fills', 'pretty', 'quickly', 'earlier', 'get', 'better', 'favor', 'get', 'bloody', 'mary', 'phenomenal', 'simply', 'best', 'ever', 'im', 'pretty', 'sure', 'use', 'ingredients', 'garden', 'blend', 'fresh', 'order', 'amazing', 'everything', 'menu', 'looks', 'excellent', 'white', 'truffle', 'scrambled', 'eggs', 'vegetable', 'skillet', 'tasty', 'delicious', 'came', '2', 'pieces', 'griddled', 'bread', 'amazing', 'absolutely', 'made', 'meal', 'complete', 'best', 'toast', 'ever', 'anyway', 'cant', 'wait', 'go', 'back']

#analysing input data
pd.set_option('display.precision', 2)
print(data.describe())


#representation of data in pie chart

labels = '5-Stars', '4-Stars', '1-Star', '3-Stars', '2-Stars'
sizes = data["stars"].value_counts()
colors = ['red', 'blue', 'orange', 'lightskyblue', 'green']

# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')
plt.show()

# we're interested in the text of each review
# and the stars rating, so we load these into
# separate lists

texts = []
stars = [data['stars'] for review in data]
pbar = tqdm(total=data.shape[0]+1)
for index, row in data.iterrows():
    texts.append(get_words(row['text']))
    pbar.update(1)
pbar.close()

# Vectorizing our Text Data - the TF-IDF algorithm along with n-grams
# and tokenization (splitting the text into individual words).

# Estimated time: 40 s
from sklearn.feature_extraction.text import TfidfVectorizer

# This vectorizer breaks text into single words and bi-grams
# and then calculates the TF-IDF representation
vectorizer = TfidfVectorizer(ngram_range=(1,3))

# the 'fit' builds up the vocabulary from all the reviews
# while the 'transform' step turns each indivdual text into
# a matrix of numbers.
vectors = vectorizer.fit_transform(texts)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, stars[1], test_size=0.15, random_state=42, shuffle =False)

# We now have 15% of our data in X_test and y_test. We’ll teach our system using 85%
# of the data (X_train and y_train), and then see how well it does by comparing its predictions for
# the reviews in X_test with the real ratings in y_test.

# Estimated time: 20 s
from sklearn.svm import LinearSVC

# initialise the SVM classifier
classifier = LinearSVC()

# train the classifier
classifier.fit(X_train, y_train)

# classifier has been fitted, it can now be used to make predictions.
# predicting the rating for the first ten reviews in our test set


# Using our trained classifier to predict the ratings from text

preds = classifier.predict(X_test)
#print("Actual Ratings(Stars): ",end = "")
#print(y_test[:5])
#print("Predicted Ratings: ",end = "")
#print(preds[:5])

# performance indices

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(accuracy_score(y_test, preds))

print ('Precision: ' + str(precision_score(y_test, preds, average='weighted')))
print ('Recall: ' + str(recall_score(y_test, preds, average='weighted')))

from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn import metrics
names = ['1','2','3','4','5']

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, preds)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


##analysing for only 2 classes
#i.e. if star rating>3 ---->> 5 star or 'p' for positive review
#     if star rating<=3 ---->> 1 star or 'n' for negative review

sentiments = []
for star in stars[1]:
    if star <= 3:
        sentiments.append('n')
    if star > 3:
        sentiments.append('p')


X2_train, X2_test, y2_train, y2_test = train_test_split(vectors, sentiments, test_size=0.20, random_state=42)
classifier2 = LinearSVC()
# train the classifier
classifier2.fit(X2_train, y2_train)

preds2 = classifier2.predict(X2_test)
print("Actual Class:    ",end = "")
print(y2_test[:10])
print("\nPredicted Class: ",end = "")
print(list(preds2[:10]))



print(accuracy_score(y2_test, preds2))
# result is found to be 92.6% accurate.

#classificaiton report for binary classification
print(classification_report(y2_test, preds2))
