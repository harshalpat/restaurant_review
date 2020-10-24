import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import json as j

json_data = None
#with open('../data/yelp_academic_dataset_review.json/yelp_academic_dataset_review.json') as data_file:
with open('C:\Users\harshal\Downloads\text_classification-master\text_classification-master\data\yelp_academic_dataset_review.json\yelp_academic_dataset_review.json') as data_file:
    # make sure to specify the correct path to the dataset(.json file)
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]" # array conversion

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data) # using data as dataframe for manipulation
print(data.head())
# Creating X and Y for the classifier. X is the review text and Y is the rating
x = data['text']
y = data['stars']
#print(x[1])

# Text preprocessing
import string
def text_preprocessing(text):
    no_punctuation = [ch for ch in text if ch not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return [w for w in no_punctuation.split() if w.lower() not in stopwords.words('english')]



# Vectorization
# Converting each review into a vector using bag-of-words approach

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(analyzer=text_preprocessing).fit(x)
x1 = vector.transform(x)

# Spitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x1, y, test_size=0.20, random_state=0, shuffle =False)

# Building Multinomial Naive Bayes modle and fit it to our training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)

# Using our trained classifier to predict the ratings from text
# Testing our model on the test set

preds = classifier.predict(X_test)
print("Actual Ratings(Stars): ",end = "")
#display(Y_test[:15])
print("Predicted Ratings: ",end = "")
print(preds[:15])



# Accuracy of the model

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, preds)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print ('Precision: ' + str(precision_score(Y_test, preds, average='weighted')))
print ('Recall: ' + str(recall_score(Y_test,preds, average='weighted')))

# Evaluating the model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, preds))
print('\n')
print(classification_report(Y_test, preds))


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
class_names = ['1','2','3','4','5']

# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(Y_test, preds
                                     )
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

##analysing for only 2 classes
#i.e. if star rating>3 ---->> 5 star or 'p' for positive review
#     if star rating<=3 ---->> 1 star or 'n' for negative review

# Reducing the dataset to 2 classes i.e 1 and 5 star rating
data['stars'][data.stars == 3] = 1
data['stars'][data.stars == 2] = 1
data['stars'][data.stars == 4] = 5

#Undersampling of the dataset to get a balanced dataset
review1 = data[data['stars'] == 1]
review5 = data[data['stars'] == 5][0:34062]
frames = [review1, review5]
reviews = pd.concat(frames)

x2 = reviews['text']
y2 = reviews['stars']

# Vectorization
# Converting each review into a vector using bag-of-words approach

from sklearn.feature_extraction.text import CountVectorizer
vector2 = CountVectorizer(analyzer=text_preprocessing).fit(x2)
x2 = vector.transform(x2)


# Spitting data into training and test set
from sklearn.model_selection import train_test_split
X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y2, test_size=0.20, random_state=0)


# Building Multinomial Naive Bayes modle and fit it to our training set
from sklearn.naive_bayes import MultinomialNB
classifier2 = MultinomialNB()
classifier2.fit(X2_train, Y2_train)


MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)


# Testing our model on the test set
Y2_pred = classifier2.predict(X2_test)

#Classification Report

# Evaluating the model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y2_test, Y2_pred))
print('\n')
print(classification_report(Y2_test, Y2_pred))

#accuracy of SVM turns out to be 88.3% for binary classification

