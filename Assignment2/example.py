# Load libraries
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,classification_report
from sklearn import tree

def predict_and_test(model, X_test_bag_of_words):
    num_dec_point = 3
    predicted_y = model.predict(X_test_bag_of_words)
    print(y_test, predicted_y)
    print(model.predict_proba(X_test_bag_of_words))
    a_mic = accuracy_score(y_test, predicted_y)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test,
                        predicted_y,
                        average='micro',
                        warn_for=())
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test,
                        predicted_y,
                        average='macro',
                        warn_for=())
    print('micro acc,prec,rec,f1: ',round(a_mic,num_dec_point), round(p_mic,num_dec_point), round(r_mic,num_dec_point), round(f1_mic,num_dec_point),sep="\t")
    print('macro prec,rec,f1: ',round(p_mac,num_dec_point), round(r_mac,num_dec_point), round(f1_mac,num_dec_point),sep="\t")

# Create text
text_data = np.array(['I love Brazil. Brazil is best',
                      'I like Italy, because Italy is beautiful',
                      'Malaysia is ok, but I do not like spicy food',
                      'I like Germany more, Germany beats all',
                      'I do not like hot weather in Singapore'])
X = text_data
# Create target vector
y = ['positive','positive','negative','positive','negative']

# split into train and test
X_train = X[:3]
print("xtrain",X_train)
X_test = X[3:]
print("xtest",X_test)
y_train = y[:3]
print("ytrain",y_train)
y_test = y[3:]
print("ytest",y_test)

# create count vectorizer and fit it with training data
count = CountVectorizer()
X_train_bag_of_words = count.fit_transform(X_train)

# transform the test data into bag of words creaed with fit_transform
X_test_bag_of_words = count.transform(X_test)

print("----bnb")
clf = BernoulliNB()
model = clf.fit(X_train_bag_of_words, y_train)
predict_and_test(model, X_test_bag_of_words)

print("----mnb")
clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, y_train)
predict_and_test(model, X_test_bag_of_words)

# if random_state id not set. the feaures are randomised, therefore tree may be different each time
print("----dt")
clf = tree.DecisionTreeClassifier(min_samples_leaf=1,criterion='entropy',random_state=0)
model = clf.fit(X_train_bag_of_words, y_train)
predict_and_test(model, X_test_bag_of_words)