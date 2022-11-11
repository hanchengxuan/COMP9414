# Load libraries
import re
import sys
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split

# read file from command line input
data = pd.read_csv(sys.stdin, sep='\t',
                   header=None, quoting=csv.QUOTE_NONE)

# get comments and rating
sequence_number = np.array(data[0])
rating = np.array(data[1])
comment = np.array(data[2])

# rating_sentiments = np.array(data[1]).astype(str)
# for i in range(0, len(rating)):
#     if rating[i] < 4:
#         rating_sentiments[i] = "negative"
#     elif rating[i] == 4:
#         rating_sentiments[i] = "neutral"
#     else:
#         rating_sentiments[i] = "positive"

# set training set(80%) and test set(20%)
sequence_number_train, sequence_number_test, comment_train, comment_test, rating_train, rating_test = \
    train_test_split(sequence_number, comment, rating, test_size=0.2, random_state=10, shuffle=False)


def predict_and_test(model, comment_test_bag_of_words):
    num_dec_point = 3
    predicted_y = model.predict(comment_test_bag_of_words)
    # print(rating_test, predicted_y)
    # print(model.predict_proba(comment_test_bag_of_words))
    a_mic = accuracy_score(rating_test, predicted_y)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(rating_test,
                                                              predicted_y,
                                                              average='micro',
                                                              warn_for=())
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(rating_test,
                                                              predicted_y,
                                                              average='macro',
                                                              warn_for=())
    # print('micro acc,prec,rec,f1: ', round(a_mic, num_dec_point), round(p_mic, num_dec_point),
    #       round(r_mic, num_dec_point), round(f1_mic, num_dec_point), sep="\t")
    # print('macro prec,rec,f1: ', round(p_mac, num_dec_point), round(r_mac, num_dec_point), round(f1_mac, num_dec_point),
    #       sep="\t")
    #
    # topic_classes = ['micro-acc', 'micro-prec', 'micro-rec', 'micro-f1', 'macro-prec', 'macro-rec', 'macro-f1', ]
    # topic_classes_number = [round(a_mic, num_dec_point), round(p_mic, num_dec_point), round(r_mic, num_dec_point),
    #                         round(f1_mic, num_dec_point), round(p_mac, num_dec_point), round(r_mac, num_dec_point),
    #                         round(f1_mac, num_dec_point)]
    # x = np.arange(len(topic_classes))
    # plt.xlabel('micro acc,prec,rec,f1 AND macro prec,rec,f1')
    # plt.ylabel('value')
    # plt.title(' ')
    # plt.bar(x, topic_classes_number)
    # plt.xticks(x, topic_classes, size='xx-small')
    # for a, b in zip(x, topic_classes_number):
    #     plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=8)
    # plt.tight_layout()
    # plt.show()


def data_processing(comment):
    """
    the function is used to replace two successive hyphens - -, the tilde symbol ˜ and
    any ellipsis (three or more dots ...) by a space,
    then removing tags (minimal text spans between < and > inclusive) and all other characters.
    """
    processed_comment = []
    for sentence in comment:
        replace_format = re.compile(r'\-{2,}|\.{3,}|\~')
        temp = re.sub(replace_format, ' ', sentence, re.MULTILINE | re.IGNORECASE)
        temp = re.sub(r'<.*?>', '', temp)
        temp = re.sub(r'[^\/\-\$\%\s\w\d]', '', temp)
        temp = re.findall(r'[\/\-\$\%\s\w\d]{2,}', temp)
        result = ' '.join(e for e in temp)
        processed_comment.append(result)
    return processed_comment


# replace specific characters in training and testing sentences
processed_comment_train, processed_comment_test = data_processing(comment_train), data_processing(comment_test)

# store the result into counter, add 'max_feature' to limit the number of feature
# lowercase default is True and convert and words to lowercase,
# if u want to remain the uppercase, set it to lowercase = False
token = r'[a-zA-Z0-9$%/-]{2,}'
count = CountVectorizer(token_pattern=token, max_features=3100)
comment_train_bag_of_words = count.fit_transform(processed_comment_train)
comment_test_bag_of_words = count.transform(processed_comment_test)
count.get_feature_names_out()


clf = MultinomialNB()
model = clf.fit(comment_train_bag_of_words, rating_train)
predict_y = model.predict(comment_test_bag_of_words)
predict_and_test(model, comment_test_bag_of_words)

# print the valid output
for i in range(len(comment_test)):
    print(f'{sequence_number_test[i]} {predict_y[i]} ')
