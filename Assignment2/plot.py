import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('reviews.tsv', sep='\t',
                   header=None, quoting=csv.QUOTE_NONE)
rating = np.array(data[1])
print(rating)
# plot the distribution of topics
rating_dict = {}
for i in rating:
    if i not in rating_dict:
        rating_dict[i] = 1
    else:
        rating_dict[i] += 1

rating_classes = sorted(rating_dict.keys())
rating_classes_number = []
for c in rating_classes:
    rating_classes_number.append(rating_dict[c])
print(rating_classes)
print(rating_classes_number)
x = np.arange(len(rating_classes))

plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Frequency distribution over ratings')
plt.bar(x, rating_classes_number)
plt.xticks(x, rating_classes, size='xx-small')
for a, b in zip(x, rating_classes_number):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.show()
