import nltk
from sklearn import svm
import csv
import numpy
import re

training_data = csv.reader(open('trainingdata/clues_tags.csv'))

tagdict = nltk.load('help/tagsets/upenn_tagset.pickle')
tags = list(tagdict.keys())

# tag to category integer
def tag_to_class(tag):
    return tags.index(tag)

# tag length, words in each clue, capital letters in the clue, probability tags associated with the clue
X = []
y = []

capitalre = re.compile(r'[A-Z]')
for i, l in enumerate(training_data):
    clue_tokens = nltk.word_tokenize(l[2])
    num_words = len(clue_tokens),
    num_capital = len(capitalre.findall(l[2]))
    targ_len = len(l[3])
    tag_vec = [tag_to_class(tag) for (_, tag) in nltk.pos_tag(clue_tokens)] # TODO: currently not a list of probability vectors, instead a list of category integers
    X.append([num_words, num_capital, targ_len] + tag_vec)
    y.append(tag_to_class(l[4]))
    if i > 100:
        break

# TODO: not sure how to use token probability vectors as a feature because these are variable length. Might be something to ask Prof Blake.
X = numpy.array(X)
y = numpy.array(y)


clf = svm.SVC()
clf.fit(X,y)