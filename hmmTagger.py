# TODO: trying to train an HMM tagger to extract the prior distribution. Unfortunately the NLTK classes aren't very transparent and I didn't have that much luck with this. If you want to give it a try, might have better luck

import nltk
import pickle
from nltk.util import unique_list

corpus = nltk.corpus.brown.tagged_sents()

tag_set = unique_list(tag for sent in corpus for (word, tag) in sent)

symbols = unique_list(word for sent in corpus for (word,tag) in sent)

trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)

train_corpus = []
test_corpus = []

for i in range(len(corpus)):
    if i % 10:
        train_corpus += [corpus[i]]
    else:
        test_corpus += [corpus[i]]

def train_and_test(est):
    return trainer.train_supervised(train_corpus, estimator = est)

hmm = train_and_test(nltk.probability.LaplaceProbDist)

f = open('model.pkl', 'wb')
pickle.dump(hmm, f)