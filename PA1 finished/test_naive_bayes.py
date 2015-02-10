# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip


class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]


class BagOfWords(Document):
    def features(self):
        return self.data.split()


class BagOfWordsImproved(Document):
    def features(self):
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        split_data = self.data.split()
        n = 0
        while n <= (len(split_data) - 1):
            split_data[n] = split_data[n].lower().translate(table_no_punct)
            if split_data[n] == u'':
                del split_data[n]
                continue
            n += 1
        return split_data


class BagOfWordsImprovedForBernoulli(Document):
    def features(self):
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        split_data = self.data.split()
        n = 0
        temp = {}
        while n <= (len(split_data) - 1):
            split_data[n] = split_data[n].lower().translate(table_no_punct)
            if (split_data[n] in temp) or split_data[n] == u'':
                del split_data[n]
                continue
            else:
                temp[split_data[n]] = True
            n += 1

        return split_data


class GenderPrefer(Document):
    def features(self):
        """Trivially tokenized words."""
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        list_word_ends = [u'ous', u'able', u'al', u'ful', u'ible', u'ic', u'ive', u'less', u'ly', u'sorry']
        split_data = self.data.split()
        preferred_data = []

        for n in range(len(split_data)):
            split_data[n].lower().translate(table_no_punct)
            for m in range(len(list_word_ends)):
                if split_data[n].endswith(list_word_ends[m]) or split_data[n].startswith(u'apolog'):
                    preferred_data.append(split_data[n])
                    break

        n = -1
        temp = {}
        while n < (len(preferred_data) - 1):
            n += 1
            if preferred_data[n] in temp:
                del preferred_data[n]
                continue
            else:
                temp[preferred_data[n]] = True
        return preferred_data


class NGrams(Document):
    def features(self):
        table_no_punct = dict((ord(i), u'') for i in u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\x0b\x0c\r')
        split_data = self.data.split()
        n_grams_data = []
        for n in range(len(split_data)):
            split_data[n].lower().translate(table_no_punct)

        for n in range(len(split_data)-1):
            temp = ''
            for m in range(3):
                temp += split_data[n+m-1]  # from n-1 to n, n+1
            n_grams_data.append(temp)
        return n_grams_data


class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()])


class NameNgrams(Document):
    def features(self):
        name = self.data
        return [name[0].lower()+name[1].lower(), name[-1].lower()]


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)


class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def split_names_corpus(self, document_class=NameNgrams):
        """Split the names corpus into training and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return names[:6000], names[6000:]

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return blogs[:3000], blogs[3000:]

    def test_blogs_bag(self):
        """Classify blog authors using bag-of-words"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        #classifier.load("model")
        #classifier.save("model")
        #classifier.load("model_Bernoulli")
        #classifier.save("model_Bernoulli")
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)