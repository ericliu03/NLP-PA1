# -*- mode: Python; coding: utf-8 -*-

from math import log
from classifier import Classifier

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model={}, classify_type="multinomial"):
        self.parameters = [{}]
        self.labels = []
        self.type = classify_type
        self.possibilities_of_features = [{}]
        self.possibilities_of_classes = {}
        self.sum_number_of_instances = 0
        self.number_of_instances = []
        self.number_of_words_in_one_class = []

    def get_model(self):
        return [self.possibilities_of_classes, self.possibilities_of_features, self.number_of_words_in_one_class,
                self.parameters, self.labels, self.type, self.number_of_instances]

    def set_model(self, model):
        [self.possibilities_of_classes, self.possibilities_of_features,
         self.number_of_words_in_one_class, self.parameters, self.labels, self.type, self.number_of_instances] = model

    model = property(get_model, set_model)

    def train(self, instances):
        """Remember the labels associated with the features of instances."""
        for instance in instances:
            self.sum_number_of_instances += 1
            # to find which class this instance is, if it is a new one, add to labels[]
            if instance.label in self.labels:
                for n in range(len(self.labels)):
                    if self.labels[n] == instance.label:
                        index = n
                        break
            else:
                self.labels.append(instance.label)  # add the new class's label
                self.parameters.append({})  # add a new variable for storing new class's parameters
                self.number_of_words_in_one_class.append(0)
                self.number_of_instances.append(0)
                index = len(self.labels)-1

            # to count the number of instances
            self.number_of_instances[index] += 1

            for feature in instance.features():
                if feature in self.parameters[index]:
                    self.parameters[index][feature] += 1
                else:
                    self.parameters[index][feature] = 1
                # to count how many words are there in the training documents
                self.number_of_words_in_one_class[index] += 1

        if self.type == "multinomial":
            self.get_class_probability('m')
        else:
            self.get_class_probability('b')

    def get_class_probability(self, classify_type):
        for n in range(len(self.labels)):
            self.possibilities_of_classes[n] = float(self.number_of_instances[n])/self.sum_number_of_instances
            for (feature, number) in self.parameters[n].iteritems():
                if len(self.possibilities_of_features) <= n:
                    self.possibilities_of_features.append({})
                if classify_type == 'm':
                    self.possibilities_of_features[n][feature] = float(number+1)/(self.number_of_words_in_one_class[n]
                                                                                  + len(self.parameters[n]))
                else:
                    self.possibilities_of_features[n][feature] = float(number+1)/(self.number_of_instances[n]+2)



    def classify(self, instance):
        if self.type == "multinomial":
            return self.choose_classifier(instance, 'm')
        else:
            return self.choose_classifier(instance, 'b')

    def choose_classifier(self, instance, classify_type):
        """Classify an instance using the features seen during training."""
        possibilities = []
        features = instance.features()
        max_pos = [0, -float('Inf')]

        for n in range(len(self.possibilities_of_features)):
            possibilities.append(log(self.possibilities_of_classes[n]))
            if classify_type == 'm':
                for feature in instance.features():
                    if feature in self.possibilities_of_features[n]:
                        pass
                    else:
                        self.possibilities_of_features[n][feature] = 1.0/(self.number_of_words_in_one_class[n]
                                                                          + len(self.parameters[n]))
                    possibilities[n] += log(self.possibilities_of_features[n][feature])
            else :
                for iter_item in self.possibilities_of_features[n].iteritems():
                    if iter_item[0] in features:
                        possibilities[n] += log(iter_item[1])
                    else:
                        possibilities[n] += log((1-iter_item[1]))
                for word in features:
                    if not word in self.possibilities_of_features[n]:
                        possibilities[n] += log(1.0/(self.number_of_instances[n]+2))
            if max_pos[1] < possibilities[n]:
                max_pos[0] = n
                max_pos[1] = possibilities[n]

        return self.labels[max_pos[0]]




