'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-22 12:21:19
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 20:17:02
 # @ Description:
 '''

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

class Features_Generator:

    def __init__(self) -> None:
        self.train_data = None
        self.test_data = None
        self.text_label = None
        self.minimum = 1
        self.maximum = 1
    
    def clear(self):
        self.train_data = None
        self.test_data = None

    def count_train_words(self):
        words_count = {}
        for text_words in self.train_data[self.text_label]:
            for word in text_words:
                if word in words_count.keys():
                    words_count[word] += 1
                else:
                    words_count[word] = 1
        return words_count
    
    def count_test_words(self):
        words_count = {}
        for text_words in self.test_data[self.text_label]:
            for word in text_words:
                if word in words_count.keys():
                    words_count[word] += 1
                else:
                    words_count[word] = 1
        return words_count

    def get_features_names(self):
        # Initialize features_names with an empty set.
        features_names = set()
        # When minimum and maximum are default, all words
        # should be considered as features in train data.
        if self.minimum == 1 and self.maximum == 1:
            for text_words in self.train_data[self.text_label]:
                    for word in text_words:
                        features_names.add(word)
        # Otherwise, some words can't be considered as a feature,
        # either because it appeared only in less than self.minimum
        # texts or because it appeared in more than self.maximum % of
        # all texts.
        else :
            _min = self.minimum
            _max = round(self.maximum*len(self.train_data))
            train_words_count = self.count_words('train')
            for word, count in train_words_count.itema():
                if count < _min or count > _max:
                    features_names.add(word)
        # For test data, it's prefered to not ignore too much words.
        # For that, we will try at first to keep all words as features. 
        for text_words in self.test_data[self.text_label]:
            for word in text_words:
                features_names.add(word)
        return features_names
            

    def generate_features(self):
        features = self.get_features_names()
        train_len, test_len = len(self.train_data), len(self.test_data)
        print("Initializing features columns with 0 value...")
        for feature in features:
            self.train_data[feature] = np.zeros(train_len)
            self.test_data[feature] = np.zeros(test_len)
        print("Filling new train dataset...")
        for index in self.train_data.index:
            for feature in self.train_data[self.text_label][index]:
                self.train_data[feature].iloc[index] = 1
        print("Filling new test dataset...")
        for index in self.test_data.index:
            for feature in self.test_data[self.text_label][index]:
                self.test_data[feature].iloc[index] = 1

        try: return self.train_data, self.test_data
        finally: self.clear()

    def fit(self, train:pd.DataFrame, test:pd.DataFrame, text_label:str, f_max:float=1, f_min:int=1):
        self.train_data = train
        self.test_data = test
        self.text_label = text_label
        self.maximum = f_max
        self.minimum = f_min
        return self.generate_features()