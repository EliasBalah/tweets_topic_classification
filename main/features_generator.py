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
    
    def clear(self):
        self.train_data = None
        self.test_data = None

    def count_words(self, data_str:str='train'):
        if data_str == 'test':
            words_count = {}
            for text_words in self.test_data[self.text_label]:
                for word in text_words:
                    if word in words_count.keys():
                        words_count[word] += 1
                    else:
                        words_count[word] = 1
            return words_count
        else:
            words_count = {}
            for text_words in self.train_data[self.text_label]:
                for word in text_words:
                    if word in words_count.keys():
                        words_count[word] += 1
                    else:
                        words_count[word] = 1
            return words_count

    def get_train_test_full_features(self):

        train_features = self.count_words('train')
        test_features = self.count_words('test')

        for train_word in train_features.keys():
            if train_word not in test_features.keys():
                test_features[train_word] = 0

        for testing_word in test_features.keys():
            if testing_word not in train_features.keys():
                train_features[testing_word] = 0
        
        return train_features, test_features

    def get_features_names(self):
        print("Getting features names...")
        features_names = set()

        for text_words in self.train_data[self.text_label]:
                for word in text_words:
                    features_names.add(word)

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

    def fit(self, train:pd.DataFrame, test:pd.DataFrame, text_label:str):
        self.train_data = train
        self.test_data = test
        self.text_label = text_label
        return self.generate_features()