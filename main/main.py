'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-21 14:53:27
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 08:51:45
 # @ Description: @Python :: main
 '''

import time
import pandas as pd
from features_generator import Features_Generator
from text_preprocessing import Text_Preprocessor


def test():
    print("Loading datasets...!")
    training_data = pd.read_csv('../data/original_train.csv')
    testing_data = pd.read_csv('../data/original_test.csv')
    print("Preprocessing tweets...!")
    text_preprocessor = Text_Preprocessor()
    training_data.TweetText = training_data.TweetText.apply(text_preprocessor.preprocessing)
    testing_data.TweetText = testing_data.TweetText.apply(text_preprocessor.preprocessing)
    print("Extracting features...!")
    features_generator = Features_Generator()
    training_data, testing_data = features_generator.fit(training_data, testing_data, 'TweetText')
    print("Done!")
    training_data.to_csv('../data/corrected_train.csv')
    print("New train dataset is saved to: /data/corrected_train.csv")
    testing_data.to_csv('../data/corrected_test.csv')
    print("New train dataset is saved to: /data/corrected_test.csv")


def main():
    start_time = time.time()
    ########################################################
    test()
    ########################################################
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {round(execution_time,3)} seconds.")

if __name__ == '__main__': main()