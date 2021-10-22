'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-21 14:53:27
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 08:51:45
 # @ Description: @Python :: main
 '''

import time
import pandas as pd
import numpy as np
from features_generator import Features_Generator
from text_preprocessing import Text_Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def test():
    print("\n>>> Loading datasets...")

    training_data = pd.read_csv('../data/original_train.csv')
    testing_data = pd.read_csv('../data/original_test.csv')

    print("\n>>> Preprocessing tweets...")

    text_preprocessor = Text_Preprocessor()
    training_data.TweetText = training_data.TweetText.apply(text_preprocessor.preprocessing)
    testing_data.TweetText = testing_data.TweetText.apply(text_preprocessor.preprocessing)

    print("\n>>> Extracting features...")

    features_generator = Features_Generator()
    training_data, testing_data = features_generator.fit(training_data, testing_data, 'TweetText')

    print("\n>>> Building prediction model...")

    y = training_data.Label.apply(lambda x: 0 if x == 'Politics' else 1)
    X = training_data.drop(['TweetId', 'TweetText', 'Label'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Logistic Regression finished!")
    score = model.score(X_train, y_train)
    print("X_train, y_train score is:", score)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=("Politics", "Sports"))
    print("\n##### Classification Report ###########################################\n")
    print(report)
    print("\n#######################################################################\n")

    print(">>> Predicting testing data Label...")
    result = pd.DataFrame(testing_data.TweetId, columns=['TweetId'])

    X_testing = testing_data.drop(['TweetId', 'TweetText'], axis=1)
    y_testing_pred = model.predict(X_testing)

    result['Label'] = np.array(['Politics' if y_testing_predected == 0 else 'Sports' for y_testing_predected in y_testing_pred])

    result.to_csv('../data/sample_submission.csv')





def main():
    start_time = time.time()
    ########################################################
    test()
    
    # training_data = pd.read_csv('../data/original_train.csv')
    # y = training_data.Label.apply(lambda x: 0 if x == 'Politics' else 1)
    # X = training_data.drop(['Label'], axis=1)

    # print("X:\n", X)
    # print("y:\n", y)


    ########################################################
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {execution_time//60} min and {round(execution_time%60, 3)}.")

if __name__ == '__main__': main()
