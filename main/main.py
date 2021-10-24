'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-21 14:53:27
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 08:51:45
 # @ Description: @Python :: main
 '''
print("Starting ...")
import time
import pandas as pd
import numpy as np
from features_generator import Features_Generator
from text_preprocessing import Text_Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def test():
    # Load the original datasets.
    print("\n>>> Loading datasets...")
    training_data = pd.read_csv('../data/original_train.csv')
    testing_data = pd.read_csv('../data/original_test.csv')
    (train_rows, train_cols), (test_rows, test_cols) = training_data.shape, testing_data.shape
    print("\nTrain Data:")
    print(training_data.head())
    print(f"{train_rows} rows x {train_cols} columns")
    print("\nTest Data:")
    print(testing_data.head())
    print(f"{test_rows} rows x {test_cols} columns")
    # Preprocess all texts stored in both datasets
    print("\n>>> Preprocessing tweets...")
    text_preprocessor = Text_Preprocessor()
    training_data.TweetText = training_data.TweetText.apply(text_preprocessor.preprocessing)
    testing_data.TweetText = testing_data.TweetText.apply(text_preprocessor.preprocessing)
    (train_rows, train_cols), (test_rows, test_cols) = training_data.shape, testing_data.shape
    print("T\nrain Data:")
    print(training_data.head())
    print(f"{train_rows} rows x {train_cols} columns")
    print("\nTest Data:")
    print(testing_data.head())
    print(f"{test_rows} rows x {test_cols} columns")
    # Extract all possible features
    print("\n>>> Extracting features...")
    features_generator = Features_Generator()
    training_data, testing_data = features_generator.fit(training_data, testing_data, 'TweetText')
    (train_rows, train_cols), (test_rows, test_cols) = training_data.shape, testing_data.shape
    print("\nTrain Data:")
    print(training_data.head())
    print(f"{train_rows} rows x {train_cols} columns")
    print("\nTest Data:")
    print(testing_data.head())
    print(f"{test_rows} rows x {test_cols} columns")
    # Build the prediction model.
    print("\n>>> Building prediction model...")
    # - Define features and the target variables.
    # NOTE: Since we are attempting to build a binary
    # classification model, we need to replace labels with 0 and 1.
    y = training_data.Label.apply(lambda x: 0 if x == 'Politics' else 1)
    X = training_data.drop(['TweetId', 'TweetText', 'Label'], axis=1)
    # Split the train dataset into training and testing
    # data using sklearn.model_selection.train_test_split
    # function such testing data represent 20% of the whole
    # data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
    # Declare the model.
    # In this first try we are trying to use Logistic
    # Regression algorithms built in scikit-learn.
    model = LogisticRegression()
    # Fitting the model with new traning data (80% of the
    # whole data).
    model.fit(X_train, y_train)
    # Printing the score obtained by our model (accuracy).
    score = model.score(X_train, y_train)
    print("\nX_train, y_train score is:", score)
    # Predicting test data to get a conclusion about
    # how much our model is accurate.
    y_pred = model.predict(X_test)
    # Display accuracy result using classification report.
    report = classification_report(y_test, y_pred, target_names=("Politics", "Sports"))
    print("\n##### Classification Report ###########################################\n")
    print(report)
    print("\n#######################################################################\n")
    # Start predict labels of our main testing data.
    print(">>> Predicting testing data Label...")
    # Defining a dataframe to store our final result
    # with only TweetId and Label attributes.
    result = pd.DataFrame(testing_data.TweetId, columns=['TweetId'])
    # Prepare testing data to use in prediction by
    # removing TweetId and TweetText attributes.
    X_testing = testing_data.drop(['TweetId', 'TweetText'], axis=1)
    # Predicting main testing data.
    y_testing_pred = model.predict(X_testing)
    # NOTE: The model we build do not predict directly
    # labels, but it give binary result. Hence we need to
    # reverse what we did after we have defined the taret
    # variable (replace the binary result by the correspondent
    # label [0 for 'Politics' and 1 for 'Sports']).
    result['Label'] = np.array(['Politics' if y_testing_predected == 0 else 'Sports' for y_testing_predected in y_testing_pred])
    # Fianly we save the result to a csv file.
    print("\n>>> Done!")
    result.set_index('TweetId', inplace=True)
    print("\nFinal result:")
    print(result)
    result.to_csv('../data/submission.csv')
    print("\nThe result is saved to data/submission.csv")

def main():
    start_time = time.time()
    ########################################################

    test()

    ########################################################
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {execution_time//60} minutes and {round(execution_time%60, 3)} seconds.")

if __name__ == '__main__': main()
