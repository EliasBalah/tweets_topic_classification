import time
import pandas as pd
from text_preprocessor import preprocessing



def main(): ################################################
    start_time = time.time()
    ########################################################
    original_training_data = pd.read_csv('../data/original_train.csv')
    original_training_data['PreprocessedTweetText'] = original_training_data.TweetText.apply(preprocessing)
    print(original_training_data)
    ########################################################
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {round(execution_time,3)} seconds.")
    ########################################################


if __name__ == '__main__': main()