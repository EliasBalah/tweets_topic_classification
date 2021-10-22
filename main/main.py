'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-21 14:53:27
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 08:51:45
 # @ Description: @Python :: main
 '''

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

    """
                    TweetId     Label                                          TweetText                              PreprocessedTweetText
    0     304271250237304833  Politics  '#SecKerry: The value of the @StateDept and @U...  #seckerry value measure dollar term deepest am...
    3     304366580664528896    Sports  'RT @chelscanlan: At Nitro Circus at #AlbertPa...  nitro circus #albertpark #theymakeitlooksoeasy...
    4     296770931098009601    Sports  '@cricketfox Always a good thing. Thanks for t...                  always good thing thanks feedback
    ...                  ...       ...                                                ...                                                ...
    6520  296675082267410433  Politics  'Photo: PM has laid a wreath at Martyrs Monume...  photo pm laid wreath martyr monument algiers #...
    6521  306677536195231746    Sports  'The secret of the Chennai pitch - crumbling o...  secret chennai pitch crumble edge solid middle...
    6522  306451295307431937    Sports            @alinabhutto he isn't on Twitter either                                     twitter either
    6523  306088574221176832    Sports  'Which England player would you take out to di...       england player would take dinner feature amp
    6524  277090953242759169  Politics  'Dmitry #Medvedev expressed condolences to the...  dmitry #medvedev express condolence family fri...

    [6525 rows x 4 columns]
    Executed in 138.762 seconds.
    """


if __name__ == '__main__': main()