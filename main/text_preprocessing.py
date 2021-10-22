'''
 # @ Author: Ilias BALAH
 # @ Create Time: 2021-10-21 15:17:47
 # @ Modified by: Ilias BALAH
 # @ Modified time: 2021-10-22 13:44:24
 # @ Description: @Python :: Text Preprocessing 
 '''

# For dealing with regular expressions
import re
# For text preprocessing
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
# For utils
import time



class Text_Preprocessor:

    """
    NOTE: To use this preprocessor, it's mandatory to have already
    downloaded these data in NLTK:
    - To use nltk.wordnet.WordNetLemmatizer: >>> nltk.download('wordnet')
    - To use nltk.tokenize.TweetTokenizer:   >>> nltk.download('punkt')
    - To use nltk.pos_tag:                   >>> nltk.download('averaged_perceptron_tagger')
    - To access stopwords:                   >>> nltk.download('stopwords')
    - To use Open Multilingual Wordnet:      >>> nltk.download('omw')
    """

    def __init__(self, ignore_stopwords=True, pos_tag=False, bigrams=False, ) -> None:
        self.ignore_stopwords = ignore_stopwords
        self.pos_tag = pos_tag
        self.bigrams = bigrams

    # Handle links
    def replace_links(self, text: str, replace_link_by='web_link') -> str:
        # replace_by is the argument used to replace
        # links with: if it's an empty string, then links
        # will be removed from the text [default value].
        text = re.sub('(https?://\S+)', replace_link_by, text)
        return text

    # Handle usernames
    def replace_usernames(self, text: str, replace_username_by='') -> str:
        # replace_by is the argument used to replace a
        # username with: if it's an empty string, then usernames
        # will be removed from the text [default value].
        if replace_username_by == '':
            text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', replace_username_by, text)
        text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', replace_username_by, text)
        return text

    # Handle ponctuations & numbers
    def replace_punctuations_nums(self, text: str, replace_num_by='', replace_punct_by=' ') -> str:
        # replace_num_by is the argument used to replace
        # a number with: the default value is an empty string.
        # replace_punct_by is the argument used to replace
        # a punctuation with: the default value is, to avoid
        # words concatenation, a space character.
        punctuations = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
        text = re.sub('([' + punctuations + ']+)', replace_punct_by, text)
        text = re.sub('([0-9]+)', replace_num_by, text)
        # NOTE: Replacing a punctuation/number with a space/embty
        # character may cause multiple spaces. For that we need
        # to replace it by a single space.
        text = re.sub('(\s+)', ' ', text)
        return text

    # Handle text
    def clean_text(self, text: str, replace_link_by='', replace_username_by='', replace_num_by='', replace_punct_by=' ') -> list:
        # clean_text function should clean links, usernames,
        # numbers and punctuations from the text, and return
        # a list of all words in lower cases.
        text = self.replace_links(text, replace_link_by)
        text = self.replace_usernames(text, replace_username_by)
        text = self.replace_punctuations_nums(text, replace_num_by, replace_punct_by)
        text = text.lower()
        # NOTE: In our case we are dealing with tweets, means that
        # texts may contain hashtags and they needs be left as
        # they are. For that, we use nltk.tokenize.TweetTokenizser
        # instead of nltk.tokenize.word_tokenise
        text = TweetTokenizer().tokenize(text)
        return text

    # Preprocessing
    def preprocessing(self, text:str) -> list:
        # For preprocessing the text, we need to remove all
        # stop words and replace other words by its root (lemma).
        # for that purpose, we take the advantage ofthe NLTK
        # library.
        text = self.clean_text(text)
        # Load stop words from nltk.corpus
        # NOTE: nltk.corpus.stopwords is prefered to be a set
        # for more efficiency.
        stop_words = set(stopwords.words('english'))
        text_words = [word for word in text if word not in stop_words]
        # For rooting our text' words, there are many ways available
        # on NLTK library. For many reasons: text ambiguity, word density
        # reduction, preparing the accurate features for training, memory
        # saving and computational cost, we prefere to use the
        # lemmatizer WordNetLemmatizer.
        # NOTE: WordNetLemmatizer lemmatizer relies on the lexical database
        # wordnet that is downloadable to nltk library.
        word_lemmatizer = WordNetLemmatizer()
        # Normally, lemmatizing a word directly without any information about
        # its gramatical category (Part of speech), will not be usefull as
        # much as providing its POS. For that reason we need to provide the
        # lemmatizer, beside the word, its appropriate part of speech as a 
        # second argument.
        # Defining a function that gets the POS tag appropriate to
        # a word and acceptable by the Lemmatizer.
        def get_POS_tag(word):
            POS_tag = pos_tag([word])[0][1][0]
            POS_tag_list = {
                'V': wordnet.VERB,
                'N': wordnet.NOUN,
                'J': wordnet.ADJ,
                'R': wordnet.ADV
            }
            return POS_tag_list.get(POS_tag, wordnet.NOUN)
        # NOTE: In some texts (twwets) there some words (hashtags) starts
        # with '#'. Those words should not be lemmatized. In fact they are
        # probably the most significant word, and hence lemmatize them would
        # be a bad idea.
        if self.pos_tag:
            text_words = [word_lemmatizer.lemmatize(word, get_POS_tag(word)) if '#' not in word else word for word in text_words]
        else:
            text_words = [word_lemmatizer.lemmatize(word) if '#' not in word else word for word in text_words]
        # if self.bigrams: 
        #     text_words = text_words + [text_words[i] + '_' + text_words[i+1] for i in range(len(text_words)-1)]
        return text_words
            

def main():
    start_time = time.time()
    text = "#SecKerry: The value of the @StateDept and @USAID is measured, not in dollars, but in terms of our deepest American values."
    print("##### Testing...!")
    print("Text to be preprocessed:", text)
    preprocessor = Text_Preprocessor(pos_tag=True)
    text_words = preprocessor.preprocessing(text)
    print("Text after preprocessing:", ' '.join(text_words))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {round(execution_time,3)} seconds.")

if __name__ == '__main__': main()

