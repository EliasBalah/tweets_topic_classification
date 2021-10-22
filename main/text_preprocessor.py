import pandas as pd
import numpy as np
# For dealing with regular expressions
import re
# For text preprocessing
import nltk
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
# For utils
import time

# Handle links
def replace_links(text: str, replace_link_by='web_link') -> str:
    # replace_by is the argument used to replace
    # links with: if it's an empty string, then links
    # will be removed from the text [default value].
    text = re.sub('(https?://\S+)', replace_link_by, text)
    return text

# Handle usernames
def replace_usernames(text: str, replace_username_by='') -> str:
    # replace_by is the argument used to replace a
    # username with: if it's an empty string, then usernames
    # will be removed from the text [default value].
    if replace_username_by == '':
        text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', replace_username_by, text)
    text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', replace_username_by, text)
    return text

# Handle ponctuations & numbers
def replace_punctuations_nums(text: str, replace_num_by='', replace_punct_by=' ') -> str:
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
def clean_text(text: str, replace_link_by='', replace_username_by='', replace_num_by='', replace_punct_by=' ') -> list:
    # clean_text function should clean links, usernames,
    # numbers and punctuations from the text, and return
    # a list of all words in lower cases.
    text = replace_links(text, replace_link_by)
    text = replace_usernames(text, replace_username_by)
    text = replace_punctuations_nums(text, replace_num_by, replace_punct_by)
    text = text.lower()
    # NOTE: In our case we are dealing with tweets, means that
    # texts may contain hashtags and they needs be left as
    # they are. For that, we use nltk.tokenize.TweetTokenizser
    # instead of nltk.tokenize.word_tokenise
    text = TweetTokenizer().tokenize(text)
    return text

# Preprocessing
def preprocessing(text:str, bigrams=False) -> str:
    # For preprocessing the text, we need to remove all
    # stop words and replace other words by its root (lemma).
    # for that purpose, we take the advantage ofthe NLTK
    # library.
    text = clean_text(text)
    # Load stop words from nltk.corpus
    # NOTE: nltk.corpus.stopwords is prefered to be a set
    # for more efficiency.
    try:
    # Using try block avoids exceptions may be raised if
    # stopwords list is not downloaded.
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    text_words = [word for word in text if word not in stop_words]
    # For rooting our text' words, there are many ways available
    # on NLTK library. For many reasons: text ambiguity, word density
    # reduction, preparing the accurate features for training, memory
    # saving and computational cost, we prefere to use the
    # lemmatizer WordNetLemmatizer.
    # WordNetLemmatizer lemmatizer relies on the lexical database
    # wordnet that is downloadable to nltk library.
    try:
        word_lemmatizer = WordNetLemmatizer()
    except:
        nltk.download('wordnet')
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
    try:
        text_words = [word_lemmatizer.lemmatize(word, get_POS_tag(word)) if '#' not in word else word for word in text_words]
    except:
        nltk.download('averaged_perceptron_tagger')
        text_words = [word_lemmatizer.lemmatize(word, get_POS_tag(word)) if '#' not in word else word for word in text_words]

    # if bigrams: 
    #     text_words = text_words + [text_words[i] + '_' + text_words[i+1] for i in range(len(text_words)-1)]
        
    text = ' '.join(text_words)

    return text

def main(): ################################################
    start_time = time.time()
    ########################################################
    text = "#SecKerry: The value of the @StateDept and @USAID is measured, not in dollars, but in terms of our deepest American values."
    print("Testing...!")
    print("Text to be preprocessed:", text)
    print("Text after preprocessing:", preprocessing(text))
    ########################################################
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Executed in {round(execution_time,3)} seconds.")
    ########################################################

if __name__ == '__main__': main()