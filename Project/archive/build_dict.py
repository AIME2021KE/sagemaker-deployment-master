import numpy as np

from collections import Counter

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    #NOTE 12/27/2021 KAE: we note that this is kind of a combination of previous Sentiment projects of preprocess_data and BoW processing
    # We assume we'll not need to cache the results (for now....)
    #NOTE KAE: NOT quite what we were expecting -- is a list of lists, sentences are a list of individual words
#    print(type(data))
#    print(data[0:100])
    
    #Grab the initial list of raw words (review to words will process them from a stringe paragraph(s) to single words)
    # NOTE: https://realpython.com/python-counter/
# A dict storing the words that appear in the reviews along with how often they occur
#    word_count = {} 
    word_count = Counter()
    for review in data:
        word_count.update(Counter(review))
    
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    
#    sorted_words = None
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict