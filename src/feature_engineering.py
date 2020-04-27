import numpy as np
import pickle
import string
import re
import unidecode
import slopos
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO: how many times the user has replied in a row.
# TODO: Number of replies in last N messages.

def get_markov_model(target):
    """
    Get Markov model for transitions between target values.

    Args:
        target (numpy.ndarray): state transitions to model.
    
    Returns:
        (dict): Dictionary mapping states to transition probabilities.
    """
    
    # Get unique states (sorted).
    states = np.unique(target)
    
    # Initialize dictionary mapping states to transition probabilities.
    res = {state : np.zeros(len(states), dtype=float) for state in states}

    # Go over path and accumulate transitions.
    for idx in np.arange(1, len(target)):
        state_prev = target[idx-1]
        state_curr = target[idx]
        res[state_prev][state_curr] += 1

    # Normalize to get probabilities.
    for key in res.keys():
        if np.sum(res[key]) > 0:
            res[key] /= np.sum(res[key])
    
    # Return resulting dictionary.
    return res


def get_conditional_probabilities(target, n_look_back):
    """
    Get conditional probabilities of state if previous number of specified
    are known. Compute the probabilities and return in form of dictionary.
    
    Args:
        target (numpy.ndarray): state transitions to model.
    
    Returns:
        (dict): dictionary mapping sequences of previous n
        class values to probabilities of current class value.
    """

    # Get unique states (sorted).
    states = np.unique(target)
    
    # Initialize dictionary for mapping observations to probabilties.
    res = dict()
    
    # Go over path and accumulate observations:
    for idx in np.arange(n_look_back, len(target)):
        patt = str(target[idx-n_look_back:idx])
        state_curr = target[idx]
        if patt in res:
            res[patt][state_curr] += 1
        else:
            res[patt] = np.zeros(len(states))
    
    # Normalize to get probabilities.
    for key in res.keys():
        if np.sum(res[key]) > 0:
            res[key] /= np.sum(res[key])
   
    # Return resulting dictionary.
    return res


def get_bow(corpus):
    """
    Get list of (sensible) words occuring in specified corpus.

    Args:
        corpus (list): Corpus of messages where each message is
        represented as an entry in a list.

    Returns:
        (list): List containing the sensible words found in the
        messages.
    """
    bow = set()
    WORD_LEN_MAX = 15
    for message in corpus:
        words = map(lambda x: x.lower(), message.split(' '))
        bow.update(filter(lambda x: len(x) < WORD_LEN_MAX, words)) 
    return list(bow)


def translate_nstd(message, dictionary):
    """
    Translate non-standard Slovene to standard slovene using data parsed
    from dataset obtained from JANES project repository.
    
    Args:
        message (str): Message to translate.
        dictionary (dict): Dictionary mapping non-standard Slovene words
        to standard Slovene translations.

    Returns:
        (str): Message translated to standard slovene.

    """

    # Initialize list for storing results.
    res = []

    # Go over words. Remove punctuation first.
    for w in message.translate(str.maketrans('', '', string.punctuation)).split(' '):

        # Decode unicode and make lower-case.
        dec = unidecode.unidecode(w).lower()

        # Try to translate next word.
        if dec in dictionary:
            res.append(dictionary[dec])
        else:
            res.append(dec)
    
    # Join words.
    return ' '.join(res)


def get_pos_simple(message, dictionary):
    translated = translate_nstd(message, dictionary)
    tagged = slopos.tag(translated)
    return ' '.join(map(lambda x: x[1][:2], tagged))




def eval_bow(message, bow):
    """
    Get bag-of-words feature vector for message using specified list of words.
    
    Args:
        message (str): message to convert to vector of features.
        bow (list): List representing the bag-of-words.
    
    Returns (numpy.ndarray): bag-of-words vector.
    """

    # Evaluate message against bag of words model.
    feat_vec = np.zeros(len(bow), dtype=int)
    message_words = message.split(' ')
    for idx, word in enumerate(bow):
        if word in map(lambda x: x.lower(), message_words):
            feat_vec[idx] += 1
    return feat_vec


def num_curse_words(message, curse_words):
    """
    Compute number of curse words in message using provided list.

    Args:
        message (str): The message to check for curse words.
        curse_words (list): List of curse words.
   
    Returns:
        (numpy.ndarray): Array containing number of curse words found in message.
    """
    
    # Initialize counter.
    count = 0

    # Go over list of curse words and try to locate in message.
    for curse_word in curse_words:
        # If lenght longer than three letters, search for substring.
        # Else require space before, after or on both sides.
        if len(curse_word) > 3:
            patt = re.compile(curse_word)
        else:
            patt = re.compile(' ' + curse_word + ' |' + curse_word + ' ' + '| ' + curse_word)
        if patt.search(unidecode.unidecode(message).lower()):
            count += 1
    return np.array([count])


def num_given_names(message, given_names):
    """
    Compute number of given names in message using provided list.

    Args:
        message (str): The message to check for given names.
        given_names (list): List of given names.
    
    Returns:
        (numpy.ndarray): Array containing number of given names found in message.
    """

    # Initialize counter.
    count = 0

    # Go over list of given names and try to locate in message.
    for name in given_names:
        # If lenght longer than three letters, search for substring.
        # Else require space before, after or on both sides.
        if len(name) > 4:
            patt = re.compile(name)
        else:
            patt = re.compile('\b' + name + '\b')
        if patt.search(unidecode.unidecode(message).lower()):
            count += 1
    return np.array([count])


def num_chat_names(message, chat_names):
    """
    Compute number of chat names in message using provided list.

    Args:
        message (str): The message to check for chat names.
        given_names (list): List of chat names.
    
    Returns:
        (numpy.ndarray): Array containing number of chat names found in message.
    """
    
    # Initialize counter.
    count = 0

    # Go over list of chat names and try to locate in message.
    for name in chat_names:
        # If lenght longer than three letters, search for substring.
        # Else require space before, after or on both sides.
        if len(name) > 3:
            patt = re.compile(name)
        else:
            patt = re.compile(' ' + name + ' |' + name + ' ' + '| ' + name)
        if patt.search(unidecode.unidecode(message).lower()):
            count += 1
    return np.array([count])


def num_story_names(message, story_names):
    """
    Compute number of names in stories in message using provided list.

    Args:
        message (str): The message to check for chat names.
        given_names (list): List of names in stories.
    
    Returns:
        (numpy.ndarray): Array containing number of names in stories found in message.
    """
    
    # Initialize counter.
    count = 0

    # Go over list of names in stories and try to locate in message.
    for name in story_names:
        # If lenght longer than three letters, search for substring.
        # Else require space before, after or on both sides.
        if len(name) > 3:
            patt = re.compile(name)
        else:
            patt = re.compile(' ' + name + ' |' + name + ' ' + '| ' + name)
        if patt.search(unidecode.unidecode(message).lower()):
            count += 1
    return np.array([count])


def num_repeated_letters(message):
    """
    Compute number of repeated letters in message.

    Args:
        message (str): The message to check for repeated letters.
    
    Returns:
        (numpy.ndarray): Array containing number of repeated letters in message.
    """
    
    # Count number of pairs of repeated letters.
    count = 0
    for idx in range(len(message)-1):
        if message[idx].lower() == message[idx+1].lower():
            count += 1
    return np.array([count])


def num_clues(message, clue_words):
    """
    Compute number of clue words in message using provided list.

    Args:
        message (str): The message to check for clue words.
        given_names (list): List of clue words.
    
    Returns:
        (numpy.ndarray): Array containing number of clue words found in message.
    """
    
    # Initialize counter.
    count = 0

    # Go over list of names in stories and try to locate in message.
    for word in clue_words:
        # If lenght longer than three letters, search for substring.
        # Else require space before, after or on both sides.
        if len(word) > 3:
            patt = re.compile(word)
        else:
            patt = re.compile(' ' + word + ' |' + word + ' ' + '| ' + word)
        if patt.search(unidecode.unidecode(message).lower()):
            count += 1
    return np.array([count])


def get_general_features(message):
    """
    Compute basic features of message.

    Args:
        message (str): The message from which to extract features.
    
    Returns:
        (numpy.ndarray): Constructed features vector.
    """

    # Allocate resulting vector.
    res_vec = np.empty(9, dtype=float)

    # Number of words in text.
    res_vec[0] = len(message.split(' '))
    
    # Words lengths.
    word_lengths = list(map(len, message.split(' ')))
    res_vec[1] = max(word_lengths)
    res_vec[2] = min(word_lengths)
    res_vec[3] = sum(word_lengths)/len(word_lengths)

    # Number of digits in text.
    res_vec[4] = sum((c.isdigit() for c in message))
    
    # Number of punctuation marks in text.
    res_vec[5] = sum(c in string.punctuation for c in message)

    # Number of capital letters in text.
    res_vec[6] = sum(c.isupper() for c in message)

    # Starts with capital?
    res_vec[7] = 1 if len(message) > 0 and message[0].isupper() else 0
    
    # Ends with period?
    res_vec[8] = 1 if len(message) > 0 and message[-1] == '.' else 0
    
    # Return resulting vector.
    return res_vec


def get_repl_processor():
    """
    Get function for extracting features from messages for use in REPL.

    Returns:
        (function): Function that takes a message and constructs its features.
    """
    
    # Parse vectorizers and bag-of-words list.
    with open('../data/cached/repl/count_vectorizer.p', 'rb') as f1, \
            open('../data/cached/repl/pos_tfidf_vectorizer.p', 'rb') as f2, \
            open('../data/cached/repl/bow.p', 'rb') as f3:
        vectorizer1 = pickle.load(f1)
        vectorizer2 = pickle.load(f2)
        bow = pickle.load(f3)
    
    # Parse data dictionary.
    with open('../data/data-processed/data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Load dictionary for non-standard Slovene.
    with open('../data/slo_nstd_dict.p', 'rb') as f:
        slo_nstd_dict = pickle.load(f)


    def message_features(message):
        """
        Construct message features (for use in REPL).
        
        Args:
            message (str): Message for which to construct the features

        Returns:
            (numpy.ndarray): Resulting feature vector.

        """

        # Decode unicode and fix missing spaces after punctuation.
        message = re.sub(r'(?<=[.,])(?=[^\s])', r' ', message)

        # Compute message features.
        general_features = get_general_features(message)
        curse_words = num_curse_words(message, data['curse-words'])
        repeated_letters = num_repeated_letters(message)
        clue_words = num_clues(message, data['clue-words'])
        given_names = num_given_names(message, data['names'])
        chat_names = num_chat_names(message, data['chat-names'])
        story_names = num_story_names(message, data['story-names'])
        bow_features = eval_bow(message, bow)
        
        # Compute (simplified) POS tags.
        pos_tags = get_pos_simple(message, slo_nstd_dict)
        
        # Compute bigram counts and pos tags tf-idf scores.
        bigram_count = vectorizer1.transform([message]).toarray()[0]
        pos_tfidf = vectorizer2.transform([message]).toarray()[0]

        # Construct features vector.
        feat_vec = np.hstack((general_features, curse_words, repeated_letters, clue_words, 
            given_names, chat_names, story_names, bow_features, bigram_count, pos_tfidf))
        
        # Return constructed feature.
        return feat_vec
    
    # Return function for creating features.
    return message_features


def construct_features(data, category):
    """
    Extract various features from messages.

    Args:
        data (dict): Dictionary representing the preprocessed dataset.
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'
    
    Returns:
        (tuple): Extracted features and vector of target values.
    """

    # If computing features for book relevance prediction.
    if category == 'book-relevance':
        
        # Get data.
        data_category = data['book-relevance']
        messages_list = data_category['x'].values.astype(str)
        target_vals = data_category['y'].values.astype(int)

        # Initialize matrix for features.
        data_mat_res = None
        first = True

        # Load dictionary for non-standard Slovene.
        with open('../data/slo_nstd_dict.p', 'rb') as f:
            slo_nstd_dict = pickle.load(f)
        
        # Get bag-of-words list.
        bow = get_bow(data['book-relevance']['x'].values.astype(str))

        # Save bag-of-words list.
        with open('../data/cached/repl/bow.p', 'wb') as f:
            pickle.dump(bow, f, pickle.HIGHEST_PROTOCOL)
        
        # Get list for storing POS tagged messages (simplified).
        messages_pos = []
        
        # Set flag for getting feature subset lengths using first message.
        get_feature_subset_lengths = True
        feature_subset_lengths = None

        # Go over messages.
        for idx, message in enumerate(messages_list):

            # Decode unicode and fix missing spaces after punctuation.
            message = re.sub(r'(?<=[.,])(?=[^\s])', r' ', message)

            # Compute message features.
            general_features = get_general_features(message)
            curse_words = num_curse_words(message, data['curse-words'])
            repeated_letters = num_repeated_letters(message)
            clue_words = num_clues(message, data['clue-words'])
            given_names = num_given_names(message, data['names'])
            chat_names = num_chat_names(message, data['chat-names'])
            story_names = num_story_names(message, data['story-names'])
            bow_features = eval_bow(message, bow)


            # Append POS tagged (simplified) message to list.
            messages_pos.append(get_pos_simple(message, slo_nstd_dict))

            # Construct features vector.
            feat_vec = np.hstack((general_features, curse_words, repeated_letters, clue_words, given_names, chat_names, story_names, bow_features))
            
            # If getting feature subset lengths (first message).
            if get_feature_subset_lengths:
                feature_subset_lengths = [len(general_features), len(curse_words) + len(repeated_letters) +\
                        len(clue_words) + len(given_names) + len(chat_names) + len(story_names), len(bow_features)]
                get_feature_subset_lengths = False

            # Add to matrix of features.
            if first:
                data_mat_res = feat_vec
                first = False
            else:
                data_mat_res = np.vstack((data_mat_res, feat_vec))
            print("Processed message {0}/{1}.".format(idx+1, len(messages_list)))
        
        # Count bigrams.
        vectorizer1 = CountVectorizer(min_df=2, ngram_range=(2, 2))
        bigram_count = vectorizer1.fit_transform(messages_list).toarray()
        data_mat_res = np.hstack((data_mat_res, bigram_count))
        feature_subset_lengths.append(bigram_count.shape[1])
        with open('../data/cached/repl/count_vectorizer.p', 'wb') as f:
            pickle.dump(vectorizer1, f, pickle.HIGHEST_PROTOCOL)
        
        # Get tf-idf results for POS tags (simplified).
        vectorizer2 = TfidfVectorizer()
        pos_tfidf = vectorizer2.fit_transform(messages_pos).toarray()
        data_mat_res = np.hstack((data_mat_res, pos_tfidf))
        feature_subset_lengths.append(pos_tfidf.shape[1])
        with open('../data/cached/repl/pos_tfidf_vectorizer.p', 'wb') as f:
            pickle.dump(vectorizer2, f, pickle.HIGHEST_PROTOCOL)

        # Save matrix of features and vector of target variables.
        np.save('../data/cached/data_book_relevance.npy', data_mat_res)
        np.save('../data/cached/target_book_relevance.npy', target_vals)

        # Save feature subset lengths.
        np.save('../data/cached/target_book_feature_subset_lengths.npy', feature_subset_lengths)
    elif category == 'type':
        # TODO
        pass
    elif category == 'category':
        # TODO
        pass
    elif category == 'category-broad':
        # TODO
        pass


if __name__ == '__main__':
    import os
    DATA_PATH = '../data/data-processed/data.pkl'
    if os.path.isfile(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        construct_features(data, 'book-relevance')
        print("Feature engineering successfully performed. You may now run the evaluate.py script.")
    else:
        print("Processed dataset not found. Run the parse.py script before running this script.")

