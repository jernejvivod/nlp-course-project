import numpy as np
import pickle
import string
import re
import itertools
import unidecode
import slopos
from emoji import UNICODE_EMOJI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lemmagen
from lemmagen.lemmatizer import Lemmatizer


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
    for w in message.translate(str.maketrans('', '', string.punctuation)).split():
        
        # Remove emojis and any repeated characters.
        w_filt = ''.join(filter(lambda x: x not in UNICODE_EMOJI, ''.join(c[0] for c in itertools.groupby(w))))

        # Decode unicode and make lower-case.
        dec = unidecode.unidecode(w_filt).lower()

        # Try to translate next word.
        if dec in dictionary:
            res.append(dictionary[dec])
        else:
            res.append(dec)
    
    # Join words.
    return ' '.join(res)


def get_pos_simple(message, dictionary):
    """
    Translate non-standard Slovene, tag message and return POS tags.

    Args:
        message (str): The message to tag
        dictionary (dict): Dictionary mapping non-standard Slovene words
        to standard slovene words.
    """

    # Remove emojis and any repeated letters.
    message_filt = ''.join(filter(lambda x: x not in UNICODE_EMOJI and x not in string.punctuation, 
        ''.join(c[0] for c in itertools.groupby(message))))
    translated = translate_nstd(message_filt, dictionary)
    tagged = slopos.tag(translated)
    return ' '.join(map(lambda x: x[1][:2], tagged))


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


def num_messages_in_row(user_id, user_id_history_map, idx):
    """
    Get number of messages in a row that the user
    with specified user ID has made.

    Args:
        user_id (int): The user's ID
        user_id_history_map (dict): Dictionary mapping
        user ID's to arrays containing the value 1 at index where
        the message with same index was made by user with that user ID.

    Returns:
        (int): Number of messages posted in a row by that user.
    """

    # Initialize counter.
    count = 0

    # Go over values in array in reverse and return when
    # first zero value found.
    for el in user_id_history_map[user_id][idx::-1]:
        if el == 1:
            count += 1
        else:
            return count
    return np.array([count])


def num_messages_last_n(user_id, user_id_history_map, idx, n=20):
    """
    Get number of messages in the last n messages posted
    by user with specified user ID.

    Args:
        user_id (int): The user's ID
        user_id_history_map (dict): Dictionary mapping
        user ID's to arrays containing the value 1 at index where
        the message with same index was made by user with that user ID.
        n (int): Number of messages in the past to use.

    Returns:
        (int): Number of messages in the last n messages posted by that
        user.
    """
    return np.array([np.sum(user_id_history_map[user_id][max(idx-n, 0):idx+1])])


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
    res_vec[0] = len(message.split())
    
    # Words lengths.
    word_lengths = list(map(len, message.split()))
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
            open('../data/cached/repl/pos_tfidf_vectorizer.p', 'rb') as f2:
        vectorizer1 = pickle.load(f1)
        vectorizer2 = pickle.load(f2)
    
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
        messages_in_row_id = 0    # pre-set value
        messages_in_last_n_id = 5 # pre-set value
        
        # Compute (simplified) POS tags.
        pos_tags = get_pos_simple(message, slo_nstd_dict)
        
        # Compute bigram counts and pos tags tf-idf scores.
        bow_count = vectorizer1.transform([message]).toarray()[0]
        pos_tfidf = vectorizer2.transform([message]).toarray()[0]

        # Construct features vector.
        feat_vec = np.hstack((general_features, curse_words, repeated_letters, clue_words, 
            given_names, chat_names, story_names, messages_in_row_id, messages_in_last_n_id, 
            bow_count, pos_tfidf))
        
        # Return constructed feature.
        return feat_vec
    
    # Return function for creating features.
    return message_features


def construct_features(data, category, use_bow_features=True):
    """
    Extract various features from messages and format data
    for classification.

    Args:
        data (dict): Dictionary representing the preprocessed dataset.
        category (str): Specification of the type of prediction being made.
        Valid values are 'book-relevance', 'type', 'category' and 'category-broad'
    """
    
    # Get data.
    data_category = data[category]
    messages_list = data_category['x'].values.astype(str)
    target_vals = data_category['y'].values.astype(int)
    user_ids = data['user-ids']

    # Initialize dictionary of lists where each list contains a 1 at index corresponding
    # to a message if the poster's user ID equals to key.
    user_id_history_map = {user_id : np.zeros(len(messages_list), dtype=int) 
            for user_id in np.unique(user_ids)}

    # Initialize matrix for features.
    data_mat_res = None
    first = True

    # Initialize list for storing feature names (for feature scoring).
    feature_names = []

    # Load dictionary for non-standard Slovene.
    with open('../data/slo_nstd_dict.p', 'rb') as f:
        slo_nstd_dict = pickle.load(f)
    
    # Get list for storing POS tagged messages (simplified).
    messages_pos = []
    
    # Set flag for getting feature subset lengths using first message.
    get_feature_subset_lengths = True
    feature_subset_lengths = None
    
    # Add initial feature names to list.
    feature_names.extend(['general-feature-' + str(idx) 
        for idx in range(len(get_general_features(messages_list[0])))])
    feature_names.append('curse-words')
    feature_names.append('repeated-letters')
    feature_names.append('clue-words')
    feature_names.append('given-names')
    feature_names.append('chat-names')
    feature_names.append('story-names')
    feature_names.append('messages-in-a-row-id')
    feature_names.append('last-n-id')
    
    # Go over messages.
    for idx, (message, user_id) in enumerate(zip(messages_list[:10], user_ids[:10])):

        # Decode unicode and fix missing spaces after punctuation.
        message = re.sub(r'(?<=[.,])(?=[^\s])', r' ', message)

        # Set value in post history for user posting the current message.
        user_id_history_map[user_id][idx] = 1

        # Compute message features.
        general_features = get_general_features(message)
        curse_words = num_curse_words(message, data['curse-words'])
        repeated_letters = num_repeated_letters(message)
        clue_words = num_clues(message, data['clue-words'])
        given_names = num_given_names(message, data['names'])
        chat_names = num_chat_names(message, data['chat-names'])
        story_names = num_story_names(message, data['story-names'])
        messages_in_row_id = num_messages_in_row(user_id, user_id_history_map, idx)
        messages_in_last_n_id = num_messages_last_n(user_id, user_id_history_map, idx=idx, n=20)

        # Append POS tagged (simplified) message to list.
        messages_pos.append(get_pos_simple(message, slo_nstd_dict))

        # Construct features vector.
        feat_vec = np.hstack((general_features, curse_words, repeated_letters, clue_words, given_names, chat_names, story_names, messages_in_row_id, messages_in_last_n_id))
        
        # If getting feature subset lengths (first message).
        if get_feature_subset_lengths:
            feature_subset_lengths = [len(general_features), len(curse_words) + len(repeated_letters) +\
                    len(clue_words) + len(given_names) + len(chat_names) + len(story_names) + len(messages_in_row_id) + len(messages_in_last_n_id)]
            get_feature_subset_lengths = False

        # Add to matrix of features.
        if first:
            data_mat_res = feat_vec
            first = False
        else:
            data_mat_res = np.vstack((data_mat_res, feat_vec))
        print("Processed message {0}/{1}.".format(idx+1, len(messages_list)))
    
    # If using BOW(like) features, compute unigram and bigram counts and simplified POS tags counts.
    if use_bow_features:

        # Remove emoji and any punctuation.
        messages_list_filt = [''.join(filter(lambda x: x not in UNICODE_EMOJI and x not in string.punctuation, message)) for message in messages_list]

        # Remove repeated and lower-case characters.
        messages_list_filt = [''.join(c[0] for c in itertools.groupby(message.lower())) for message in messages_list_filt]
        
        # Initialize lemmatizer and lemmatize.
        lemmatizer = lemmagen.lemmatizer.Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
        messages_list_filt_lemm = [' '.join([lemmatizer.lemmatize(word) for word in message.split()]) for message in messages_list_filt]

        # Count unigrams and bigrams.
        vectorizer1 = CountVectorizer(min_df=2, ngram_range=(1, 2))
        bow_count = vectorizer1.fit_transform(messages_list_filt_lemm).toarray()
        import pdb
        pdb.set_trace()
        feature_names.extend(['bow-feature-' + str(idx) for idx in range(bow_count.shape[1])])
        data_mat_res = np.hstack((data_mat_res, bow_count))
        feature_subset_lengths.append(bow_count.shape[1])
        with open('../data/cached/repl/count_vectorizer.p', 'wb') as f:
            pickle.dump(vectorizer1, f, pickle.HIGHEST_PROTOCOL)

        # Get tf-idf results for POS tags (simplified).
        vectorizer2 = TfidfVectorizer()
        pos_tfidf = vectorizer2.fit_transform(messages_pos).toarray()
        feature_names.extend(['pos-feature-' + str(idx) for idx in range(pos_tfidf.shape[1])])
        data_mat_res = np.hstack((data_mat_res, pos_tfidf))
        feature_subset_lengths.append(pos_tfidf.shape[1])
        with open('../data/cached/repl/pos_tfidf_vectorizer.p', 'wb') as f:
            pickle.dump(vectorizer2, f, pickle.HIGHEST_PROTOCOL)
   
    # Save feature names (for scoring).
    with open('../data/cached/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    # Save matrix of features and vector of target variables.
    np.save('../data/cached/data_' + category.replace('-', '_') + '.npy', data_mat_res)
    np.save('../data/cached/target_' + category.replace('-', '_') + '.npy', target_vals)

    # Save feature subset lengths.
    np.save('../data/cached/target_' + category.replace('-', '_') + '_feature_subset_lengths.npy', feature_subset_lengths)


if __name__ == '__main__':
    import os
    import argparse

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-bow', action='store_true', help='do not compute BOW-like features')
    args = parser.parse_args()

    DATA_PATH = '../data/data-processed/data.pkl'
    if os.path.isfile(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        categories = ('book-relevance', 'type', 'category', 'category-broad')
        for category in categories:
            construct_features(data, category, use_bow_features=not args.no_bow)
        print("Feature engineering successfully performed. You may now run the evaluate.py script.")
    else:
        print("Processed dataset not found. Run the parse.py script before running this script.")

