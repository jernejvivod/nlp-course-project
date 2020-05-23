import pandas as pd
import re
import unidecode
import glob
import pickle
from sklearn import preprocessing
import argparse
import urllib.request
from bs4 import BeautifulSoup


def parse_discussions_raw(file_path):
    """
    Parse discussions from .xls file, remove redundant rows and columns and return.

    Args:
        file_path (str): Path to .xls file containing the discussions

    Returns:
        (pandas.core.frame.DataFrame): Parsed raw discussions.
    """

    # Parse pandas dataframe and drop redundant rows and columns if any.
    sheet_raw = pd.read_excel(file_path)
    return sheet_raw.dropna(how='all', axis='rows').dropna(how='all', axis='columns')


def parse_stories(folder_path):
    """
    Parse stories from files, remove redundant symbols, transform to ascii and return
    as dictionary mapping indices to stories.

    Args:
        folder_path (str): Path to the folder containing the stories.

    Returns:
        (dict): dictionary mapping IDs (0..N) to stories.
    """
    
    # Initialize dictionary for storing stories.
    stories = dict()

    # Set symbols to delete.
    to_delete = {',', '>', '<', '(', ')', ':', '-', '\n'}

    # Parse stories one by one.
    for idx, f in enumerate(glob.glob(folder_path + '*.txt')):
        with open(f, 'r') as f:

            # Translate to ascii, remove redundant symbols, fix punctuation spaces.
            story = unidecode.unidecode(f.read())
            for ch in to_delete:
                story = story.replace(ch, '')
            story = re.sub(r'(?<=[.,!,?])(?=[^\s])', r' ', story)
            story = re.sub(r'[0-9]+', '', story)
            stories[idx] = story
    
    # Return dictionary mapping IDs (0..N) to stories.
    return stories


def preprocess_for_target(data, category):
    """
    Extract relevant columns for different prediction scenarios. Encode the target value column
    using label encoding. Valid values for target are 'book-relevance', 'type', 'category' and 'category-broad'.

    Args:
        data (pandas.core.frame.DataFrame): Parsed raw discussions
        category (str): Prediction target specification

    Returns:
        (tuple): Data with x (message) and y (target) columns, array of user IDs corresponding to the
        messages.
    """

    if category == 'book-relevance':
        # Encode labels.
        data['Book relevance'] = preprocessing.LabelEncoder().fit_transform(data['Book relevance'].values.astype(str))
        return data.rename(columns={'Message': 'x', 'Book relevance': 'y'})[['x', 'y']], data['User ID'].to_numpy().astype(int)
    elif category == 'type':
        # Encode labels.
        data['Type'] = preprocessing.LabelEncoder().fit_transform(data['Type'].values.astype(str))
        return data.rename(columns={'Message': 'x', 'Type': 'y'})[['x', 'y']], data['User ID'].to_numpy().astype(int)
    elif category == 'category':
        # Encode labels.
        data['Category'] = preprocessing.LabelEncoder().fit_transform(data['Category'].values.astype(str))
        return data.rename(columns={'Message': 'x', 'Category': 'y'})[['x', 'y']], data['User ID'].to_numpy().astype(int)
    elif category == 'category-broad':
        # Encode labels.
        data['CategoryBroad'] = preprocessing.LabelEncoder().fit_transform(data['CategoryBroad'].values.astype(str))
        return data.rename(columns={'Message': 'x', 'CategoryBroad': 'y'})[['x', 'y']], data['User ID'].to_numpy().astype(int)
    else:
        raise(ValueError('unknown category specified'))


def get_names(data):
    """
    Parse chat names from data and save to file.

    Args:
        data (pandas.core.frame.DataFrame): Parsed raw discussions
    """
    with open('../data/chat_names.txt', 'w') as outfile:
        outfile.write('\n'.join(map(lambda x: unidecode.unidecode(x).lower(), set(data['Name'].values))))


def obtain_names():
    """
    Parse list of common first names from website and write to file.
    """
    

    url='http://www.mojmalcek.si/clanki_in_nasveti/nosecnost/118/650_imen_za_novorojencka.html'
    m = urllib.request.urlopen(url)
    soup = BeautifulSoup(m, 'html5lib')
    res = map(lambda x: unidecode.unidecode(x.lower()), soup.find(text=re.compile("Abecedni seznam imen")).next.next.text.split('\n'))
    res = filter(lambda x: len(x) > 1, res)
    with open('../data/names.txt', 'w') as outfile:
        outfile.write('\n'.join(res))


def obtain_curse_words():
    """
    Parse curse words from website and write to file.
    """

    url='http://razvezanijezik.org/?page=kletvice+v+sloven%C5%A1%C4%8Dini'
    m = urllib.request.urlopen(url)
    soup = BeautifulSoup(m, 'html5lib')
    to_write = [unidecode.unidecode(i.string).lower() for i in soup.find_all('a') if i.string is not None and i.string != '?']
    idx_start = to_write.index('\nkletvice v slovenscini') + 1
    with open('../data/curse_words.txt', 'w') as outfile:
        outfile.write('\n'.join(list(set(to_write[idx_start:]))))


def parse(discussions_path):
    """
    Parse chat names, names from website and curse words. After this, 
    the words may have to be manually checked for false entries.
    
    Args:
        discussions_path (str): Path to .xls file containing the discussions
    """
    
    data = parse_discussions_raw(discussions_path)
    get_names(data)
    obtain_names()
    obtain_curse_words()


def initialize(discussions_path, stories_path, save_path):
    """
    Initialize dictionary containing data needed for feature engineering and prediction
    and save it to file.

    Args:
        discussions_path (str): Path to .xls file containing the discussions
        stories_path (str): Path to folder containing the stories.
    """


    # Parse data from discussions and get data
    # for different prediction goals.
    data = parse_discussions_raw(discussions_path)
    data_book_relevance, user_ids = preprocess_for_target(data, 'book-relevance')
    data_type, _ = preprocess_for_target(data, 'type')
    data_category, _ = preprocess_for_target(data, 'category')
    data_broad_category, _ = preprocess_for_target(data, 'category-broad')

    # Parse stories.
    stories = parse_stories(stories_path)
    
    # Parse stored chat names, names and curse words.
    with open('../data/chat_names.txt', 'r') as f1, open('../data/names.txt', 'r') as f2,\
            open('../data/curse_words.txt', 'r') as f3, open('../data/story_names.txt', 'r') as f4, \
            open('../data/clue_words.txt', 'r') as f5:
        chat_names = list(map(lambda x: x.strip(), f1.readlines()))
        names = list(map(lambda x: x.strip(), f2.readlines()))
        curse_words = list(map(lambda x: x.strip(), f3.readlines()))
        story_names = list(map(lambda x: x.strip(), f4.readlines()))
        clue_words = list(map(lambda x: x.strip(), f5.readlines()))
    
    
    # Create results dictionary and save.
    res = {'book-relevance' : data_book_relevance, 
           'type' : data_type, 
           'category' : data_category, 
           'category-broad' : data_broad_category,
           'chat-names' : chat_names,
           'names' : names,
           'user-ids' : user_ids,
           'curse-words' : curse_words,
           'story-names' : story_names,
           'clue-words' : clue_words,
           'stories' : stories}

    with open(save_path + 'data.pkl', 'wb') as f:
        pickle.dump(res, f)
    

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    g1 = parser.add_argument_group(title='', description='')
    g2 = g1.add_mutually_exclusive_group(required=True)
    g2.add_argument('--parse',action='store_true', help='Parse data from the web (names, list of curse words) and \
            chat names from discussions file. Run script with this argument before running with --initialize')
    g2.add_argument('--initialize', action='store_true', help='Initialize data dictionary. \
            Run this after the script has bean run with the --parse argument and the resulting data manually checked.')
    args = parser.parse_args()
    
    # Set path to file containing the discussions.
    DISCUSSIONS_PATH = '../data/discussions.xlsx'
    
    # Parse or initialize.
    if args.parse:
        parse(DISCUSSIONS_PATH)
    else:
        try:
            initialize(DISCUSSIONS_PATH, '../data/stories/', '../data/data-processed/')
            print("Data successfully preprocessed and saved!")
        except FileNotFoundError:
            print("All files containing chat names, given names and curse words not found. \
Run the script with the --parse argument first before running with the --initialize argument.")


