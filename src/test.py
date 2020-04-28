import pickle
from feature_engineering import translate_nstd

# Load dictionary for non-standard Slovene.
with open('../data/slo_nstd_dict.p', 'rb') as f:
    slo_nstd_dict = pickle.load(f)

