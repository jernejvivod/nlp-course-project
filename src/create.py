import pickle
import unidecode

with open('./slo_nstd_dict.p', 'rb') as fp:
    d = pickle.load(fp)

def translate_nstd(text, dictionary):
    res = []
    for w in text.split(' '):
        dec = unidecode.unidecode(w).lower()
        if dec in dictionary:
            res.append(dictionary[dec])
        else:
            res.append(dec)
    return ' '.join(res)


