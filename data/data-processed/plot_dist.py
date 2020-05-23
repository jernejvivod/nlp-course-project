import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle as pkl

df = pd.read_excel('../discussions2.xlsx')
u1, c1 = np.unique(df['Book relevance'].dropna(), return_counts=True)
u2, c2 = np.unique(df['Type'].dropna(), return_counts=True)
u3, c3 = np.unique(df['CategoryBroad'].dropna(), return_counts=True)
import pdb
pdb.set_trace()

label_map1 = {
        'Yes':'Yes',
        'No':'No',
        }

label_map2 = {
        'A':'Answer',
        'Q':'Question',
        'S':'Statement',
        }

label_map3 = {
        'C':'Chatting',
        'D':'Discussion',
        'I':'Identity',
        'M':'Moderation',
        'O':'Other',
        'S':'Switching',
        }

labels1 = [label_map1[el] for el in u1]
labels2 = [label_map2[el] for el in u2]
labels3 = [label_map3[el] for el in u3]

indices = np.arange(len(labels1 + labels2 + labels3))
width = 0.35

fig, ax = plt.subplots(figsize=(20, 20))
rects = ax.barh(indices, np.hstack((c3/np.sum(c3), c2/np.sum(c2), c1/np.sum(c1))))
ax.set_yticks(indices)
ax.set_yticklabels(labels3 + labels2 + labels1, fontsize=15)

rects[-1].set_color('#601a4a')
rects[-2].set_color('#601a4a')

rects[-3].set_color('#ee442f')
rects[-4].set_color('#ee442f')
rects[-5].set_color('#ee442f')

rects[-6].set_color('#63acbe')
rects[-7].set_color('#63acbe')
rects[-8].set_color('#63acbe')
rects[-9].set_color('#63acbe')
rects[-10].set_color('#63acbe')
rects[-11].set_color('#63acbe')

rects[-1].set_label('Book relevance')
rects[-3].set_label('Type')
rects[-6].set_label('Broad category')
ax.set_xlabel('Proportion of messages', fontsize=18)

plt.gcf().subplots_adjust(bottom=0.20, left=0.20)

plt.legend([rects[-1], rects[-3], rects[-6]], ['Book relevance', 'Type', 'Broad category'], prop={'size': 12})
