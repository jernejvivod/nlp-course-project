import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

fig1 = plt.figure()
ax1 = SubplotHost(fig1, 111)
fig1.add_subplot(ax1)

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


# First X-axis
rects = ax1.barh(indices, np.hstack((c3/np.sum(c3), c2/np.sum(c2), c1/np.sum(c1))))
ax1.set_yticks(indices)
ax1.set_yticklabels(labels1 + labels2 + labels3, rotation=90)

# Second X-axis
ax2 = ax1.twiny()
offset = 0, -25 # Position of the second axis
new_axisline = ax2.get_grid_helper().new_fixed_axis
ax2.axis["left"] = new_axisline(loc="left", axes=ax2, offset=offset)
ax2.axis["top"].set_visible(False)

ax2.set_xticks([0.0, 0.6, 1.0])
ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
ax2.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.8]))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter(['mammal', 'reptiles']))

ax1.grid(1)
