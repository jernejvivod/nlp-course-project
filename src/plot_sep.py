from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import random

idx_to_select = [3, 0]

data = np.load('../data/cached/data_book_relevance.npy')
target = np.load('../data/cached/target_book_relevance.npy')

with open('../data/data-processed/data.pkl', 'rb') as f:
    loaded = pkl.load(f)

messages = loaded['book-relevance']['x']

# sc = StandardScaler()
# data = sc.fit_transform(data)

fig = plt.figure(figsize=(2, 1))
ax = Axes3D(fig)
# 
# sequence_containing_x_vals = list(range(0, 100))
# sequence_containing_y_vals = list(range(0, 100))
# sequence_containing_z_vals = list(range(0, 100))
# 
# random.shuffle(sequence_containing_x_vals)
# random.shuffle(sequence_containing_y_vals)
# random.shuffle(sequence_containing_z_vals)

ax.scatter(data[target == 0, idx_to_select[0]], data[target == 0, idx_to_select[1]], np.zeros(data[target == 0, :].shape[0]), c='#ee442f')
ax.scatter(data[target == 1, idx_to_select[0]], data[target == 1, idx_to_select[1]], 0.1*np.ones(data[target == 1, :].shape[0]), c='#63acbe')
# plt.scatter(data[target == 0, idx_to_select[0]], data[target == 0, idx_to_select[1]], c='#ee442f', s=10)
# plt.scatter(data[target == 1, idx_to_select[0]], data[target == 1, idx_to_select[1]], c='#63acbe', s=10)
ax.set_xlabel('average word length', fontsize=13)
ax.set_ylabel('number of words in message', fontsize=13)
ax.set_zticks([0, 0.1])
ax.set_zticklabels(['No', 'Yes'])
plt.gcf().subplots_adjust(bottom=0.5)
plt.show()
