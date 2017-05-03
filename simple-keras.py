import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import cv2
from tqdm import tqdm

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

x_train = []
x_test = []
y_train = []
y_test = []

df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values[:], miniters=1000):
    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 15000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(x_valid, y_valid))
          
from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
# print(y_valid)
# print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > 0.4, beta=2, average='macro'))


# for predict in p_valud
# for f in tqdm(df_test.values[:20000], miniters=1000):
for f in tqdm(df_test.values[:], miniters=1000):
    path = ('input/test-jpg/%s.jpg' % f[0])
    # print(path)
    img = cv2.imread(path)
    x_test.append(cv2.resize(img, (32, 32)))

x_test = np.array(x_test, np.float16) / 255.
p_test = model.predict(x_test, batch_size=128)

n = len(p_test)
index = 0
k = 0

while k < 41:
    start = k*1000
    end = (k+1)*1000
    if k==40:
        end = n
    
    for p in tqdm(p_test[start:end], miniters=100):
        result = [p > 0.5]
        idx = np.where(p > 0.5)
        # print(idx)
        ans = []
        for x in idx[0]:
            # print(x)
            ans.append(labels[x])
        y_test.append(" ".join(ans))
        index+=1
        
    # makeitastring = '\n'.join(map(str, y_test))
    # index += 1
    # file_name = "answer_%s_%s.txt" % (k,index)
    # fh = open(file_name, 'w')
    # fh.write(makeitastring) 
    # fh.close()
    k+=1
    
makeitastring = '\n'.join(map(str, y_test))
file_name = "final_ans.txt"
fh = open(file_name, 'w')
fh.write(makeitastring) 
fh.close()




