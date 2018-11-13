from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

np.random.seed(7)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# generate 2d classification dataset
X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.0, n_features=8)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=10)
# Compile model
adam_optimizer = adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
# Fit the model
model.fit(train_x, train_y, epochs=50, batch_size=10)
# evaluate the model
scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


logReg = LogisticRegression()
logReg.fit(train_x, train_y)
train_score = logReg.score(train_x, train_y)
test_score = logReg.score(test_x, test_y)
print(train_score, test_score)
