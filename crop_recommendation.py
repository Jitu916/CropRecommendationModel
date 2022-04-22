#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("Crop_recommendation.csv")
data = np.array(data)

X = data[1:, :-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
#print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
nb = GaussianNB()


nb.fit(X_train, y_train)

inputt=[int(x) for x in "77 50 21 22 64 6 100".split(' ')]
final=[np.array(inputt)]

b = nb.predict(final)


pickle.dump(nb,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


