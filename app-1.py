import catboost
import numpy as np
import pickle

model=pickle.load(open('model.pkl','rb'))

inpu=np.array([778437,63573,2243430,114341,1157324,2008,12,1
                  ,0.749,6,-6.57,1,0.23,0.757,0.0,0.76,0.693,108.863,274560,4])

print(model.predict(inpu)-1)


