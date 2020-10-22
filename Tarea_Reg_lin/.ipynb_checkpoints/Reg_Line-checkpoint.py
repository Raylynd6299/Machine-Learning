import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lm
from sklearn.metrics import mean_squared_error, r2_score

somok = lambda cad: 2 if(cad=="yes") else 1
sexx = lambda cad:1 if(cad == "female") else 2
somok_lm = lambda cad: 1 if(cad=="yes") else 0
sexx_lm = lambda cad:0 if(cad == "female") else 1

Regions = {"northeast":1,"northwest":2,"southeast":3,"southwest":4}
regi = lambda cad: Regions[cad]

datas = pd.read_csv("insurance.csv",converters={"sex":sexx,"smoker":somok,"region":regi})
datas_lm = pd.read_csv("insurance.csv",converters={"sex":sexx_lm,"smoker":somok_lm,"region":regi})

print(datas.agg({"charges":['min', 'max', 'median', 'skew',"mean"]}))

print(datas["charges"].describe())

datas["charges"].hist(bins=15)

datas[["age", "bmi", "children", "charges"]].corr()
datas.plot(figsize=(18,5))

datanum= datas_lm.to_numpy()
np.set_printoptions(suppress=True)
print(datanum)
datanum.shape
print(datanum[:,-1:])
Y = datanum[:,-1:]
print(datanum[:,:-1])
X = datanum[:,:-1]
def AgregarCampo(datanum):
    location = datanum[:,-1:]
    
    lista = np.transpose(location).tolist()[0]
    regionnorthwest = list(map(lambda number:1 if(number == 2) else 0 , lista))
    regionsoutheast = list(map(lambda number:1 if(number == 3) else 0 , lista))
    regionsouthwest = list(map(lambda number:1 if(number == 4) else 0 , lista))
    
    regionnorthwest = np.array(regionnorthwest).reshape(len(regionnorthwest),1)
    regionsoutheast = np.array(regionsoutheast).reshape(len(regionnorthwest),1)
    regionsouthwest = np.array(regionsouthwest).reshape(len(regionnorthwest),1)
    return np.concatenate((datanum[:,:-1],regionnorthwest,regionsoutheast,regionsouthwest),1)
    
X = AgregarCampo(X)
print(X)
print(X.shape)
print(Y.shape)
print(X)
reg_mod = lm()

reg_mod.fit(X, Y)
y_predict = reg_mod.predict(X)

reg_mod.coef_

rmse = mean_squared_error(Y, y_predict)
r2 = r2_score(Y, y_predict)

print('Slope:' ,reg_mod.coef_)
print('Intercept:', reg_mod.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)








