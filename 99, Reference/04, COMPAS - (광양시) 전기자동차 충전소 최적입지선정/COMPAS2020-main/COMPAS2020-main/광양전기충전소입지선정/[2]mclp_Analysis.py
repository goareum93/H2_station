# %%
# %%
# Data input
import pathlib
import numpy as np
from numpy import random
from functools import reduce
from collections import defaultdict
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import geopandas as gpd
from tqdm.notebook import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# 시각화
import pydeck as pdk
import shapely
from shapely.geometry import Polygon, Point
from IPython.display import display
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'

#입지선정 지수 분석
import sklearn.cluster

#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY  
# %%
df_test = gpd.read_file('df_EDA_result.geojson')
df_test
# 100X100 grid에서 central point 찾기
df_list = []
df_list2 = []
for i in df_test['geometry']:
    cent = [[i[0].centroid.coords[0][0],i[0].centroid.coords[0][1]]]
    df_list.append(cent)
    df_list2.append(Point(cent[0]))
df_test['coord_cent'] = 0
df_test['geo_cent'] = 0
df_test['coord_cent']= pd.DataFrame(df_list) # pydeck을 위한 coordinate type
df_test['geo_cent'] = df_list2 # geopandas를 위한 geometry type
df_test

#%%


df_result = df_test
df_result['정규화_인구'] = df_result['val'] / df_result['val'].max()
df_result['정규화_교통량_07'] = df_result['교통량_07'] / df_result['교통량_07'].max()
df_result['정규화_교통량_15'] = df_result['교통량_15'] / df_result['교통량_15'].max()
df_result['정규화_혼잡빈도강도합'] = df_result['혼잡빈도강도합'] / df_result['혼잡빈도강도합'].max()
df_result['정규화_혼잡시간강도합'] = df_result['혼잡시간강도합'] / df_result['혼잡시간강도합'].max()
df_result['정규화_자동차등록'] = df_result['자동차등록'] / df_result['자동차등록'].max()
df_result['정규화_전기자동차등록'] = df_result['전기자동차등록'] / df_result['전기자동차등록'].max()

df_result

# %%
# Logistic Regression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats


df_LR = df_result
X = df_LR[["정규화_인구","정규화_교통량_07","정규화_교통량_15","정규화_혼잡빈도강도합","정규화_혼잡시간강도합", "정규화_자동차등록","정규화_전기자동차등록" ]]
y = df_LR["FS_station"]
regr = linear_model.LogisticRegression()
regr.fit(X, y)
FS_coeff = regr.coef_[0]
print('급속충전소 Intercept: ', regr.intercept_)
print('급속충전소 Coefficients: \n', FS_coeff)

X2 = sm.add_constant(X)
est = sm.OLS(1+np.exp(y), X2)
est2 = est.fit()
print(est2.summary())
with open('summary_OLS_FF.txt', 'w') as fh:
    fh.write(est2.summary().as_text())



df_LR = df_result
y = df_LR["SS_station"]
regr = linear_model.LogisticRegression()
regr.fit(X, y)
SS_coeff = regr.coef_[0]
print('완속충전소 Intercept: ', regr.intercept_)
print('완속충전소 Coefficients: \n', SS_coeff)

X2 = sm.add_constant(X)
est = sm.OLS(1+np.exp(y), X2)
est2 = est.fit()
print(est2.summary())

with open('summary_OLS_SF.txt', 'w') as fh:
    fh.write(est2.summary().as_text())

import math
df_result['w_FS'] = 0 

df_result['w_FS']  = 1/(1+np.exp(-(FS_coeff[0]*df_result['정규화_인구']+
                     FS_coeff[1]*df_result['정규화_교통량_07']+
                     FS_coeff[2]*df_result['정규화_교통량_15']+
                     FS_coeff[3]*df_result['정규화_혼잡빈도강도합']*0+
                     FS_coeff[4]*df_result['정규화_혼잡시간강도합']+
                     FS_coeff[5]*df_result['정규화_자동차등록']+
                     FS_coeff[6]*df_result['정규화_전기자동차등록']
                    )))



df_result['w_SS'] = 0 
df_result['w_SS'] =  1/(1+np.exp(-1*(SS_coeff[0]*df_result['정규화_인구']+
                     SS_coeff[1]*df_result['정규화_교통량_07']*0+
                     SS_coeff[2]*df_result['정규화_교통량_15']*0+
                     SS_coeff[3]*df_result['정규화_혼잡빈도강도합']+
                     SS_coeff[4]*df_result['정규화_혼잡시간강도합']+
                     SS_coeff[5]*df_result['정규화_자동차등록']+
                     SS_coeff[6]*df_result['정규화_전기자동차등록']
                    )))

try:    
    df_result[['grid_id','geometry',
               '정규화_인구','정규화_교통량_07','정규화_교통량_15',
              '정규화_혼잡빈도강도합', '정규화_혼잡시간강도합','정규화_자동차등록','정규화_전기자동차등록',
               'w_FS','w_SS','geo_Poss_FS','geo_Poss_SS','FS_station','SS_station']].to_file("df_result_LR.geojson", driver="GeoJSON")
except:
    pass
# %%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sksurv.linear_model import CoxPHSurvivalAnalysis
# %%

df_LR = df_result
y = df_LR["SS_station"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ridge = Ridge().fit(X_train, y_train)
print("훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test,y_test)))

#%%
Logit = LogisticRegression().fit(X_train, y_train)
print("Logit.coef_: {}".format(Logit.coef_))
print("Logit.intercept_ : {}".format(Logit.intercept_))
print("훈련 세트의 정확도 : {:.2f}".format(Logit.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(Logit.score(X_test,y_test)))

#%%
Logit = LogisticRegression(penalty='l1', solver='liblinear').fit(X_train, y_train)
print("Logit.coef_: {}".format(Logit.coef_))
print("Logit.intercept_ : {}".format(Logit.intercept_))
print("훈련 세트의 정확도 : {:.2f}".format(Logit.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(Logit.score(X_test,y_test)))


# %%
Linear = LinearRegression().fit(X_train, y_train)
print("Linear.coef_: {}".format(Linear.coef_))
print("Linear.intercept_ : {}".format(Linear.intercept_))
print("훈련 세트의 정확도 : {:.2f}".format(Linear.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(Linear.score(X_test,y_test)))


# %%
from sklearn.linear_model import Lasso
Lasso = Lasso(alpha = 0.000000000005, normalize= True).fit(X_train, y_train)
print("Lasso.coef_: {}".format(Lasso.coef_))
print("Lasso.intercept_ : {}".format(Lasso.intercept_))
print("훈련 세트의 정확도 : {:.2f}".format(Lasso.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(Lasso.score(X_test,y_test)))


# %%

Cox = CoxPHSurvivalAnalysis().fit(np.array(list(map(int, y_train)),df_LR[['w_SS']]))
print("Cox.coef_: {}".format(Cox.coef_))
print("Cox.intercept_ : {}".format(Cox.intercept_))
print("훈련 세트의 정확도 : {:.2f}".format(Cox.score(X_train,y_train)))
print("테스트 세트의 정확도 : {:.2f}".format(Cox.score(X_test,y_test)))


# %%
X = df_LR[["정규화_인구","정규화_교통량_07","정규화_교통량_15","정규화_혼잡빈도강도합","정규화_혼잡시간강도합", "정규화_자동차등록","정규화_전기자동차등록" ]]

# %%
X = X.astype(float)
Cox = CoxPHSurvivalAnalysis().fit(X, np.array(list(map(int, y_train))))
# %%
np.array(df_LR[['w_SS']])
#%%
from sksurv.datasets import load_whas500
X, y = load_whas500()
X = X.astype(float)
estimator = CoxPHSurvivalAnalysis().fit(X, y)
chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:10])

for fn in chf_funcs:
    plt.step(fn.x, fn(fn.x), where="post")

plt.ylim(0, 1)
plt.show()
# %%
X
y

df_LR[df_LR['SS_station']==1]
