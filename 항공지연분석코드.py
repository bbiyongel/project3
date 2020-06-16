# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
%matplotlib inline
#%%
# 데이터 불러오기
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT.csv",encoding='CP949')
len(AFSNT) #행:987709개
len(AFSNT.columns[:]) #열:17개
#%%
### AFSNT 전처리(1)
#결항 존재 행8259개 제거 => 979450개 남음
AFSNT=AFSNT[AFSNT.CNL=='N'] 
# 결항 변수 2개(CNL,CNR) 제거
AFSNT=AFSNT.drop(columns=['CNL','CNR']) 
len(AFSNT) #행:979450개
len(AFSNT.columns[:]) #열:17개

# 실제 시각 결측치 존재 행 2개 제거 => 행 979448개
AFSNT.isnull().sum()# 실제 시각 결측치 존재, DRR 지연사유 결측치 841598개
AFSNT=AFNST.dropna(subset=['ATT'])

## IRR(부정기편) 부정기편 해당 행 제거
AFSNT=AFSNT[AFSNT.IRR=='N']
len(AFSNT) #행:958150개
len(AFSNT.columns[:]) #열:15개

## 결측치 확인
AFSNT.isnull().sum() # DRR 지연사유 841598개

## 실제시각-계획시각 변수 생성(지연시간 확인)
# Y-M-D H:M으로 합치기
AFSNT['STT_DATE'] = AFSNT['SDT_YY'].apply(str) + '-' + AFSNT['SDT_MM'].apply(str) +'-' + AFSNT['SDT_DD'].apply(str) + ' ' + AFSNT['STT'].apply(str)
AFSNT['ATT_DATE'] = AFSNT['SDT_YY'].apply(str) + '-' + AFSNT['SDT_MM'].apply(str) +'-' + AFSNT['SDT_DD'].apply(str) + ' ' + AFSNT['ATT'].apply(str)
# datetime으로 변환  
AFSNT['STT_DATE'] = pd.to_datetime(AFSNT['STT_DATE'],  format = '%Y-%m-%d %H:%M')
AFSNT['ATT_DATE'] = pd.to_datetime(AFSNT['ATT_DATE'], errors = 'coerce', format = '%Y-%m-%d %H:%M')
# 시간 차이 값 변수 만들기
AFSNT['DIFF_TIME']=(AFSNT['ATT_DATE']-AFSNT['STT_DATE'])/np.timedelta64(1,'m')
AFSNT['DIFF_TIME'].head(5)
# 국내선 항공 지연 : 30분 이상
# 지연인데 차이값이 30분 미만인 행 확인
set(AFSNT[AFSNT['DLY']=='Y'].index)-set(AFSNT[AFSNT['DIFF_TIME'] >= 30].index)
thirty_down=set(AFSNT[AFSNT['DLY']=='Y'].index)-set(AFSNT[AFSNT['DIFF_TIME'] >= 30].index)
len(thirty_down) # 29개 행 존재
# 제거
AFSNT=AFSNT.drop(thirty_down)
len(AFSNT) # 958121
len(AFSNT.columns[:])
#지연인데 30분미만인 존재행 재확인
thirty_down=set(AFSNT[AFSNT['DLY']=='Y'].index)-set(AFSNT[AFSNT['DIFF_TIME'] >= 30].index)
len(thirty_down) #0

# STT(계획시각) 시,분 나누기
AFSNT[["STT_H","STT_M"]]=AFSNT["STT"].str.split(":",n=1,expand=True)
AFSNT.head(10)
AFSNT[:].columns
#%%
### 시각화탐색
# 지연여부비율
sns.countplot(x="DLY",data=AFSNT)
plt.show()
dl=AFSNT[AFSNT['DLY']=='Y']
dlcount=len(dl)
undl=AFSNT[AFSNT['DLY']=='N']
undlcount=len(undl)
dlcount/(dlcount+undlcount)*100 
undlcount/(dlcount+undlcount)*100 

# 월별 지연비율
delayed=AFSNT[AFSNT["DLY"]=='Y']["SDT_MM"].value_counts()
undelayed=AFSNT[AFSNT["DLY"]=='N']["SDT_MM"].value_counts()
df = pd.DataFrame([delayed,undelayed])
df.index = ["delayed","undelayed"]
df = df.T
delayed_rate=(df.delayed/(df.delayed+df.undelayed))*100
x=df.index
y=delayed_rate
plt.plot(x,y)
plt.xlabel("SDT_MM")
plt.ylabel("delay percent")
plt.bar(x,y)
plt.show()

# 요일별 지연비율
delayed=AFSNT[AFSNT["DLY"]=='Y']["SDT_DY"].value_counts()
undelayed=AFSNT[AFSNT["DLY"]=='N']["SDT_DY"].value_counts()
df = pd.DataFrame([delayed,undelayed])
df.index = ["delayed","undelayed"]
df = df.T
delayed_rate=(df.delayed/(df.delayed+df.undelayed))*100
x=df.index
y=delayed_rate
plt.plot(x,y)
plt.xlabel("SDT_DY")
plt.ylabel("delay percent")
plt.bar(x,y)
plt.show()

# AOD 출도착별 지연비율
delayed=AFSNT[AFSNT["DLY"]=="Y"]["AOD"].value_counts()
undelayed=AFSNT[AFSNT["DLY"]=="N"]["AOD"].value_counts()
df = pd.DataFrame([delayed,undelayed])
df.index = ["delayed","undelayed"]
df = df.T
delayed_rate=(df.delayed/(df.delayed+df.undelayed))*100
x=df.index
y=delayed_rate
plt.xlabel("AOD")
plt.ylabel("delay percent")
plt.bar(x,y)
plt.show()

# 공항별 지연비율
delayed=AFSNT[AFSNT["DLY"]=="Y"]["ARP"].value_counts()
undelayed=AFSNT[AFSNT["DLY"]=="N"]["ARP"].value_counts()
df = pd.DataFrame([delayed,undelayed])
df.index = ["delayed","undelayed"]
df = df.T
delayed_rate=(df.delayed/(df.delayed+df.undelayed))*100
x=df.index
y=delayed_rate
plt.plot(x,y)
plt.xlabel("ARP")
plt.ylabel("delay percent")
plt.bar(x,y)
plt.show()

# 시간대별 지연비율
delayed=AFSNT[AFSNT["DLY"]=="Y"]["STT_H"].value_counts()
undelayed=AFSNT[AFSNT["DLY"]=="N"]["STT_H"].value_counts()
df = pd.DataFrame([delayed,undelayed])
df.index = ["delayed","undelayed"]
df = df.T
df.isnull().sum()
df.fillna(0, inplace = True)
pd.set_option('display.max_rows', 1000)
delayed_rate=(df.delayed/(df.delayed+df.undelayed))*100
x=df.index
y=delayed_rate
plt.xlabel("STT_H")
plt.ylabel("delay percent")
plt.bar(x,y)
plt.show()
#%%
### AFSNT 전처리(2)
##값 변환
# DLY(지연여부) Y->1, N->0
AFSNT['DLY']=AFSNT['DLY'].replace("Y",1)
AFSNT['DLY']=AFSNT['DLY'].replace("N",0)
AFSNT['DLY'].head(10)
AFSNT['DLY'].unique()

###변수생성
# 시간에 따른 비행기/ 시간에 따른 공항별  비행기에 관한 변수
hourly_flt = AFSNT.groupby(['SDT_YY','SDT_MM', 'SDT_DD', 'STT_H'])['SDT_DY'].count().reset_index(name = 'HOURLY_FLT')
arp_hourly_flt = AFSNT.groupby(['SDT_YY','SDT_MM', 'SDT_DD', 'ARP', 'STT_H'])['SDT_DY'].count().reset_index(name = 'ARP_HOURLY_FLT')
odp_arp_hourly_flt = AFSNT.groupby(['SDT_YY','SDT_MM', 'SDT_DD', 'ODP', 'STT_H'])['SDT_DY'].count().reset_index(name = 'ODP_HOURLY_FLT')
AFSNT=pd.merge(AFSNT,hourly_flt ,left_on="SDT_YY",right_index=True)
AFSNT=pd.merge(AFSNT,arp_hourly_flt  ,left_on="SDT_YY",right_index=True)
AFSNT=pd.merge(AFSNT,arr_arp_hourly_flt  ,left_on="SDT_YY",right_index=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer
scaler = MinMaxScaler() 
#최소값(Min)과 최대값(Max)을 사용해서 '0~1' 사이의 범위(range)로 데이터를 표준화해주는 '0~1 변환
AFSNT[['HOURLY_FLT','ARP_HOURLY_FLT','ODP_HOURLY_FLT']] = scaler.fit_transform(AFSNT[['HOURLY_FLT','ARP_HOURLY_FLT','ODP_HOURLY_FLT']])

# 항공사마다 몇편의 항공편을 운행하는지
airline_df = AFSNT.groupby('FLO')['FLT'].nunique().reset_index(name = 'FLO_FLT')
AFSNT=pd.merge(AFSNT,airline_df )

# 출도착 공항편에 대한 비행편명에 대한 가중치 반영
arp_flt_cnt = AFSNT.groupby(['AOD','ODP'])['FLT'].count().reset_index(name = 'ARP_ODP_FLT')
AFSNT= pd.merge(AFSNT, arp_flt_cnt)

# 요일별 항공편수 반영
day_arp = AFSNT.groupby(['SDT_DY','AOD','ODP'])['FLT'].count().reset_index(name = 'DAY_ARP')
AFSNT= pd.merge(AFSNT, day_arp)

#scaling 기본스케일링
scaler = StandardScaler()
AFSNT[['FLO_FLT', 'ARP_ODP_FLT', 'DAY_ARP']] = scaler.fit_transform(AFSNT[['FLO_FLT', 'ARP_ODP_FLT', 'DAY_ARP']])

##one-hot encoding
#AOD(출도착)
sorted(AFSNT['AOD'].unique())
# 'A', 'D'
one_hot_AOD=pd.get_dummies(AFSNT['AOD'])
one_hot_AOD.head()
len(one_hot_AOD)
# 열합치기
AFSNT=pd.merge(AFSNT,one_hot_AOD,left_on="SDT_YY",right_index=True)
AFSNT[:].columns 
AFSNT.rename(columns={'A': 'AOD_A', 'D': 'AOD_D'}, inplace=True)

# FLO(항공사) 
sorted(AFSNT['FLO'].unique())
# 'A', 'B', 'F', 'H', 'I', 'J', 'L'
one_hot_FLO=pd.get_dummies(AFSNT['FLO'])
one_hot_FLO.head()
# 열합치기
AFSNT=pd.merge(AFSNT,one_hot_FLO,left_on="SDT_YY",right_index=True)
AFSNT[:].columns 
AFSNT.rename(columns={'A': 'FLO_A', 'B': 'FLO_B', 'F': 'FLO_F', 'H': 'FLO_H', 
                      'I': 'FLO_I', 'J': 'FLO_J', 'L': 'FLO_L'}, inplace=True)

# SDT_MM(월)
AFSNT['SDT_MM'].unique()
# 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
one_hot_MM=pd.get_dummies(AFSNT['SDT_MM'])
one_hot_MM.head()
AFSNT=pd.merge(AFSNT,one_hot_MM,left_on="SDT_YY",right_index=True)
AFSNT[:].columns
AFSNT.rename(columns={1: '1월',2: '2월',  3:'3월',  4:'4월',  5:'5월',  
                      6: '6월',  7:'7월',  8:'8월',  9:'9월', 10:'10월', 11:'11월', 12:'12월'}, inplace=True)

# DRR(지연사유)
AFSNT["DRR"].isnull().sum() 
AFSNT["DRR"] = AFSNT["DRR"].fillna(0)
AFSNT["DRR"].isnull().sum() #0
AFSNT["DRR_ch"] = AFSNT["DRR"].str.slice(start=0,stop=1)
AFSNT.head(10)
AFSNT["DRR_ch"].unique()
# nan, 'C', 'D', 'Z', 'A', 'B'
one_hot_DRR=pd.get_dummies(AFSNT['DRR_ch'])
one_hot_DRR.head()
# 열합치기
len(AFSNT)
AFSNT=pd.merge(AFSNT,one_hot_DRR,left_on="SDT_YY",right_index=True)
AFSNT[:].columns 
AFSNT.rename(columns={'A': 'DRR_A', 'B': 'DRR_B', 'C': 'DRR_C', 'D': 'DRR_D', 'Z': 'DRR_Z'}, inplace=True)

# SDT_DY(요일) -> 금토일1, 그외0
AFSNT['SDT_DY'].unique()
# '일', '월', '화', '수', '목', '금', '토'
AFSNT['week']=[1 if i=='금' or i=='토' or i=='일' else 0 for i in AFSNT['SDT_DY']]
AFSNT.head()  
AFSNT.to_csv("C:/Users/shimeunyoung/Desktop/AFSNT_J.csv",encoding='CP949')
#%%
# 변수제거
AFSNT.columns[:]
AFSNT=AFSNT.drop(columns=['SDT_DY','SDT_DD', 'FLO', 'FLT','REG', 'AOD', 'IRR', 'STT', 'ATT','DRR',
       'DIFF_TIME', 'STT_DATE', 'ATT_DATE', 'STT_M', 'DRR_ch'])

# 2017,2018 / 2019 데이터 나누기
AFSNT_19=AFSNT[AFSNT.SDT_YY==2019]
len(AFSNT_19) #행:189508개
AFSNT_19.to_csv("C:/Users/shimeunyoung/Desktop/AFSNT_19.csv",encoding='CP949')

AFSNT_T=AFSNT[AFSNT.SDT_YY!=2019]
len(AFSNT_T) #행:768613개
len(AFSNT_T.columns[:]) #39
AFSNT_T.to_csv("C:/Users/shimeunyoung/Desktop/AFSNT_1718.csv",encoding='CP949')
sns.countplot(x="DLY",data=AFSNT_T)
plt.show()
dl=AFSNT_T[AFSNT_T['DLY']==1]
dlcount=len(dl)
undl=AFSNT_T[AFSNT_T['DLY']==0]
undlcount=len(undl)
dlcount/(dlcount+undlcount)*100 #13%  12.713290043233721
undlcount/(dlcount+undlcount)*100 #87%    87.28670995676629
#%%
# 2017-2018 데이터 학습및 검증한 뒤 2019 데이터 적용해서 예측하기
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT_1718.csv",encoding='CP949')
AFSNT.columns[:]
AFSNT=AFSNT.drop('Unnamed: 0', 1)
len(AFSNT.columns[:]) 
len(AFSNT) #768613

X_train = AFSNT.drop('DLY', 1)
y_train=AFSNT['DLY']

# 언더샘플링
# train데이터를 넣어 복제함
from imblearn.under_sampling import *
rus = RandomUnderSampler(random_state=0)
rus.fit(X_train,list(y_train))

X_resampled, y_resampled = rus.fit_resample(X_train,list(y_train))
print('After UnderSampling, the shape of train_X: {}'.format(X_resampled.shape))
print('After UnderSampling, the shape of train_y: {} \n'.format(X_resampled.shape))
print("After UnderSampling, counts of label '1': {}".format(sum(y_resampled==1)))
print("After UnderSampling, counts of label '0': {}".format(sum(y_resampled==0)))

#데이터 정리
dfx=pd.DataFrame(X_resampled)
dfx.rename(columns={0:'SDT_YY', 1:'SDT_MM', 2:'ARP', 3:'ODP',4: 'STT_H', 5:'HOURLY_FLT',
      6: 'ARP_HOURLY_FLT', 7:'ODP_HOURLY_FLT', 8:'FLO_FLT', 9:'ARP_ODP_FLT',10: 'DAY_ARP',
      11: 'AOD_A', 12:'AOD_D', 13:'FLO_A', 14:'FLO_B', 15:'FLO_F',16: 'FLO_H',17: 'FLO_I', 
      18:'FLO_J', 19:'FLO_L', 20:'1월', 21:'2월', 22:'3월',23: '4월', 24:'5월', 25:'6월', 
      26:'7월',27: '8월',28: '9월', 29:'10월', 30:'11월', 31:'12월', 32:'DRR_A',
      33: 'DRR_B',34: 'DRR_C', 35:'DRR_D', 36:'DRR_Z',37: 'week'}, inplace=True)
dfy=pd.DataFrame(y_resampled)
dfy.rename(columns={0:'DLY'},inplace=True)
AFSNT=pd.merge(dfx,dfy,left_index=True, right_index=True)
AFSNT.to_csv("C:/Users/shimeunyoung/Desktop/AFSNT_S.csv",encoding='CP949')
AFSNT=AFSNT.drop(columns=['ARP','ODP'])
AFSNT.to_csv("C:/Users/shimeunyoung/Desktop/AFSNT_U.csv",encoding='CP949')
#%%
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT_U.csv",encoding='CP949')
AFSNT.columns[:]
len(AFSNT.columns[:])
AFSNT.info()
AFSNT=AFSNT.drop(columns=['Unnamed: 0','SDT_YY'])

#로지스틱회귀
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm

x = AFSNT.drop('DLY', 1)
y=AFSNT['DLY']

# train, test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(len(x_train), len(x_test), len(y_train), len(y_test))

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
print('정확도 :', metrics.accuracy_score(y_test, y_pred)) #정확도 : 0.566744443079946
cnf_metrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_metrix)
'''
[[13653 10651]
 [10517 14037]]
'''

#교차검증
from sklearn.model_selection import cross_val_score
# 5겹 교차검증
scores = cross_val_score(log_reg, x_train, y_train, cv=5) # model, train, target, cross validation
print('cross-val-score \n{}'.format(scores))
[0.56962068 0.56423114 0.563178   0.56641878 0.56447431]
print('cross-val-score.mean \n{:.3f}'.format(scores.mean())) #0.566

# 2019데이터 적용
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT_19.csv",encoding='CP949')
AFSNT.columns[:]
AFSNT=AFSNT.drop(columns=['Unnamed: 0','SDT_YY','ARP','ODP'])
x_test=AFSNT.drop('DLY', 1)
y_test=AFSNT['DLY']
y_pred = log_reg.predict(x_test)
print('정확도 :', metrics.accuracy_score(y_test, y_pred)) # 정확도 : 0.5318297908267725
cnf_metrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_metrix)
'''
[[89386 82148]
 [ 6574 11400]]
''' 
print(classification_report(y_test,y_pred))
'''
              precision    recall  f1-score   support

           0       0.93      0.52      0.67    171534
           1       0.12      0.63      0.20     17974

    accuracy                           0.53    189508
   macro avg       0.53      0.58      0.44    189508
weighted avg       0.85      0.53      0.62    189508
'''

#%%
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT_U.csv",encoding='CP949')
AFSNT.head()
len(AFSNT) 
AFSNT.columns[:]
AFSNT=AFSNT.drop(columns=['Unnamed: 0','SDT_YY'])

# xgboost
from xgboost import plot_importance
from xgboost import XGBClassifier
x = AFSNT.drop('DLY', 1)
y=AFSNT['DLY']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(len(x_train), len(x_test), len(y_train), len(y_test))
xgb=XGBClassifier(n_estimators=300, learning_rate=0.1,max_depth=8)
xgb.fit(x_train,y_train)
xgb_pred=xgb.predict(x_test)
print('정확도 :', metrics.accuracy_score(y_test, xgb_pred)) # 정확도 : 0.6964059110074092
print(classification_report(y_test,xgb_pred))
'''
              precision    recall  f1-score   support

           0       0.70      0.67      0.69     24304
           1       0.69      0.72      0.71     24554

    accuracy                           0.70     48858
   macro avg       0.70      0.70      0.70     48858
weighted avg       0.70      0.70      0.70     48858
'''

#교차검증
from sklearn.model_selection import cross_val_score
# 5겹 교차검증
scores = cross_val_score(xgb, x_train, y_train, cv=5) # model, train, target, cross validation
print('cross-val-score \n{}'.format(scores))
#[0.69095375 0.68941875 0.6849287  0.68776011 0.69304769]
print('cross-val-score.mean \n{:.3f}'.format(scores.mean())) #0.689

#변수중요도
fig,ax=plt.subplots()
plot_importance(xgb,ax=ax)

#2019데이터 적용
AFSNT=pd.read_csv("C:/Users/shimeunyoung/Desktop/AFSNT_19.csv",encoding='CP949')
AFSNT.columns[:]
AFSNT=AFSNT.drop(columns=['Unnamed: 0','SDT_YY','ARP','ODP'])
len(AFSNT.columns[:]) 
AFSNT.head()
x_test=AFSNT.drop('DLY', 1)
y_test=AFSNT['DLY']
xgb_pred=xgb.predict(x_test)
print('정확도 :', metrics.accuracy_score(y_test, xgb_pred)) #정확도 :0.5794214492264179
print(classification_report(y_test,xgb_pred))
'''
              precision    recall  f1-score   support

           0       0.95      0.56      0.71    171534
           1       0.15      0.74      0.25     17974

    accuracy                           0.58    189508
   macro avg       0.55      0.65      0.48    189508
weighted avg       0.88      0.58      0.66    189508
'''

