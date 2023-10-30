Soil Moisture Prediction
===========
해당 프로젝트는 머신러닝을 사용하여 토양의 수분을 예측하기 위해 진행한 프로젝트입니다.

### 설명

사용언어 : Python

작업툴 : Jupyter Notebook

사용 라이브러리 : pandas, numpy, sklearn, xgboost, lightgbm , 기타등등

인원 : 1명

기간 : 2022.12.28 ~ 2023.02.14

I. 프로젝트 개요
--------
1. 프로젝트 배경 및 목적
* 노지 스마트팜에서 현재 기온과 강수량, 습도 등으로 관수의 양을 조절해 땅속의 습도를 최적의 상태로 제어하여 수확량을 극대화한다.
* 토양의 수분을 측정할 때 센서의 길이가 길어질수록 제작 단가가 상승하기 때문에 범위를 넘어서는 수분은 머신러닝을 통한 예측으로 제작 비용의 절감을 기대한다.

2. 데이터셋 소개

* 농업기상 데이터를 기상청 기상자료개방포털에서 가져온다.
* 해당 데이터는 전국 9개의 관측소(대곡, 보성, 수원, 안동옥동, 오창가곡, 익산, 철원장흥, 춘천신북, 화순능주)에서 측정된 1시간 간격의 기상 데이터이다.
* 측정한 항목은 습도, 기온, 풍속, 지면온도, 지중온도, 토양수분, 누적복사 등 세부적인 항목이 존재하고 2012년부터 2021년까지 약 10년치의 데이터를 사용한다.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 455880 entries, 0 to 455879
Data columns (total 21 columns):
 #   Column             Non-Null Count   Dtype  
---  ------             --------------   -----  
 0   일시                 455880 non-null  object 
 1   50CM 정시 토양수분(%)    455880 non-null  float64
 2   0.5M 정시 습도(%)      455880 non-null  float64
 3   1.5M 정시 습도(%)      455880 non-null  float64
 4   4.0M 정시 습도(%)      455880 non-null  float64
 5   10CM 정시 토양수분(%)    455880 non-null  float64
 6   20CM 정시 토양수분(%)    455880 non-null  float64
 7   30CM 정시 토양수분(%)    455880 non-null  float64
 8   0.5M 정시 기온(°C)     455880 non-null  float64
 9   1.5M 정시 기온(°C)     455880 non-null  float64
 10  4.0M 정시 기온(°C)     455880 non-null  float64
 11  1.5M 정시 풍속(m/s)    455880 non-null  float64
 12  4.0M 정시 풍속(m/s)    455880 non-null  float64
 13  정시 지면온도(°C)        455880 non-null  float64
 14  5CM 정시 지중온도(°C)    455880 non-null  float64
 15  10CM 정시 지중온도(°C)   455880 non-null  float64
 16  20CM 정시 지중온도(°C)   455880 non-null  float64
 17  30CM 정시 지중온도(°C)   455880 non-null  float64
 18  정시 누적 순복사(MJ/m2)   455880 non-null  float64
 19  정시 누적 전천복사(MJ/m2)  455880 non-null  float64
 20  정시 누적 반사복사(MJ/m2)  455880 non-null  float64
dtypes: float64(20), object(1)
memory usage: 73.0+ MB
```

Train data: 307719 rows / validation data: 102573 rows / Test data: 45588 rows // Total data: 455880 rows

II. 전처리
--------
1. 보간법

* 몇 시간 단위부터 길게는 10일 이상 누락된 데이터가 존재하며 연속적으로 6시간까지 공백이 발생한 경우에는 공백의 시작 직전시간과 마지막 직후시간의 평균값으로 복원한다.

2. 스케일링

* Standard Scaler 사용
  
III. 모델별 실험
--------
* 사용모델: Random Forest, XGB, LGBM, SVR 회기모델
* 성능지표: MAE, MSE, RMSE, r2 score
* 지표별 특징
  1) MAE (Mean Absolute Error)
실제 값과 예측값의 차이를 절댓값으로 변환해 평균한 것

  2) MSE (Mean Squared Log Error)
실제 값과 예측값의 차이를 제곱해 평균한 것

  3) RMSE (Root Mean Squared Eerror)
MSE 같은 오류의 제곱을 구할때 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 씌운 것

  4) R 제곱
분산 기반으로 예측 성능을 평가한다. 실제 값의 분싼 대비 예측값의 분산 비율을 지표로 하며, 1에 가까울 수록 예측 정확도가 높다. 

실험 시 SVR 모델도 학습을 시켰지만 너무나 많은 시간 소요로인해 실험에서 배제시켰다.
```
  * 모델별 주요 지표 비교

  Model : Random Forest
  val  RMSE:  0.151 , r2 score:  0.977
  Test RMSE:  0.148 , r2 score:  0.976
  ------------------------------------
  Model : XGB
  val  RMSE:  0.14 ,  r2 score:  0.98
  Test RMSE:  0.131 , r2 score:  0.982
  ------------------------------------
  Model : LGBM
  val  RMSE:  0.155 , r2 score:  0.976
  Test RMSE:  0.149 , r2 score:  0.976
  ```

세 가지 모델이 거의 비슷한 성능을 보여줬지만 학습시간에서 Random Forest 모델은 697초, XGB 모델은 3070초, LGBM 모델은 62초의 성능으로
LGBM 모델이 압도적으로 짧은 학습시간을 보여줘 선정하게 되었다.

IV. 오차범위 검증
--------
* 데이터 전처리 과정 중 누락된 측정 값이 너무 많은 세트는 학습에서 제외시켰다.
* 학습에 노출되지않은 세트를 실제 토양 습도와 예측값의 오차범위가 어느 정도인지 판단하기 위해 비교과정을 진행했다.
* 학습에 제외된 데이터 세트는 9개로 그 중 모든 항목이 온전한 테이블을 대상으로 시간별 테이블을 세트당 500개씩 무작위로 선정하여 50CM 정시 토양수분 실제값과 예측값을 비교한다.

```
오차범위 평균 : 2.709849213987809
무작위 검사 시행개수 : 3117개

오차범위 최소 : 0.0
        최대  : 29.2

오차범위의 평균값 초과 개수 : 777개

평균을 초과하는 값들의 최소 : 2.8
                      최대 : 29.2
약 75%의 예측값들이 평균적으로 2.7정도 실제값과 차이를 보이고 있다.
```
![download](https://github.com/ldh9669/soil-moisture-prediction/assets/98334298/68ec5ef4-b631-41f1-b7a9-09fff6738af9)

![download](https://github.com/ldh9669/soil-moisture-prediction/assets/98334298/d74236fa-c95f-4913-92ca-4cfb0fba9784)

