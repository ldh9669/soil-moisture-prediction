Soil Moisture Prediction
===========
해당 프로젝트는 머신러닝을 사용하여 토양의 수분을 예측하기 위해 진행한 프로젝트입니다.

### 설명

사용언어 : Python

작업툴 : Jupyter Notebook

사용 라이브러리 : pandas, numpy, sklearn, xgboost, lightgbm

인원 : 1명

기간 : 2022.12.28 ~ 2023.02.14

I. 프로젝트 개요
--------
1. 프로젝트 배경 및 목적
##### 농사를 지을 때 수확량에 영향을 끼치는 환경적인 요인들이 다수 존재한다. 수확량을 극대화 하기위해 사람이 통제 할 수 있는 요인은 땅속의 습도를 제어하는 것이다.
##### 이것을 노지 스마트팜을 사용하여 농사를 지음으로 현재 기온과 강수량, 습도등으로 관수의 양을 조절해 최대의 수확량을 내기위한 최적의 토양 습도를 조절한다.
##### 토양의 수분을 측정할 때는 실물 센서를 사용하여 측정한다. 센서의 길이가 길어질수록 제작 단가가 상승하기 때문에 측정할 수 있는 깊이를 넘어서는 수분은 머신러닝을 통한 예측으로 제작 비용의 절감을 기대한다.

2. 데이터셋 소개

* 농업기상 데이터를 기상청 기상자료개방포털에서 가져온다.
* 해당 데이터는 전국 9개의 관측소(대곡, 보성, 수원, 안동옥동, 오창가곡, 익산, 철원장흥, 춘천신북, 화순능주)에서 측정된 1시간 간격의 기상 데이터이다.
* 측정한 항목은 습도, 기온, 풍속, 지면온도, 지중온도, 토양수분, 누적복사 등등 다양한 항목이 존재하고 2012년부터 2021년까지 약 10년치의 데이터를 사용한다.
* 기상 데이터의 특성상 온전하게 기록된 데이터가 없다보니 누락된 데이터가 많은 년도의 경우 해당 년도의 데이터는 제외시켰다.

Train data: 341910 rows / Test data: 113970 rows

II. 전처리
--------
1. 보간법

* 많은 항목들이 적게는 몇 시간 단위부터 길게는 10일 이상 데이터가 누락된 경우가 빈번하다. 이 누락된 데이터가 연속적으로 6시간까지 공백이 발생한 경우에는 기준점으로부터 앞 뒤시간의 평균값으로 복원한다.
* 연속적으로 6시간이 넘는 장시간의 결측치에는 각각 해당 시간대의 앞뒤로 3일치의 평균값으로 결정한다.
* 예를들어 2월 5일 10시 부터 2월 6일 10시까지 24시간이 누락됬다면 2월 5일 10시는 해당 시간대의 앞 2일, 3일 ,4일 뒤로 7일, 8일, 9일의 평균값으로 결정한다.

2. 스케일링

* StandardScaler 사용
  
III. 모델별 실험
--------
* 사용모델: Random Forest, XGB, LGBM, SVR
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

IV. 검증
--------
