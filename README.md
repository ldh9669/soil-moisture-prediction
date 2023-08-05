Soil Moisture Prediction
===========
해당 프로젝트는 머신러닝을 사용하여 토양의 수분을 예측하기 위해 진행한 프로젝트입니다.

I. 프로젝트 개요
---------
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

*
