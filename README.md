# Anomaly Detection
시계열 데이터에 대한 이상치 탐지
<br><br><br>
## 1. Kernel Density Estimation을 활용한 이상치 탐지
- *train_data_path*와 *test_data_path*에 존재하는 시점 정보를 포함하고 있는 csv 형태의 train data와 test data를 input으로 사용함<br>
- Train data로 kernel density estimation 모델을 적합하여 정상 데이터의 분포를 추정함<br>
- 추정된 분포를 기반으로 test data의 각 시점에 대한 anomaly score를 도출하고 이를 csv 파일 및 그래프로 *save_root_path*에 저장함<br>

```c
python kde.py --train_data_path='./data/nasa_bearing_train.csv' \
              --test_data_path='./data/nasa_bearing_test.csv' \
              --save_root_path='./result/kde'
```
<br><br>
## 2. Local Outlier Factor를 활용한 이상치 탐지
- *train_data_path*와 *test_data_path*에 존재하는 시점 정보를 포함하고 있는 csv 형태의 train data와 test data를 input으로 사용함<br>
- Train data로 Local Outlier Factor 모델을 적합하여 *n_neighbors* 개수의 이웃을 기반으로 정상 데이터의 밀도를 추정함<br>
- 추정된 밀도를 기반으로 test data의 각 시점에 대한 anomaly score를 도출하고 이를 csv 파일 및 그래프로 *save_root_path*에 저장함<br>

```c
python lof.py --train_data_path='./data/nasa_bearing_train.csv' \
              --test_data_path='./data/nasa_bearing_test.csv' \
              --save_root_path='./result/lof' \
              --n_neighbors=5
```
<br><br>
## 3. Isolation Forest를 활용한 이상치 탐지
- *train_data_path*와 *test_data_path*에 존재하는 시점 정보를 포함하고 있는 csv 형태의 train data와 test data를 input으로 사용함<br>
- Train data로 isolation forest 모델을 적합함<br>
- Train data를 reference set으로 사용하여 test data의 각 시점에 대한 anomaly score를 도출하고 이를 csv 파일 및 그래프로 *save_root_path*에 저장함<br>

```c
python iforest.py --train_data_path='./data/nasa_bearing_train.csv' \
                  --test_data_path='./data/nasa_bearing_test.csv' \
                  --save_root_path='./result/iforest'
                  
<br><br>
## 4. Spectral Residual을 활용한 이상치 탐지
- 설정된 window size 와 score window size 를 통해 window 구간 내 이상치를 탐지함<br>
- score window size 는 window size 보다 크게 설정해야함

```c
python spectral.py --window= 24 \
                  --score_window=100 
