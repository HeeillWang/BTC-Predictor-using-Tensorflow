# BTC-Predictor-with-Tensorflow
# 텐서플로우를 이용한 비트코인 가격 예측

본 프로그램은 Python3와, 파이선 라이브러리 numpy, tensorflow, requests가 설치되어 있음을 전제로 한다.

    python3 controller.py

controller.py를 실행시킴으로 유저 인터페이스를 실행할 수 있다.

controller.py에는 총 4가지의 선택지가 주어진다.

0 - 프로그램을 종료한다.

1 - 현재 시점까지의 가격 정보를 수집한다.

2 - 학습을 진행한다.

3 - 예측을 진행한다.

1. 가격 정보 수집
코드는 ./Crawler/crawler_bithumb.py 와 ./Crawler/crawler_upbit.py에 존재한다. 가격 정보는 1시간 단위로 수집되며, 단위는 USD이다. 이 전에 수집된 data.csv파일에 이어서 수집하게 된다. 만약 data.csv가 존재하지 않는다면 2013년 6월 1일 0시 0분 이후의 가격부터 수집한다. 해당하는 날짜는 ./Crawler/crawler_bithumb.py에서 변경할 수 있다. 

2. 학습
코드는 ./RNNs/train.py 에 존재한다. 또한 학습된 모델들은 ./RNNs/saved/에 저장된다. 사용자는 저장된 모델을 불러와서 학습을 진행하거나, 새로운 모델을 생성하며 학습을 진행할 수 있다. 또한 training epoch를 직접 입력하여 원하는 만큼 학습을 진행할 수 있다. epoch에 0을 입력하면 학습이 진행되지 않고, test data의 정확도만 출력해준다. 학습이 완료된 모델은 자동적으로 저장된다.

3. 예측
코드는 ./RNNs/predict.py 에 존재한다. 1.에서 수집된 data.csv를 기반으로, 바로 다음 timestamp의 가격을 예측하여 출력한다.
