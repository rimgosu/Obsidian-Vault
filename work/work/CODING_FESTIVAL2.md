
문제 1. 울타리
```
# 입력 받기
N, M, K = map(int, input().split())
carrots = []

# 당근의 위치 저장
for _ in range(K):
    x, y = map(int, input().split())
    carrots.append((x, y))

# 각 변의 길이 초기화

left = N # x축이 가장 낮은값
right = 0 # x축이 가장 높은값
top = 0 # y축이 가장 높은값
bottom = M # y축이 가장 낮은값

# 각 당근의 위치를 반올림하여 울타리의 모서리 좌표 업데이트
for x, y in carrots:
    left = min(left, x)
    right = max(right, x)
    top = max(top, y)
    bottom = min(bottom, y)
# 울타리의 네 변의 길이 계산
length = ((right+1) - (left-1)) * 2 +  ((top+1) - (bottom-1)) * 2

# 결과 출력
print(length)
```


2. 룰렛 게임
```
N, K, P, L = map(int, input().split())
player_numbers = [[] for _ in range(K)]  # 각 플레이어가 외친 숫자 저장

# 각 플레이어가 외친 숫자 입력
for i in range(K):
    player_numbers[i] = list(map(int, input().split()))

current_position = 1  # 원판의 현재 위치 (화살표가 가리키는 위치)
current_round = 1  # 현재 라운드

winner = -1  # 승리한 플레이어의 번호, 초기에는 아무도 승리하지 않음

while current_round <= L:
    for i in range(K):
        current_number = player_numbers[i][current_round - 1]  # 현재 플레이어가 외친 숫자

        for _ in range(current_number):
            current_position -= 1
            if current_position < 1:
                current_position = N
        if current_position == P:
            winner = i + 1
            break  # 승자를 찾았으므로 더 이상 루프를 실행하지 않음
    if winner != -1:  # 승자를 찾았을 경우 루프 종료
        break
    current_round += 1

if winner != -1:
    print(winner, current_round)
else:
    print(-1)
```

3. 배열 더하기
```
import numpy as np

# 첫 번째 배열 입력 받기
array1 = np.array(list(map(int, input().split(','))))
array1 = array1.reshape((2, 6))  # 2x6 형태로 변환

# 두 번째 배열 입력 받기
array2 = np.array(list(map(int, input().split(','))))
array2 = array2.reshape((1, 6))  # 1x6 형태로 변환

# 두 배열을 브로드캐스팅을 사용하여 더하기
result = array1 + array2

# 결과 출력
print(result)
```




4. OECD 휘발유 가격 그래프

```
import matplotlib.pyplot as plt
import pandas as pd
from elice_utils import EliceUtils

fig, ax = plt.subplots()

# 1. OECDGas.csv 파일을 pandas 라이브러리를 이용하여 불러옵니다.
df = pd.read_csv("OECDGas.csv")

# 2. France의 year와 price 데이터를 추출합니다.
france_data = df[df['country'] == 'France']

# 3. x 축: country가 "France"인 경우의 year로 설정합니다.
#    y 축: country가 "France"인 경우의 price로 설정합니다.
ax.plot(france_data['year'], france_data['price'], marker='o', linestyle='-')

# 4. 그래프의 제목, x축 이름, y축 이름 설정
ax.set_title("Gasoline Prices in France over Years")
ax.set_xlabel("Year")
ax.set_ylabel("Price")

# 그래프를 저장하고 Elice 환경에 출력합니다. 수정하지 마세요!
fig.savefig("plot.png")
EliceUtils().send_image("plot.png")

```




5. OECD 자동차 재고 그래프
```
import matplotlib.pyplot as plt
import pandas as pd
from elice_utils import EliceUtils

fig, ax = plt.subplots()

# 1. OECDGas.csv 파일을 pandas 라이브러리를 이용하여 불러옵니다.
data = pd.read_csv("OECDGas.csv")

# 2. country가 "Turkey"인 경우의 year와 cars 데이터를 선택합니다.
turkey_data = data[data['country'] == "Turkey"]
x = turkey_data['year']
y = turkey_data['cars']

# 3. 꺾은선 그래프를 그립니다.
ax.plot(x, y, label="Cars in Turkey", marker='o', linestyle='-')
ax.set_xlabel('Year')
ax.set_ylabel('Cars per person')
ax.set_title('Number of Cars per person in Turkey over the years')
ax.legend()

# 그래프를 저장하고 Elice 환경에 출력합니다. 수정하지 마세요!
fig.savefig("plot.png")
EliceUtils().send_image("plot.png")

```




6. 와인 데이터 EDA
```
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def load_csv(path):
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 불러오는 함수입니다."""
    df = pd.read_csv(path)
    return df
def make_bar(df: pd.DataFrame):
    """matplotlib 라이브러리를 이용해 막대그래프를 그리는 함수입니다."""
    x = df['quality'].unique() # 와인의 품질
    x.sort() # 정렬
    y = df.groupby('quality')['alcohol'].mean() # 와인의 품질 별 알콜 도수의 평균

    plt.bar(x, y)
    plt.xlabel('Wine Quality')
    plt.ylabel('Average Alcohol Content')
    plt.title('Average Alcohol Content by Wine Quality')
    
    show_plot()
    return x, y

def make_scatter(df: pd.DataFrame):
    """matplotlib 라이브러리를 이용해 산점도를 그리는 함수입니다."""
    x = df['alcohol'] # 와인의 알콜 도수
    y = df['density'] # 와인의 비중

    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Alcohol Content')
    plt.ylabel('Density')
    plt.title('Scatter Plot of Alcohol Content vs. Density')
    
    show_plot()
    return x, y


def show_plot():
    """그래프를 나타내는 함수입니다."""
    plt.savefig("plot1.png")
    elice_utils.send_image("plot1.png")
    plt.cla()


def main():
    data_path = "data/wine.csv"

    df = load_csv(data_path)

    make_bar(df)

    make_scatter(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())

```




7. 데이터 클러스터링
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from elice_utils import elice_utils


def load_data() -> pd.DataFrame:
    iris = load_iris()
    df_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_data["target"] = iris.target
    return df_data

def k_means_clus(df_data: pd.DataFrame) -> pd.DataFrame:
    """df_data를 K-means 클러스터링을 이용해 군집화합니다."""
    df_train = df_data.drop("target", axis=1)
    
    # KMeans 모델 생성
    kmeans = KMeans(init="random", n_clusters=3, random_state=100)
    
    # 모델 학습
    kmeans.fit(df_train)
    
    # df_data에 cluster 컬럼 추가
    df_data["cluster"] = kmeans.labels_
    
    return df_data

def pca_trans(df_data: pd.DataFrame, n_components: int) -> np.ndarray:
    """df_data를 PCA를 이용해 n_components만큼 차원 축소합니다."""
    
    # PCA 모델 생성
    pca = PCA(n_components=n_components)
    
    # 데이터 변환
    pca_transformed = pca.fit_transform(df_data)
    
    return pca_transformed


def visualize(df_data: pd.DataFrame):
    "결과를 시각화합니다."
    pca_transformed = pca_trans(df_data, 2)

    df_data["pca_x"] = pca_transformed[:, 0]
    df_data["pca_y"] = pca_transformed[:, 1]

    # 군집된 값이 0, 1, 2 인 경우, 인덱스 추출
    idx_0 = df_data[df_data["cluster"] == 0].index
    idx_1 = df_data[df_data["cluster"] == 1].index
    idx_2 = df_data[df_data["cluster"] == 2].index

    # 각 군집 인덱스의 pca_x, pca_y 값 추출 및 시각화
    fig, ax = plt.subplots()

    ax.scatter(x=df_data.loc[idx_0, "pca_x"], y=df_data.loc[idx_0, "pca_y"], marker="o")
    ax.scatter(x=df_data.loc[idx_1, "pca_x"], y=df_data.loc[idx_1, "pca_y"], marker="s")
    ax.scatter(x=df_data.loc[idx_2, "pca_x"], y=df_data.loc[idx_2, "pca_y"], marker="^")

    ax.set_title("K-menas")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")


def main():
    # 데이터 불러오기
    df_data = load_data()

    # K-means 클러스터링 수행
    df_data = k_means_clus(df_data)

    # 시각화
    visualize(df_data)


if __name__ == "__main__":
    main()

```


8. 랜덤 포레스트 학습

```
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def load_data(path: str) -> (pd.DataFrame, pd.Series):
    """데이터를 DataFrame의 형태로 불러와 price를 제외한 모든 column을 x로, price column을 y로 반환합니다."""
    df = pd.read_csv(path)
    x = df.drop(labels=["price"], axis=1)
    y = df["price"]
    return x, y

def select_model() -> GridSearchCV:
    """DecisionTreeRegressor 모델을 위한 최적의 하이퍼파라미터를 찾기 위한 GridSearchCV객체를 반환합니다."""
    estimator = DecisionTreeRegressor(random_state=40)
    param_grid = {'max_depth': [3, 5, 10]}
    grid_search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
    )
    return grid_search

def train_model(model: GridSearchCV, x_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeRegressor:
    """model, x_train, y_train을 입력받아 모델을 학습시키고 가장 좋은 모델을 반환합니다."""
    model.fit(x_train, y_train)
    best_model = model.best_estimator_
    return best_model

def evaluate_model(model: DecisionTreeRegressor, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """model, x_test, y_test를 입력받아 MSE를 계산하여 반환합니다."""
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    return mse

# 학습 데이터 경로
TRAIN_DATA = "data/converteddata.csv"

# 테스트 데이터 경로
TEST_DATA = "data/test.csv"

def main():
    # 데이터를 x, y로 나누어 불러옵니다.
    x_train, y_train = load_data(TRAIN_DATA)
    x_test, y_test = load_data(TEST_DATA)

    # 1. 최적의 모델을 찾기 위한 GridSearchCV 객체를 생성합니다.
    model = select_model()

    # 2. 모델을 학습시킵니다.
    model = train_model(model, x_train, y_train)

    # 3. 모델을 평가합니다.
    mse = evaluate_model(model, x_test, y_test)
    print("MSE:", mse)

if __name__ == "__main__":
    main()

```



9. 레드와인 VS 화이트와인

```
import sys

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option("mode.chained_assignment", None)


def load_csv(path):
    """pandas를 이용하여 데이터를 DataFrame의 형태로 불러와 반환하는 함수입니다."""
    df = pd.read_csv(path)
    return df


def label_encoding(series: pd.Series):
    """범주형 데이터를 시리즈형태로 받아 숫자형 데이터로 변환하는 함수입니다."""
    series = series.map({"red": 0, "white": 1})
    return series


def divide_data(df: pd.DataFrame):
    """학습용 데이터와 검증용 데이터를 나눠 X_train, X_test, y_train, y_test로 반환하는 함수입니다."""
    X = df.drop("type", axis=1)
    y = df["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def model_train(X_train, y_train):
    """학습용 데이터를 의사결정나무 모델을 이용해 학습시키고 학습된 모델을 반환하는 함수입니다."""
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def get_accuracy_score(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def main():
    # 데이터 주소
    data_path = "data/wine.csv"

    # load_csv 함수를 사용하여 데이터를 불러와 저장합니다.
    df = load_csv(data_path)

    # 데이터 변환
    df["type"] = label_encoding(df["type"])

    # 데이터 분리
    X_train, X_test, y_train, y_test = divide_data(df)

    # 모델 학습
    model = model_train(X_train, y_train)
    y_pred = model.predict(X_test)

    # 평가지표 계산
    score = get_accuracy_score(y_test, y_pred)
    print(f"Accuracy: {score:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

```





10. 다양한 최적화 알고리즘을 적용하고 성능 높여보기

```
import numpy as np
import tensorflow as tf
from visual import *
from tensorflow.keras.layers import Dropout, BatchNormalization, GlobalAveragePooling2D

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(100)

'''
1. 다양한 최적화 알고리즘들을 적용할 하나의 모델을 자유롭게 생성합니다.
'''

def OUR_model():
    model = tf.keras.Sequential([
        
        tf.keras.layers.Flatten(input_shape=(784,)), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


'''
2. 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. OUR_model 함수를 이용해 모델을 불러옵니다.
   
   Step02. 모델의 손실 함수, 최적화 알고리즘, 평가 방법을 설정합니다.
   
   Step03. 모델을 각각 학습시킵니다. 검증용 데이터도 설정해주세요.
           모델의 'epochs'는 20, 'batch_size'는 500으로 설정합니다.
   
   Step04. 학습된 모델을 테스트하고 sparse categorical crossentropy
           값을 출력합니다. 모델의 성능을 확인해보고, 목표값을 달성해보세요.
'''

def main():
    
    # Fashion mnist data 를 load 합니다.
    train_data = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    train_labels = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    test_data = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    test_labels = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)
    
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    
    our_model = OUR_model()
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) # You can start with Adam and adjust learning rate if necessary
    our_model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    history = our_model.fit(train_data, train_labels, 
                            epochs=20, 
                            batch_size=500, 
                            validation_data=(test_data, test_labels))
    
    scores = our_model.evaluate(test_data, test_labels)
    
    print('\nscores: ', scores[-1]) # Printing sparse_categorical_crossentropy
    
    Visualize([('our_model', history)])
    
    return history

if __name__ == "__main__":
    main()
```