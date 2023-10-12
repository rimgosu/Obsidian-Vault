

1번
```
N = int(input())
weights = list(map(int, input().split()))

# 각 칸과 그 칸의 양 옆 칸의 금의 무게 합을 계산
sums = [weights[0] + weights[1]]
for i in range(1, N-1):
    sums.append(weights[i-1] + weights[i] + weights[i+1])
sums.append(weights[N-2] + weights[N-1])

# 두 손으로 선택할 수 있는 조합 중에서 최대의 합을 찾기
max_sum = 0
for i in range(N):
    for j in range(i+3, N):
        if sums[i] + sums[j] > max_sum:
            max_sum = sums[i] + sums[j]

print(max_sum)

```



2번
```
def is_mountain(heights):
    n = len(heights)
    peak_index = heights.index(max(heights))  # 가장 높은 판자의 위치
    left = heights[:peak_index]  # 산 모양 왼쪽 부분
    right = heights[peak_index + 1:]  # 산 모양 오른쪽 부분

    # 왼쪽 부분은 오름차순, 오른쪽 부분은 내림차순이어야 함
    if left == sorted(left) and right == sorted(right, reverse=True):
        return "Yes"
    else:
        return "No"

# 테스트 케이스 개수 입력
t = int(input())
results = []

for _ in range(t):
    # 판자의 개수 입력
    n = int(input())

    # 각 판자의 세로 길이 입력
    heights = list(map(int, input().split()))

    # 판자 배열이 "Mountain 꼴"을 만족하는지 확인하고 결과를 리스트에 저장
    results.append(is_mountain(heights))

# 모든 결과 출력
for result in results:
    print(result)
```


3번.

```
import numpy as np

# 입력받은 문자열을 ,로 구분하여 int로 변환한 후, NumPy 배열로 만든다.
arr = np.array(list(map(int, input().split(','))))

# 입력받은 배열을 4x5 형태로 바꾼다.
arr = arr.reshape(4, 5)

# 첫 번째 행을 기준으로 나눈다.
arr1 = arr[0:1, :]
arr2 = arr[1:, :]

# 결과 출력
print(arr1)
print(arr2)

```


4번

```
import pandas as pd

# iris.csv 파일 불러오기
df = pd.read_csv("iris.csv")

# 사용자 입력값 받기
num_of_rows = int(input())
order = int(input())
column_name = input()

# 오름차순/내림차순 정렬
if order == 1:
    sorted_df = df.sort_values(by=column_name, ascending=True)
else:
    sorted_df = df.sort_values(by=column_name, ascending=False)

# 결과 출력
print(sorted_df.head(num_of_rows))

```



5번

```
import matplotlib.pyplot as plt
import pandas as pd
from elice_utils import elice_utils

def load_csv(path) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 불러오는 함수입니다."""
    df = pd.read_csv(path)
    return df

def make_box(df: pd.DataFrame):
    """matplotlib.pyplot 라이브러리를 이용해 혈중 나트륨 농도의 박스 그래프를 그립니다."""
    box = plt.boxplot(df['serum_sodium'])
    show_plot("box")
    return box

def make_scatter(df: pd.DataFrame):
    """matplotlib.pyplot 라이브러리를 이용해 나이와 혈소판 수치의 산점도 그래프를 그립니다."""
    x = df['age']
    y = df['platelets']
    scatter = plt.scatter(x, y)
    show_plot("scatter")
    return x, y

def make_hist(df: pd.DataFrame):
    """matplotlib.pyplot 라이브러리를 이용해 사망자들의 나이의 히스토그램을 그립니다."""
    df_die = df[df['DEATH_EVENT'] == True]
    hist = plt.hist(df_die['age'], bins=15, histtype="bar")
    show_plot("hist")
    return hist

def show_plot(fig_name: str):
    """그래프를 보여주는 함수입니다."""
    plt.savefig(fig_name + ".png")
    elice_utils.send_image(fig_name + ".png")
    plt.cla()

def main():
    # 데이터 경로
    data_path = "data/heart.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 혈중 나트륨 농도의 박스 그래프 그리기
    make_box(df)

    # 나이와 혈소판 수치의 산점도 그래프 그리기
    make_scatter(df)

    # 사망자들의 나이의 히스토그램
    make_hist(df)

    return 0

if __name__ == "__main__":
    main()

```




6번
```
import pandas as pd

# 파일을 불러옵니다.
df = pd.read_csv('exercise.csv')

# 입력을 받습니다.
start_row = int(input())
end_row = int(input())
sort_column = input()

# 입력받은 컬럼 명을 기준으로 오름차순 정렬합니다.
df_sorted = df.sort_values(by=sort_column, ascending=True)

# 정렬된 데이터를 입력받은 행 번호를 기준으로 출력합니다.
print(df_sorted.iloc[start_row:end_row])

```


7번
```
import pandas as pd
from imblearn.over_sampling import SMOTE

def load_csv(path: str) -> pd.DataFrame:
    """pandas를 이용하여 path의 csv파일을 DataFrame의 형태로 불러와 반환합니다."""
    df = pd.read_csv(path)
    return df

def get_counts(df: pd.DataFrame) -> pd.Series:
    """df에서 quality의 값이 각각 몇개씩 있는지 구하여 반환합니다."""
    counts = df["quality"].value_counts()
    return counts

def over_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """SMOTE기법의 사용하여 데이터를 오버 샘플링하여 반환합니다."""
    # 데이터 분리
    features = df.columns.drop("quality")
    df_feat = df[features]
    y = df["quality"]

    # 샘플러 정의
    sampler = SMOTE(k_neighbors=4)

    # 오버 샘플링
    df_feat, y = sampler.fit_resample(df_feat, y)
    df_feat["quality"] = y
    return df_feat

def transform_type(text: str):
    """text가 white면 0을 그렇지 않으면 1을 반환합니다."""
    if text == "white":
        return 0
    else:
        return 1

def main():
    # 데이터 주소
    DATA_PATH = "data/wine.csv"

    # 데이터 로드
    df = load_csv(DATA_PATH)

    # 데이터 전처리
    df["type"] = df["type"].map(transform_type)

    # 1. 데이터 수 표시하기
    print("처리 전")
    print(get_counts(df))

    # 2. 오버 샘플링
    df_new = over_sampling(df)

    # 데이터 수 표시하기
    print("처리 후")
    print(get_counts(df_new))

if __name__ == "__main__":
    main()

```


8번
```
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

pd.set_option("mode.chained_assignment", None)


def load_csv(path):
    """pandas를 이용하여 데이터를 DataFrame의 형태로 불러와 반환하는 함수입니다."""
    df = pd.read_csv(path)
    return df


def label_encoding(series: pd.Series):
    """범주형 데이터를 시리즈형태로 받아 숫자형 데이터로 변환하는 함수입니다."""
    my_dict = {}
    for i in range(len(sorted(series.unique()))):
        my_dict[sorted(series.unique())[i]] = i
    series = series.map(my_dict)
    return series


def divide_data(df: pd.DataFrame):
    """1. 학습용 데이터와 검증용 데이터를 나눠 X_train, X_test, y_train, y_test로 반환하는 함수입니다."""
    # features를 설정
    features = df.columns.drop("calories")
    
    X = df[features]
    y = df["calories"]

    # 데이터를 학습용과 검증용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def model_train(X_train, y_train):
    """2. 학습용 데이터를 의사결정나무 모델을 이용해 학습시키고 학습된 모델을 반환하는 함수입니다."""
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(y_test, y_pred):
    """3. 모델의 성능을 평가하는 함수입니다. """
    my_score = mean_absolute_error(y_test, y_pred)
    return my_score


def main():
    # 데이터 주소
    data_path = "data/fastfood.csv"

    # 데이터 로드
    df = load_csv(data_path)

    # 데이터 변환
    text_col = ['restaurant', 'item']
    for col in text_col:
        df[col] = label_encoding(df[col])

    # 데이터 분리
    X_train, X_test, y_train, y_test = divide_data(df)

    # 모델 학습
    model = model_train(X_train, y_train)
    y_pred = model.predict(X_test)

    # 평가지표 계산
    score = evaluate_model(y_test, y_pred)
    print(f"MAE: {score}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

```



9번

```
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option("mode.chained_assignment", None)

from elice_utils import EliceUtils

elice_utils = EliceUtils()


def load_csv(path):
    """pandas를 이용하여 데이터를 DataFrame의 형태로 불러와 반환하는 함수입니다."""
    None
    return None


def label_encoding(series: pd.Series):
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 반환하는 함수를 작성합니다."""
    
    def transform(text: str):
        if text == "Man":
            return 0
        else:
            return 1

    series = series.map(transform)
    return series


def divide_data(df: pd.DataFrame):
    """학습용 데이터와 검증용 데이터를 나눠 X_train, X_test, y_train, y_test로 반환하는 함수를 작성합니다."""
    None
    return X_train, X_test, y_train, y_test


def model_train(X_train, y_train):
    """학습용 데이터를 랜덤포레스트 모델을 이용해 학습시키고 학습된 모델을 반환하는 함수를 작성합니다."""
    model = None
    None
    return model
def load_csv(path):
    """pandas를 이용하여 데이터를 DataFrame의 형태로 불러와 반환하는 함수입니다."""
    df = pd.read_csv(path)  # CSV 파일을 불러옴
    return df

def divide_data(df: pd.DataFrame):
    """학습용 데이터와 검증용 데이터를 나눠 X_train, X_test, y_train, y_test로 반환하는 함수를 작성합니다."""
    # target을 y로 설정
    y = df['target']
    # X 데이터는 target을 제외한 모든 열
    X = df.drop('target', axis=1)
    # train_test_split 함수를 사용하여 데이터를 학습용과 검증용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def model_train(X_train, y_train):
    """학습용 데이터를 랜덤포레스트 모델을 이용해 학습시키고 학습된 모델을 반환하는 함수를 작성합니다."""
    model = RandomForestRegressor()  # 랜덤포레스트 모델 생성
    model.fit(X_train, y_train)  # 모델 학습
    return model


def main():

    # 데이터 주소
    DATA_PATH = "data/diabetes.csv"

    # load_csv 함수를 사용하여 데이터를 불러와 저장합니다.
    df = load_csv(DATA_PATH)

    # 데이터 변환
    df["sex"] = label_encoding(df["sex"])

    # 데이터 분리
    X_train, X_test, y_train, y_test = divide_data(df)

    # 모델 학습
    model = model_train(X_train, y_train)
    y_pred = model.predict(X_test)

    # 평가지표 계산
    score = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {score}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

```



10번

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from visual import *
from plotter import *
from dataloader import load_data

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(100)
tf.random.set_seed(100)

def Develop():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def main():
    # Fashion-MNIST 데이터를 불러오고 전처리하는 부분입니다.
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    develop_model = Develop()
    
    develop_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
    develop_model.summary()
    
    history = develop_model.fit(train_images, train_labels,
                                epochs=20, batch_size=500,
                                validation_data=(test_images, test_labels))
    
    _, accuracy_develop = develop_model.evaluate(test_images, test_labels)
    
    print('\naccuracy_develop: ', accuracy_develop)
    
    Visualize([('Develop', history)])
    
    return history

if __name__ == "__main__":
    main()

```


