### 1. 사무실 짓기

```
def office_location(N, houses):
    # Step 1: Sort the house locations
    houses.sort()

    # Step 2: Calculate the median
    if N % 2 == 1:
        # Step 3: If N is odd
        return 1
    else:
        # Step 4: If N is even
        mid1 = houses[N//2 - 1]
        mid2 = houses[N//2]
        return mid2 - mid1 + 1

# Input & Output
if __name__ == "__main__":
    N = int(input())
    houses = list(map(int, input().split()))
    print(office_location(N, houses))
```


### 2. 덱

```
def find_relation_index(s1, s2):
    m = len(s1)
    n = len(s2)

    # Initialize the DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest_length = 0

    # Fill the DP table and find the maximum length
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                longest_length = max(longest_length, dp[i][j])

    return m + n - (2 * longest_length)

m = input().strip()
n = input().strip()
print(find_relation_index(m,n))    
```


### 3. 좌표 그래프

```
# 초기 설정을 합니다. 수정하지 마세요!
import matplotlib.pyplot as plt
import pandas as pd
from elice_utils import EliceUtils

fig, ax = plt.subplots()

# anscombe.csv 파일 불러오기
df = pd.read_csv('anscombe.csv')

# dataset이 "II"인 경우의 데이터만 선택
data_II = df[df['dataset'] == 'II']

# 산점도 그리기
ax.scatter(data_II['x'], data_II['y'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scatter Plot for Dataset II')


# 그래프를 저장하고 Elice 환경에 출력합니다. 수정하지 마세요!
fig.savefig("plot.png")

EliceUtils().send_image("plot.png")
```



### 4. 패스트푸드 데이터 탐색

```
import pandas as pd


def load_csv(path) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 불러오는 함수입니다."""
    df = pd.read_csv(path)
    return df


def get_highest(df: pd.DataFrame) -> pd.DataFrame:
    """1. df에서 칼로리가 가장 높은 데이터를 찾아 데이터프레임의 형태로 반환합니다."""
    highest_calories = df['calories'].max()
    df_highest = df[df['calories'] == highest_calories]
    return df_highest


def get_calories_diff(df: pd.DataFrame) -> float:
    """2. df에서 식당(restaurant)이 Mcdonalds인 메뉴들과 Subway인 메뉴들의 평균 칼로리의 차이를 구하여 반환합니다."""
    mcdonalds_avg_calories = df[df['restaurant'] == 'Mcdonalds']['calories'].mean()
    subway_avg_calories = df[df['restaurant'] == 'Subway']['calories'].mean()
    calories_diff = abs(mcdonalds_avg_calories - subway_avg_calories)
    return calories_diff


def search_menu(df: pd.DataFrame) -> pd.DataFrame:
    """3. 아래의 조건을 모두 만족하는 메뉴 중에서 cholesterol이 가장 낮은 메뉴를 찾아 시리즈의 형태로 반환합니다."""
    filtered_df = df[(df['calories'] < 500) & (df['protein'] > 30)]
    lowest_cholesterol_menu = filtered_df[filtered_df['cholesterol'] == filtered_df['cholesterol'].min()]
    return lowest_cholesterol_menu.iloc[0]


def main():
    # 데이터 경로
    data_path = "data/fastfood.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 가장 칼로리가 높은 메뉴 찾기
    df_highest = get_highest(df)
    print("칼로리가 가장 높은 메뉴\n", df_highest)

    # 지시사항을 참고하여 특정 식당간의 평균 칼로리 차이 구하기
    calories_diff = get_calories_diff(df)
    print(" \n평균 칼로리 차이:", calories_diff)

    # 지시사항의 조건을 만족하는 메뉴 찾기
    my_menu = search_menu(df)
    print(" \n검색결과\n", my_menu)


if __name__ == "__main__":
    main()
```



### 5. 패스트푸드 데이터 시각화

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from elice_utils import elice_utils

def load_csv(path) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 불러오는 함수입니다."""
    df = pd.read_csv(path)
    return df

def make_hist(df: pd.DataFrame):
    """1. df의 'calories' 열을 이용하여 히스토그램을 그립니다."""
    plt.hist(df['calories'], bins=20)
    plt.xlabel('Calories')
    plt.ylabel('Frequency')
    plt.title('Histogram of Calories')
    show_plot("hist")

def make_boxplot(df: pd.DataFrame):
    """2. df의 'restaurant'를 x축으로, 'calories'를 y축으로 박스 그래프를 그립니다."""
    sns.boxplot(x='restaurant', y='calories', data=df)
    plt.xlabel('Restaurant')
    plt.ylabel('Calories')
    plt.title('Boxplot of Calories by Restaurant')
    plt.xticks(rotation=45)  # x축 레이블 회전
    show_plot("boxplot")

def make_heatmap(df: pd.DataFrame):
    """3. df의 상관관계를 구해 seaborn 라이브러리를 통해 히트맵을 그립니다."""
    df_corr = df.corr()
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df_corr, annot=False, square=True, fmt=".2f")

    show_plot("heatmap")
    return heatmap


def show_plot(fig_name: str):
    """그래프를 보여주는 함수입니다."""
    plt.savefig(fig_name + ".png")
    elice_utils.send_image(fig_name + ".png")
    plt.cla()

def main():
    # 데이터 경로
    data_path = "data/fastfood.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 칼로리 히스토그램 그리기
    make_hist(df)

    # 식당별 칼로리 박스플롯 그리기
    make_boxplot(df)

    # 상관관계 히트맵 그리기
    make_heatmap(df)

if __name__ == "__main__":
    main()
```




### 6. 불균형 데이터 처리


```
import pandas as pd
from typing import List

# 데이터 불러오기
df = pd.read_csv('data/heart.csv')

def get_counts(df: pd.DataFrame) -> List[int]:
    """DEATH_EVENT의 데이터 수를 False, True 순으로 출력하기"""
    counts = df['DEATH_EVENT'].value_counts().tolist()
    return counts

def under_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """언더샘플링 구현하기"""
    # DEATH_EVENT가 True인 데이터와 False인 데이터를 분리
    true_data = df[df['DEATH_EVENT'] == True]
    false_data = df[df['DEATH_EVENT'] == False]
    
    # DEATH_EVENT가 True인 데이터의 수
    true_count = len(true_data)
    
    # DEATH_EVENT가 False인 데이터에서 랜덤하게 true_count 개수만큼 샘플링
    false_sampled = false_data.sample(n=true_count, random_state=42)
    
    # 언더샘플링된 데이터프레임 생성
    under_sampled_df = pd.concat([false_sampled, true_data])
    
    return under_sampled_df

if __name__ == "__main__":
    # DEATH_EVENT의 데이터 수 출력
    counts_before = get_counts(df)
    print("처리 전:", counts_before)

    # 언더샘플링 수행
    under_sampled_df = under_sampling(df)
    counts_after = get_counts(under_sampled_df)
    print("처리 후:", counts_after)
```



### 7. 모델 성능 평가하기

```
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix

pd.set_option("mode.chained_assignment", None)  # 수정금지

def load_csv(path):
    """pandas를 이용하여 데이터를 DataFrame의 형태로 불러와 반환하는 함수입니다."""
    df = pd.read_csv(path)
    return df

def get_model() -> RandomForestClassifier:
    """학습이 완료된 모델을 불러와 반환합니다."""
    loaded_model = joblib.load("model.pkl")
    return loaded_model

def get_predict(x_test: pd.DataFrame) -> np.ndarray:
    """get_model로 불러온 함수에 x_test를 넣은 예측 결과를 구해 반환합니다."""
    model = get_model()
    y_pred = model.predict(x_test)  # model에 x_test를 입력했을 때 예측값을 구해 y_pred에 저장
    return y_pred

def get_accuracy(confusion_mat: list) -> float:
    """[[TN, FP],[FN, TP]] 형태로 구성된 혼동 행렬을 전달받아 정확도(accuracy)을 계산하여 반환합니다."""
    TN = confusion_mat[0, 0]
    TP = confusion_mat[1, 1]
    FP = confusion_mat[0, 1]
    FN = confusion_mat[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def get_recall(confusion_mat: np.ndarray) -> float:
    """[[TN, FP],[FN, TP]] 형태로 구성된 혼동 행렬을 전달받아 재현율(recall)을 계산하여 반환합니다."""
    TN = confusion_mat[0, 0]
    TP = confusion_mat[1, 1]
    FP = confusion_mat[0, 1]
    FN = confusion_mat[1, 0]

    recall = TP / (TP + FN)  # confusion_mat의 재현율을 계산
    return recall

def main():
    # 데이터 주소
    X_PATH = "data/x_test.csv"
    Y_PATH = "data/y_test.csv"

    # 데이터 로드
    x_test = load_csv(X_PATH)  # 입력
    y_test = load_csv(Y_PATH)  # 정답

    # 모델의 예측
    y_pred = get_predict(x_test)

    # 혼동행렬 구하기 ([[TN, FP],[FN, TP]] 형태)
    cm = confusion_matrix(y_test, y_pred)
    print("혼동행렬\n", cm)  # 혼동 행렬 출력

    # 정확도 계산
    accuracy = get_accuracy(cm)
    print("accuracy:", accuracy)

    # 재현율 계산
    recall = get_recall(cm)
    print("recall:", recall)

if __name__ == "__main__":
    main()
```





### 8. 의사결정 나무를 이용하여 약 종류 분류하기

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_data():
    """데이터를 불러오고 분할하는 함수"""
    # CSV 파일을 데이터프레임으로 불러오기
    df = pd.read_csv("drug.csv")
    
    # 종속 변수 y와 독립 변수 X로 나누기
    y = df["Drug"]
    X = df.drop(columns=["Drug"])
    
    # 데이터 분할 (7:3 비율로 학습 데이터와 테스트 데이터로 분리)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
    
    return train_X, test_X, train_y, test_y

def dt_clf(train_X, train_y):
    """의사결정 나무 모델을 학습하는 함수"""
    # 의사결정 나무 모델 생성 및 학습
    clf = DecisionTreeClassifier()
    clf.fit(train_X, train_y)
    
    return clf

# 데이터 불러오기 및 분할
train_X, test_X, train_y, test_y = load_data()

# 의사결정 나무 모델 학습
clf = dt_clf(train_X, train_y)
```





### 9. CNN 모델로 Fashion-MNIST 데이터 분류하기

```
import tensorflow as tf
import numpy as np
from plot import *
import warnings, logging, os

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(81)
tf.random.set_seed(81)

def CNN():
    # 지시사항 1번
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding="same", input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(24, kernel_size=(3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    return model
    
def main():
    
    x_train = np.loadtxt('./data/train_images.csv', delimiter=',', dtype=np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter=',', dtype=np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter=',', dtype=np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter=',', dtype=np.float32)
    
    # 지시사항 2번
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    model = CNN()
    
    # 지시사항 3번
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                  
    # 지시사항 4번
    model.fit(x_train, y_train, epochs=10, batch_size=64)
    
    # 지시사항 5번
    loss, test_acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    
    print('\nTEST 정확도:', test_acc, '\n')
    model.summary()
    plot(x_test, y_test, predictions)
    
if __name__ == "__main__":
    main()
```



### 10. 전력 데이터 시각화


```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from elice_utils import elice_utils


def load_csv(path: str) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 반환합니다."""
    df = pd.read_csv(path)
    return df


def make_bar(df: pd.DataFrame):
    """2019년의 월별 전력 소모량 평균을 막대 그래프로 그립니다."""
    # 2019년의 데이터만 추출
    df_2019 = df[df["Year"] == 2019]
    # sns.barplot을 사용하여 막대 그래프 생성
    sns.barplot(x="Month", y="Consumption", data=df_2019)
    
    # 그래프 보여주기
    show_plot("barplot")

def make_line(df: pd.DataFrame):
    """연도별 석탄 발전에 의한 전력 생산량의 선 그래프를 그립니다."""
    sns.lineplot(x="Year", y="Coal", data=df)
    
    # x축의 눈금 설정
    plt.xticks(np.arange(2019, 2024, 1))
    
    # 그래프 보여주기
    show_plot("lineplot")


def make_pie(df: pd.DataFrame):
    """2020년의 발전 방법의 생산 비율을 파이 그래프로 그립니다."""
    labels = [
        "Nuclear",
        "Wind",
        "Hydroelectric",
        "Oil and Gas",
        "Coal",
        "Solar",
        "Biomass",
    ]
    # 2020년의 데이터만 추출
    df_2020 = df[df["Year"] == 2020]
    
    # 각 항목의 합계 계산
    df_2020 = df_2020[labels].sum()
    
    # 파이 그래프 생성
    plt.pie(df_2020, labels=labels)
    
    # 그래프 보여주기
    show_plot("pie")



def show_plot(fig_name: str):
    """그래프를 보여주는 함수입니다."""
    plt.savefig(fig_name + ".png")
    elice_utils.send_image(fig_name + ".png")
    plt.cla()


def main():
    # 데이터 경로
    data_path = "data/electronic.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 1. 막대 그래프 그리기
    make_bar(df)

    # 2. 선 그래프 그리기
    make_line(df)

    # 3. 파이 그래프 그리기
    make_pie(df)


if __name__ == "__main__":
    main()
```


