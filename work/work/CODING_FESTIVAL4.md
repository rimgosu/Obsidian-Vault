
문제 1. 덧칠하기
```
# 입력 받기
N, M = map(int, input().split())
points = [tuple(map(int, input().split())) for _ in range(N)]
rectangles = [tuple(map(int, input().split())) for _ in range(M)]

# 포인트가 속한 직사각형 수를 저장할 배열
point_count = [0] * N

# 각 포인트에 대해 모든 직사각형을 확인
for i in range(N):
    x, y = points[i]
    for j in range(M):
        xiL, yiL, xiR, yiR = rectangles[j]
        if xiL < x < xiR and yiL < y < yiR:
            point_count[i] += 1

# 최대 값을 찾아 출력
max_count = max(point_count)
print(max_count)
```


2. 면적 구하기
```
def calculate_area(points):
    n = len(points)
    points.sort()  # x 좌표를 기준으로 정렬

    area = 0.0
    for i in range(n - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # 사다리꼴 면적을 계산하여 누적
        area += (x2 - x1) * (y1 + y2) / 2.0

    return abs(area)

if __name__ == "__main__":
    n = int(input())
    points = []

    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    area = calculate_area(points)

    # 면적이 정수인지 실수인지 확인 후 출력 형식 조절
    if area.is_integer():
        print(int(area))
    else:
        print(area)
```

3. 다이어트 실험 결과 정렬
```
import pandas as pd

# exercise.csv 파일을 불러옵니다.
df = pd.read_csv("exercise.csv")

# 첫 번째 줄에 출력할 행 개수를 입력받습니다.
num_rows = int(input())

# 두 번째 줄에 오름차순인지 내림차순인지 입력받습니다.
order = int(input())
ascending_order = True if order == 1 else False

# 세 번째 줄에 정렬할 기준의 컬럼 명을 입력받습니다.
column_name = input()

# 입력받은 기준으로 데이터를 정렬합니다.
sorted_df = df.sort_values(by=column_name, ascending=ascending_order)

# 첫 번째 줄에 입력받은 행 개수만큼 위에서부터 읽어 출력합니다.
print(sorted_df.head(num_rows))

```




4. 식당 방문 손님 비교 조건

```
import pandas as pd

# tips.csv 파일을 pandas를 사용하여 불러옵니다.
tips_df = pd.read_csv('tips.csv')

# 조건으로 비교할 컬럼 명을 입력받습니다.
column_name = input()

# 비교 조건으로 사용할 숫자를 입력받습니다.
compare_value = float(input())

# 입력받은 조건에 따라 DataFrame을 필터링합니다.
filtered_df = tips_df[tips_df[column_name] < compare_value]

# 상위 10줄만 출력합니다.
print(filtered_df.head(10))

```




5. 다이아몬드 가격 데이터 탐색
```
import pandas as pd

def load_csv(path) -> pd.DataFrame:
    """pandas를 이용하여 path의 데이터를 DataFrame의 형태로 불러오는 함수입니다."""
    df = pd.read_csv(path)
    return df

def get_biggest(df: pd.DataFrame) -> pd.DataFrame:
    """1. df에서 캐럿 무게가 가장 무거운 데이터를 찾아 데이터프레임의 형태로 반환합니다."""
    df_biggest = df[df['carat'] == df['carat'].max()]
    return df_biggest

def get_price_diff(df: pd.DataFrame) -> float:
    """2. df에서 절삭품질(cut)이 Fair인 제품들과 Premium인 제품들의 평균가격의 차이를 구하여 반환합니다."""
    price_fair = df[df['cut'] == 'Fair']['price'].mean()
    price_premium = df[df['cut'] == 'Premium']['price'].mean()
    price_diff = price_premium - price_fair
    return price_diff

def search_diamond(df: pd.DataFrame) -> pd.DataFrame:
    """3. 지시사항의 조건들을 만족하는 다이아몬드 중 가장 무거운 다이아몬드를 검색하여 그 데이터를 반환합니다."""
    condition = (df['price'] < 6000) & (df['cut'] == 'Premium')
    df_filtered = df[condition]
    df_biggest = df_filtered[df_filtered['carat'] == df_filtered['carat'].max()]
    return df_biggest

def main():
    # 데이터 경로
    data_path = "data/diamond.csv"

    # 데이터 불러오기
    df = load_csv(data_path)

    # 가장 무거운 다이아몬드 찾기
    df_biggest = get_biggest(df)
    print("가장 큰 다이아\n", df_biggest)

    # 지시사항을 참고하여 특정 절삭품질간의 평균가격차이 구하기
    price_diff = get_price_diff(df)
    print("평균가격 차이:", price_diff)

    # 지시사항의 조건을 만족하는 다이아몬드 찾기
    result = search_diamond(df)
    print("검색결과\n", result)

if __name__ == "__main__":
    main()

```




6. 당뇨병 데이터 EDA
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


def make_plot(df: pd.DataFrame):
    """matplotlib 라이브러리를 이용해 산점도를 그리는 함수입니다."""
    x = df['bmi']  # 'bmi' 컬럼을 x 값으로 사용
    y = df['target']  # 'target' 컬럼을 y 값으로 사용

    plt.scatter(x, y, alpha=0.5)  # 산점도 그래프 그리기
    plt.title('BMI vs. Diabetes Progression')  # 그래프 제목 설정
    plt.xlabel('BMI (Body Mass Index)')  # x축 레이블 설정
    plt.ylabel('Diabetes Progression')  # y축 레이블 설정

    return x, y


def main():
    DATA_PATH = "data/diabetes.csv"
    df = load_csv(DATA_PATH)
    x, y = make_plot(df)
    
    plt.savefig("plot1.png")
    elice_utils.send_image("plot1.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

```




7. F1 score로 AdaBoost 분류 모델 평가하기
```
import sys

import matplotlib.pyplot as plt
import numpy as np
from elice_utils import EliceUtils
from sklearn.datasets import make_circles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 채점을 위해 시드를 고정합니다. 수정하지 마세요.
np.random.seed(42)

def load_data():
    # 데이터 생성 및 분할
    X, y = make_circles(n_samples=500, factor=0.5, noise=0.1, random_state=42)
    
    # 데이터 분할
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return train_X, test_X, train_y, test_y

def predict(train_X, test_X, train_y):
    # 모델 생성, 학습과 예측
    model = AdaBoostClassifier(n_estimators=5, random_state=42)
    model.fit(train_X, train_y)
    
    predicted = model.predict(test_X)
    
    return predicted

def f1(test_y, predicted):
    # F1 score 계산
    score = f1_score(test_y, predicted)
    
    return score

def main():
    train_X, test_X, train_y, test_y = load_data()

    predicted = predict(train_X, test_X, train_y)

    score = f1(test_y, predicted)

    print(f"F1 Score: {score:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

```


8. 클러스터링과 차원 축소를 활용한 와인 데이터 분석하기

```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 와인 데이터를 불러오고 데이터 프레임으로 만들어주는 함수
def load_data():
    wine = load_wine()
    wineDF = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    wineDF['target'] = wine.target
    return wineDF

'''
1. K-Means 클러스터링을 수행하는 
   k_means_clus 함수를 구현합니다.
   
   Step01. K-Means 객체를 불러옵니다.
           
           클러스터의 개수는 3, 
           중심점 초기화는 랜덤,
           random_state는 200으로 설정합니다.
           
   Step02. K-Means 클러스터링을 수행합니다.
           
           클러스터링은 정답이 없는 데이터를
           사용합니다. 따라서 drop 메소드를 이용해
           target 변수를 제거한 데이터를 학습시켜줍니다.
           
   Step03. 클러스터링 결과 즉, 각 데이터가 
           속한 클러스터 중심점들의 label을 
           wine 데이터 프레임인 'wineDF'에 추가합니다.
           labels_ 메소드를 이용하세요.
'''

def k_means_clus(wineDF):
    kmeans = KMeans(n_clusters=3, init='random', random_state=200)
    kmeans.fit(wineDF.drop('target', axis=1))
    wineDF['cluster'] = kmeans.labels_
    
    # 클러스터링 결과를 보기 위한 groupby 함수
    wine_result = wineDF.groupby(['target', 'cluster'])['malic_acid'].count()
    print(wine_result)
    
    return wine_result, wineDF

'''
2. 주성분 분석(PCA)을 통해 클러스터링 결과를
   2차원으로 시각화해주는 함수 Visualize를 구현합니다.
   
   Step01. PCA 객체를 불러옵니다.
           
           원 데이터를 2차원으로 차원 축소할 수 있도록 
           n_components는 2로 설정합니다.
           
   Step02. 주성분 분석을 수행합니다.
   
   Step03. 주성분 분석으로 차원을 축소시킨 결과를 반환합니다.
'''

def Visualize(wineDF):
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(wineDF.drop(['target', 'cluster'], axis=1))
    wineDF['pca_x'] = pca_transformed[:, 0]
    wineDF['pca_y'] = pca_transformed[:, 1]
    
    # 클러스터링된 값이 0, 1, 2 인 경우, 인덱스 추출
    idx_0 = wineDF[wineDF['cluster'] == 0].index
    idx_1 = wineDF[wineDF['cluster'] == 1].index
    idx_2 = wineDF[wineDF['cluster'] == 2].index
    
    # 각 클러스터 인덱스의 pca_x, pca_y 값 추출 및 시각화
    fig, ax = plt.subplots()
    
    ax.scatter(x=wineDF.loc[idx_0, 'pca_x'], y=wineDF.loc[idx_0, 'pca_y'], marker='o')
    ax.scatter(x=wineDF.loc[idx_1, 'pca_x'], y=wineDF.loc[idx_1, 'pca_y'], marker='s')
    ax.scatter(x=wineDF.loc[idx_2, 'pca_x'], y=wineDF.loc[idx_2, 'pca_y'], marker='^')
    
    ax.set_title('K-means')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    
    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")
    
    return pca_transformed

def main():
    wineDF = load_data()
    wine_result, wineDF = k_means_clus(wineDF)
    Visualize(wineDF)

if __name__ == "__main__":
    main()
```



9. 의사결정나무를 이용하여 붓꽃 분류하기 

```

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_csv(path):
    df = pd.read_csv(path)
    return df

def divide_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def model_train(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    # 데이터 경로
    data_path = "iris.csv"

    # load_csv 함수를 사용하여 데이터를 불러와 저장합니다.
    df = load_csv(data_path)

    # 데이터 분리
    X_train, X_test, y_train, y_test = divide_data(df)

    # 모델 학습
    model = model_train(X_train, y_train)
    y_pred = model.predict(X_test)

    # 평가지표 계산
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {score * 100}%")

if __name__ == "__main__":
    main()

```





10. 화살표 게임

```
from collections import deque #queue 개념으로 풀기위해 import 해서 풀기 double-ended queue의 약자로 popleft, pop 둘다 진행할 수 있다.

def bfs(graph, start):
    visited = set() #set을 통해 집합으로 처리
    queue = deque([start]) #deque 객체 만들기
    
    while queue: #queue 가 존재하다면
        node = queue.popleft() # node는 큐에서 가장 왼쪽 값을 반환한 결과(list라면 index 0번)
        if node not in visited: # 노드가 방문한 집합에 존재하지 않는다면 
            visited.add(node) # visited에 위에서 반환한 node를 추가
            queue.extend(graph[node]) #graph[node]를 queue에 넘기기
    return visited #마지막으로 visited를 반환

N, K = map(int, input().split())
graph = {} # 집합객체 초기화
for i in range(1, N+1):
    left, right = map(int, input().split())
    graph[i] = [left, right]

visited_nodes = bfs(graph, K) #nodes란 데이터 요소의 저장소로서, 다른 노드와 연결될 수 있다. 즉 visited_node가 다른 노드와 연결된 경로를 뜻한다.
print(N - len(visited_nodes))
```