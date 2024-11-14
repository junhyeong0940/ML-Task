# 24-2 기계학습 과제
#### 이 작업물은 Kaggle에서 타이타닉 데이터 셋을 다운 받아 데이터 분석으로 타이타닉 데이터셋에서 발견할 수 있는 잠재적 의미를 추출하고 인사이트를 발견함

## 주요 내용
* train,test csv 데이터 불러오기
```python
train = pd.read_csv('titanic/train.csv')
train

test= pd.read_csv('titanic/test.csv')
test
```
* 데이터 분석
```python
#train의 하위 5개 데이터 확인
train.tail()

#데이터의 0번째 row 확인
train.loc[0]

#데이터의 특정 column확인
train.Name

#중복되는 값을 제거하고 유일한 확인
train.Sex.unique()

# null 값 확인
train.isnull().sum()
```
* 가설 설정
##### 저는 나이가 0세에서 20세, 20세에서 40세이면서 Pclass 등급이 높은 여성의 생존율이 높을 것이라 가정했습니다.

* 데이터 삭제
```python
# null 값이 많은 Cabin 열 삭제
df = df.drop(columns=['Cabin'])
df = df.drop(columns = ['Ticket'])
df = df.drop(columns=['Name'])
df=df.drop(columns = ['PassengerId'])
df = df.drop(columns = ['SibSp'])
df = df.drop(columns = ['Fare'])
df = df.drop(columns = ['Embarked'])
df = df.drop(columns = ['Parch'])
```
* 결측치 채우기
```python 
#Null 값 확인
df.isnull().sum()
# 결과를 보면 Age에만 null값이 있음
# Age 값 확인
df.Age.mean()
df.Age.median()
# 첫번째 행과 마지막 행에 결측치가 없기 때문에 df.fillna(df.interpolate()) 사용
df['Age'] = df['Age'].fillna(df['Age'].interpolate())
df
# 연령대별로 구별하기 위해 2차원 리스트 생성
age_list =[[0, 20], [20, 40], [40, 60], [60, 80]]

for idx, age in enumerate(age_list):
    print(age, idx)
    df.loc[(age[0] < df['Age']) & (df['Age'] <= age[1]), 'Age'] = idx
#df.isnull().sum() 실행 시 결측치가 채워진 것 확인 가능
```
* 데이터 그룹화
```python
#성별과 생존 피쳐 그룹화
survived_sex = df.groupby(['Sex', 'Survived'])
survived_sex_count = survived_sex['Survived'].count()
#나이와 생존 피쳐 그룹화
survived_age = df.groupby(['Age', 'Survived'])
survived_age_count = survived_age['Survived'].count()
survived_age_count
# 나이 그룹에서 생존자 수 확인
survived_age_0 = survived_by_age[0.0, 1] if (0.0, 1) in survived_by_age.index else 0
survived_age_1 = survived_by_age[1.0, 1] if (1.0, 1) in survived_by_age.index else 0
survived_age_2 = survived_by_age[2.0, 1] if (2.0, 1) in survived_by_age.index else 0
survived_age_3 = survived_by_age[3.0, 1] if (3.0, 1) in survived_by_age.index else 0

print(survived_age_0, survived_age_1, survived_age_2, survived_age_3)

# Pclass별 생존자 수 그룹화ㅔ, Pclass별 생존 비율 분석

# Pclass별 생존자 수
survived_pclass = df.groupby(['Pclass','Survived'])
survived_plcass_count = survived_pclass['Survived'].count()
survived_plcass_count

# 전체 인원 수
total_count = len(df)

# Pclass별 생존자 수
survived_pclass = df[df['Survived'] == 1].groupby('Pclass')['Survived'].count()

# Pclass별 생존 비율 계산: 전체 인원 수 / Pclass별 생존자 수
survival_pclass_rate = survived_pclass / total_count

survival_rate_1 = survival_pclass_rate[1]
survival_rate_2 = survival_pclass_rate[2]
survival_rate_3 = survival_pclass_rate[3]
# 결과 출력
print(survival_pclass_rate)
```
* 데이터 시각화 결과
데이터 시각화로 인해 알 수 있는 부분은
1. 여성이 남성보다 더 많이 살아 남았다.
2. 나이가 0~20, 20~40대 인 사람이 생존율이 높았다.
3. Pclass가 높은 사람이 생존율이 높았다.






## 개발환경
* python 3.10
* 필요한 라이브러리 -Numpy -Pandas -Matplotlib

## 터미널 명령어
```python
# 가상환경 생성
conda create -n ML python=3.10
# 필요한 라이브러리 설치
conda install nummpy
conda install matplotlib
conda install pandas
