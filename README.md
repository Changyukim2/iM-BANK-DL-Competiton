# Credit Score Classification using Deep Learning

> Kaggle Credit Score Classification 데이터를 활용하여 고객의 신용 점수 등급(`Good`, `Standard`, `Poor`)을 예측하는 딥러닝 기반 다중분류 프로젝트입니다.  
> 단순 정확도 향상보다 **전처리 논리, EDA 기반 해석, Feature Selection, 모델 개선 과정, 검증 성능 설명**에 초점을 두었습니다.

---

## 1. 프로젝트 개요

금융 고객의 소득, 대출, 연체, 부채, 신용 이용 패턴 등의 정보를 바탕으로 고객의 신용 점수 등급을 예측하는 프로젝트입니다.  
타깃 변수는 `Credit_Score`이며, 예측 클래스는 `Good`, `Standard`, `Poor`입니다.

본 프로젝트에서는 정형 데이터를 딥러닝 모델에 입력하기 위해 식별자 제거, 결측치 처리, 범주형 인코딩, 수치형 스케일링을 수행하고, RandomForest 기반 Feature Importance를 활용해 주요 피처를 선별했습니다. 이후 PyTorch 기반 MLP 모델을 학습하여 검증 성능을 평가했습니다.

---

## 2. 사용 데이터

- 데이터 출처: Kaggle Credit Score Classification
- 데이터 파일: `train2.csv`
- 목표 변수: `Credit_Score`
- 문제 유형: 다중분류 Classification

### 주요 변수

| 구분 | 변수 예시 | 설명 |
|---|---|---|
| 고객 정보 | `Age`, `Occupation` | 나이, 직업 |
| 소득 정보 | `Annual_Income`, `Monthly_Inhand_Salary` | 연간 소득, 월 실수령 급여 |
| 금융 이용 | `Num_Bank_Accounts`, `Num_Credit_Card`, `Num_of_Loan` | 은행 계좌 수, 신용카드 수, 대출 수 |
| 신용 위험 | `Interest_Rate`, `Delay_from_due_date`, `Outstanding_Debt` | 이자율, 연체 일수, 미지급 부채 |
| 신용 이력 | `Credit_History_Age`, `Credit_Mix` | 신용 이력 기간, 신용 혼합 유형 |
| 결제 행동 | `Payment_of_Min_Amount`, `Payment_Behaviour` | 최소금액 납부 여부, 결제 행동 |

---

## 3. 분석 및 모델링 프로세스

```text
데이터 로드
   ↓
기본 정보 확인 및 EDA
   ↓
식별자 제거 및 결측치 처리
   ↓
범주형 One-Hot Encoding / 수치형 Standard Scaling
   ↓
Type_of_Loan 파생변수 생성
   ↓
RandomForest 기반 Feature Selection
   ↓
PyTorch MLP 모델 학습
   ↓
Validation Accuracy / Macro F1 / Classification Report 평가
```

---

## 4. 전처리 과정

### 4.1 식별자 제거

다음 변수는 예측에 직접적인 의미가 낮거나 개인정보성 식별자이므로 제거했습니다.

```python
ID, Name, SSN, Customer_ID
```

특히 `Customer_ID`는 같은 고객이 여러 월에 반복 등장할 수 있기 때문에 모델이 고객 자체를 외우는 데이터 누수 위험이 있다고 판단했습니다. 따라서 모델 입력 피처에서는 제외했습니다.

### 4.2 결측치 처리

- 수치형 변수: 중앙값으로 대체
- 범주형 변수: 최빈값으로 대체

수치형 변수는 이상치 영향을 줄이기 위해 평균보다 중앙값을 사용했습니다.

### 4.3 범주형 인코딩

범주형 변수는 LabelEncoder 대신 One-Hot Encoding을 사용했습니다.  
LabelEncoder는 범주에 임의의 순서를 부여할 수 있어, 순서성이 없는 범주형 변수에 부적절할 수 있기 때문입니다.

```python
Occupation
Credit_Mix
Payment_of_Min_Amount
Payment_Behaviour
```

### 4.4 수치형 스케일링

MLP는 입력 변수의 스케일에 민감하기 때문에 수치형 변수에 `StandardScaler`를 적용했습니다.

### 4.5 Type_of_Loan 파생변수 생성

`Type_of_Loan`은 여러 대출명이 문자열로 결합된 형태였기 때문에 전체 문자열을 그대로 One-Hot Encoding하지 않았습니다.  
대신 대출 종류별 포함 여부를 0/1 변수로 변환했습니다.

예시:

```text
Loan_auto_loan
Loan_personal_loan
Loan_mortgage_loan
Loan_student_loan
```

---

## 5. EDA

EDA에서는 타깃 분포, 주요 수치형 변수의 통계량, `Credit_Score` 등급별 평균 차이, 변수 간 상관관계를 확인했습니다.

### 5.1 타깃 분포 확인

`Credit_Score`는 `Standard` 클래스가 가장 많고, `Good` 클래스가 상대적으로 적은 구조였습니다.  
따라서 Accuracy만으로 모델을 평가하면 소수 클래스 성능이 가려질 수 있으므로 Macro F1-score도 함께 확인했습니다.

<img width="597" height="460" alt="eda1" src="https://github.com/user-attachments/assets/716c0130-5d69-49d2-9cf1-fca6a8cb0b12" />



### 5.2 Credit_Score별 주요 변수 평균 비교

EDA 결과 다음 변수들이 신용 점수 등급 차이를 설명하는 데 중요해 보였습니다.

- `Outstanding_Debt`
- `Interest_Rate`
- `Delay_from_due_date`
- `Credit_History_Age`
- `Credit_Utilization_Ratio`

일반적으로 미지급 부채, 이자율, 연체 일수가 높을수록 `Poor` 등급으로 분류될 가능성이 커졌고, 신용 이력이 길수록 `Good` 등급과 관련성이 높은 경향을 보였습니다.


<img width="1384" height="716" alt="eda2" src="https://github.com/user-attachments/assets/2fb0d69b-8bae-47a9-9bb5-61417227d397" />


### 5.3 상관관계 분석

수치형 변수 간 상관관계를 확인하여 중복 정보가 강한 변수와 신용 점수 분류에 영향을 줄 수 있는 변수를 파악했습니다.



<img width="705" height="614" alt="eda3" src="https://github.com/user-attachments/assets/1ee7913e-63b5-4d1a-b386-66c205002a0a" />



---

## 6. Feature Selection

전처리 이후 모든 피처를 그대로 사용하지 않고, RandomForest 기반 Feature Importance를 활용하여 주요 피처를 확인했습니다.

상위 중요도 변수에는 EDA에서 중요하다고 판단한 다음 변수들이 포함되었습니다.

- `Interest_Rate`
- `Outstanding_Debt`
- `Delay_from_due_date`
- `Credit_History_Age`
- `Credit_Mix`

이를 통해 EDA 해석과 모델 기반 중요도가 어느 정도 일치함을 확인했습니다.  
최종적으로 중요도 기준 상위 50개 피처를 선택하여 MLP 모델의 입력으로 사용했습니다.


<img width="884" height="684" alt="feature" src="https://github.com/user-attachments/assets/7a8b304a-2ca9-4615-9ea6-a1d4c17945bc" />



---

## 7. 모델링 방법

### 7.1 모델 선택 이유

본 프로젝트는 수치형 변수와 범주형 변수가 혼합된 정형 데이터 기반 다중분류 문제입니다.  
전처리된 피처 벡터의 비선형 관계를 학습하기 위해 PyTorch 기반 MLP 모델을 사용했습니다.

### 7.2 모델 구조

```text
Input Layer
   ↓
Linear(256) + BatchNorm1d + ReLU + Dropout(0.20)
   ↓
Linear(128) + BatchNorm1d + ReLU + Dropout(0.15)
   ↓
Linear(64) + ReLU
   ↓
Output Layer(3 classes)
```

### 7.3 학습 설정

| 항목 | 설정 |
|---|---|
| Framework | PyTorch |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Batch Size | 512 |
| Epochs | 60 |
| Train / Valid Split | 8:2 Stratified Split |
| Evaluation Metrics | Accuracy, Macro F1-score, Classification Report |

### 7.4 Class Weight 미사용 이유

클래스 불균형을 확인하기 위해 class weight를 계산했지만, 최종 loss에는 적용하지 않았습니다.  
Class weight를 적용하면 소수 클래스 보정에는 도움이 될 수 있으나 전체 validation accuracy가 흔들릴 수 있다고 판단했습니다.  
대신 Accuracy와 Macro F1-score를 함께 출력하여 전체 성능과 클래스별 균형을 동시에 확인했습니다.

---

## 8. 성능 결과

최종 모델은 validation set에서 다음 성능을 기록했습니다.

| Metric | Score |
|---|---:|
| Validation Loss | 0.4658 |
| Validation Accuracy | 80.66% |
| Macro F1 Score | 0.8013 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Good | 0.7269 | 0.8315 | 0.7757 | 3,566 |
| Poor | 0.7822 | 0.8498 | 0.8146 | 5,799 |
| Standard | 0.8564 | 0.7747 | 0.8135 | 10,635 |
| Macro Avg | 0.7885 | 0.8187 | 0.8013 | 20,000 |
| Weighted Avg | 0.8118 | 0.8066 | 0.8071 | 20,000 |

---

## 9. 학습 과정 시각화

학습 과정에서 train loss와 validation loss가 함께 감소했고, train accuracy와 validation accuracy도 함께 상승했습니다.  
따라서 60 epoch 기준으로 심한 과적합은 나타나지 않았다고 판단했습니다.


<img width="1183" height="384" alt="valid" src="https://github.com/user-attachments/assets/1148b47b-c32b-4f44-b672-715575972c64" />



---

## 10. Confusion Matrix

모델은 세 클래스 모두에서 비교적 균형 잡힌 성능을 보였습니다.  
특히 `Poor`와 `Standard` 클래스는 F1-score가 0.81 수준으로 나타났고, `Good` 클래스도 recall이 0.83으로 양호했습니다.

<img width="534" height="476" alt="c" src="https://github.com/user-attachments/assets/30a1bc76-4550-48df-92a9-2cb6bce3e036" />


---

## 11. 프로젝트 한계 및 개선 방향

### 한계점

- 같은 `Customer_ID`가 여러 월에 반복 등장하는 데이터 구조상, 단순 random split은 동일 고객의 다른 월 데이터가 train과 valid에 동시에 포함될 수 있습니다.
- MLP 모델은 정형 데이터에서 강력한 트리 기반 모델보다 성능이 낮을 수 있습니다.
- Epoch, hidden size, dropout, learning rate는 수동 튜닝 중심으로 설정했습니다.

### 개선 방향

- `Customer_ID` 기준 GroupShuffleSplit 또는 GroupKFold 적용
- Optuna를 활용한 하이퍼파라미터 자동 튜닝
- LightGBM, XGBoost, TabNet 등 정형 데이터 특화 모델과 성능 비교
- 대출 조합, 연체 추세, 월별 변화량 기반 파생변수 추가
- MLP와 트리 기반 모델의 앙상블 적용

---

## 12. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Preprocessing | scikit-learn |
| Modeling | PyTorch |
| Evaluation | accuracy, macro F1, classification report, confusion matrix |

---

## 13. 핵심 요약

본 프로젝트는 신용 점수 분류 문제를 딥러닝 기반 MLP 모델로 해결한 프로젝트입니다.  
식별자 제거, 결측치 처리, One-Hot Encoding, Standard Scaling, RandomForest 기반 Feature Selection을 적용했으며, PyTorch MLP 모델을 통해 최종 validation accuracy 80.66%, macro F1-score 0.8013을 달성했습니다.  
또한 Accuracy만이 아니라 Macro F1-score와 Classification Report를 함께 확인하여 클래스별 성능 균형까지 점검했습니다.
