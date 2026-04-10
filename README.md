# IMBK_Bank_Customer_Churn_ML
## 고객 이탈 분류 ML 및 인사이트 분석

Project Date: 2026-04-10

---

## Tech Stack: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, catboost, pycaret, optuna, shap 

---

## 2. Dataset: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

---

## 3. 데이터 전처리
### 1) 불필요 컬럼 제거
`customer_id`는 고객 식별용 변수로,  
은행 이탈에 직접적인 설명력을 갖는 변수라고 보기 어렵고 고유값이 지나치게 많아 제거했다.

```python
df = df.drop(columns=['customer_id'], errors='ignore')

### 2) 결측치 확인

모든 컬럼에 결측치가 존재하지 않아 별도의 결측치 처리는 수행하지 않았다.

### 3) 범주형 변수 가변수화
country, gender는 범주형 변수이므로 pd.get_dummies()를 사용해 더미 변수로 변환했다.
또한 drop_first=True를 적용하여 다중공선성 문제를 일부 완화했다.

```python
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

### 4) 스케일링
모델 학습 시 변수 간 값의 범위를 맞추기 위해 StandardScaler를 사용했다.
평균 0, 표준편차 1로 표준화하여 학습 안정성과 비교 가능성을 높였다.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

## 4. EDA

### 1) 수치형 변수와 Churn의 관계
<img width="1790" height="989" alt="EDA-수치형" src="https://github.com/user-attachments/assets/365ba884-f4e5-4392-9630-141fdf63afb2" />

해석
·credit_score
두 분포가 거의 겹쳐 보여, 신용점수는 이탈 여부와 강한 관계를 보이지 않을 가능성이 있다.
·age
비이탈 고객은 주로 30대에, 이탈 고객은 40~50대에 상대적으로 많이 분포했다.
따라서 나이가 많을수록 이탈 가능성이 높아질 가능성이 있다.
·tenure
두 분포가 전반적으로 유사하여 거래 기간 자체는 강한 설명변수라고 보기 어렵다.
·balance
잔고 0인 고객과 잔고가 있는 고객 사이의 패턴 차이가 보일 수 있어,
향후 잔고 보유 여부를 기준으로 추가 분석할 필요가 있다.
·estimated_salary
분포가 거의 겹쳐 보여 추정 연봉은 이탈과의 관계가 크지 않을 가능성이 있다.
핵심 요약

Age는 고객 이탈을 설명하는 핵심 변수로 작용할 가능성이 높다.

### 2) 범주형 변수와 Churn의 관계
<img width="1189" height="1190" alt="EDA-범주형" src="https://github.com/user-attachments/assets/a68ee04b-9caf-4817-94d9-42691d104d07" />

해석

기본 countplot에서는 모든 변수에서 이탈하지 않은 고객 수가 더 많았지만,
비율 기준으로 보면 유의미한 차이가 나타나는 변수들이 존재했다.

·성별(gender)
여성 고객의 이탈률이 남성보다 더 높게 나타났고,
신뢰구간도 겹치지 않아 두 집단 간 차이가 있을 가능성이 있다.
·신용카드 보유 여부(has_cr_card)
집단 간 이탈률 차이가 크지 않았고, 신뢰구간도 겹쳐
뚜렷한 차이가 있다고 보기 어렵다.
·활성 고객 여부(active_member)
비활성 고객의 이탈률이 활성 고객보다 더 높았으며,
신뢰구간도 겹치지 않아 중요한 변수일 가능성이 크다.

핵심 요약

gender, active_member는 고객 이탈과 밀접하게 연결된 핵심 변수일 가능성이 높다.


## 5. 모델링

### 1) PyCaret 기반 AutoML 비교
<img width="1169" height="201" alt="스크린샷 2026-04-10 113626" src="https://github.com/user-attachments/assets/67581fcb-067a-4f4e-9093-ca7cd851e134" />

PyCaret을 활용해 여러 분류 모델을 빠르게 비교한 뒤,
성능 상위 모델 4개를 선별하여 후속 최적화를 진행했다.

### 2) Optuna 기반 Hyperparameter Tuning
GBC F1: 0.5921450151057401
CatBoost F1: 0.6060606060606061
AdaBoost F1: 0.5809248554913294
XGBoost F1: 0.5946745562130178

결과 요약: 튜닝 이후 CatBoost가 가장 높은 F1 Score를 기록했다.


### 3) SHAP Value 기반 모델 해석
<img width="918" height="683" alt="스크린샷 2026-04-10 105326" src="https://github.com/user-attachments/assets/e166c1e7-f678-43b9-b492-9a13524c40d5" />

SHAP 분석 결과
·products_number
상품 보유 수가 많은 고객, 특히 3개 이상 보유한 고객은
이탈 방향으로 영향이 커지는 경향이 나타났다.
·age
나이가 많을수록 이탈 가능성을 높이는 방향으로 작용했다.
이는 앞선 EDA 결과와도 일관된다.
·active_member / gender_Male
비활성 고객일수록, 그리고 여성 고객일수록
이탈 가능성이 높아지는 패턴이 확인되었다.
·balance
계좌 잔액이 높은 고객일수록 이탈 가능성이 커지는 방향이 관찰되었다.
·country_Germany / country_Spain
독일, 스페인 고객군에서 상대적으로 높은 이탈 위험이 나타났다.

### 4) Stacking Ensemble
4개 모델을 조합해 Stacking Ensemble을 적용한 결과,
최종 성능은 다음과 같았다.

Final Stacking F1 Score: 0.6080

즉, 단일 최고 모델(CatBoost)보다
Stacking이 소폭 더 나은 성능을 보였다.


## 6. 인사이트 제안
·products_number(상품 보유 수): 상품을 오히려 더 많이 보유(3개 이상)한 고객의 이탈율이 높다. 이는 적당히 보유(0~2개)한 고객보다 상품을 많이 보유한 것이 대출금리 부담 등의 이유로 떠나갔을 가능성이 있다고 생각된다.  
·age(나이): 나이가 많을수록 이탈율이 높다. 이는 노인 고객들이 이 은행을 이용하기에 어려움을 느꼈을 가능성이 크다고 생각된다. 노인들은 인터넷뱅킹, 모바일뱅킹 등의 최신 기술에 뒤쳐져 있으므로 이들을 따로 관리해줄 수 있는 방법이 필요하다고 판단된다.  
·activate_member(활성 고객 여부), gender_Male(남성): 여성일수록 이탈율이 높다. 이는 여성 고객들을 대상으로한 상품들이 큰 매력을 가지고 있지 않다고 생각된다. 여성들도 본 은행을 이용할 만한 메리트가 있어야 된다고 느낄만한 상품을 출시하거나 기존의 상품들을 조금 수정해볼 필요성이 있다고 생각된다.  
또한, 비활성 고객일수록 이탈율이 높으므로 휴먼 상태인 고객들을 다시 끌어들일 만한 매력 있는 상품을 제안하거나 하는 등의 필요성이 있어야 한다고 생각된다.  
·balance(계좌 잔액): 계좌 잔액이 많을수록 이탈율이 높은 경향이 보인다. 돈이 많은데도 불구하고 이탈율이 높다는 것은 타행의 매력도가 더 크게 느껴져서 떠나갈 수 있다고 생각된다. 타행의 금리 혜택이 더 크거나 하는 등의 이유가 있을 수 있겠다.  
·country_Germany, Spain: 특정한 나라에서 이탈율이 더 높다는 것은 그 나라에 있는 본 은행이 문제가 있어서 일수도 있다고 생각된다. 즉각적인 조치를 취해야 할 필요성이 있다.













