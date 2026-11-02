# SKN37-2ST-2TEAM
# 고객 이탈 예측 분석 및 Streamlit 대시보드

> 한 줄 소개: (예) 멤버십 고객 데이터를 분석해 이탈 위험 고객을 예측하고, 실무자가 쓸 수 있는 대시보드로 구현한 프로젝트

 📓 **노트북**: (Colab 링크)


## 👥 팀원 소개

| 전윤호 | 이승훈 | 신지호 | 문진호 | 김지원 | 김보영 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/46d9cec5-3aa2-4f30-9154-644edad9ccfe" alt="전윤호" width="110" height="140"> | <img src="https://github.com/user-attachments/assets/c7a442ab-13b1-4a8f-a0d0-32143ed21f7f" alt="이승훈" width="110" height="140"> | <img src="https://github.com/user-attachments/assets/5165d87d-0818-4f3c-abba-d284bf303345" alt="신지호" width="110" height="140"> | <img src="https://github.com/user-attachments/assets/6688888a-d473-47c7-8b45-1325829ac953" alt="문진호" width="110" height="140"> | <img src="https://github.com/user-attachments/assets/e55e5330-7696-42a9-80ed-60c4fa2a38a2" alt="김지원" width="110" height="140"> |<img width="110" height="140" alt="image" src="https://github.com/user-attachments/assets/0bc195a2-4462-4b50-9b5c-f9e36ed5dd85" /> |


## 🎯 핵심 결과
- 이탈 핵심 요인: (예) 멤버십 등급, 포인트, 피드백
- 최종 모델: (모델명), Test ROC-AUC (값), F1 (값), train-test 격차 (값)
- 구현: 위험 고객 관리 / 캠페인 시뮬레이터 / 고객 개별 조회 대시보드

## 1. 프로젝트 배경과 목표
## 2. 데이터
- 출처: https://huggingface.co/datasets/d0r1h/customer_churn
- 크기: 36,992(rows) X 23(columns)
- 타깃: 이커머스(e-commerce) 사이트 기업

## 데이터 컬럼 설명 및 도메인

| No. | 컬럼명 | 설명 | 자료형 | 도메인 (값·범위) | 전처리 및 활용 |
|:---:|---|---|:---:|---|---|
| 1 | `age` | 고객 나이 | 수치형 | 10 ~ 64 | 연령대(`age_group`) 파생 |
| 2 | `gender` | 성별 | 범주형 | M, F, Unknown | 범주형 변수로 사용 |
| 3 | `security_no` | 고객 식별용 보안 번호 | ID | 고객별 고유값 | 분석에서 제외 |
| 4 | `region_category` | 거주 지역 유형 | 범주형 | Town, City, Village, NaN | 결측값을 Unknown으로 처리 |
| 5 | `membership_category` | 멤버십 등급 | 범주형 | No, Basic, Silver, Gold, Platinum, Premium Membership | 이탈과 강하게 연관된 변수 |
| 6 | `joining_date` | 가입일 | 날짜형 | 범위 미확인 | 가입 후 경과일(`tenure_days`) 파생 |
| 7 | `joined_through_referral` | 추천 가입 여부 | 범주형 | Yes, No, ? | `?`를 결측값으로 처리 |
| 8 | `referral_id` | 추천인 ID | ID | 추천인 식별값 | 분석에서 제외 |
| 9 | `preferred_offer_types` | 선호 혜택 유형 | 범주형 | Gift Vouchers/Coupons, Credit/Debit Card Offers, Without Offers, NaN | 결측값을 Unknown으로 처리 |
| 10 | `medium_of_operation` | 주 이용 기기 | 범주형 | Desktop, Smartphone, Both, ? | `?`를 결측값으로 처리 |
| 11 | `internet_option` | 인터넷 연결 방식 | 범주형 | Wi-Fi, Mobile_Data, Fiber_Optic | 범주형 변수로 사용 |
| 12 | `last_visit_time` | 마지막 방문 시각 | 시간형 | 범위 미확인 | 방문 시간대(`visit_hour`, `visit_period`) 파생 |
| 13 | `days_since_last_login` | 마지막 로그인 이후 경과일 | 수치형 | 1 ~ 26 | 음수, -999 등 이상값 처리 |
| 14 | `avg_time_spent` | 평균 이용 시간 | 수치형 | 1.84 ~ 3,235.58 | 단위 추정: 분 / 음수 결측 처리 |
| 15 | `avg_transaction_value` | 평균 거래 금액 | 수치형 | 800.46 ~ 99,914.05 | 이탈과의 관계 분석 |
| 16 | `avg_frequency_login_days` | 평균 로그인 빈도 관련 지표 | 수치형 | 0.01 ~ 73.06 | Error 결측 처리 / 정확한 정의 미확인 |
| 17 | `points_in_wallet` | 보유 포인트 | 수치형 | 범위 미확인 | 이탈과 상관관계가 큰 변수 |
| 18 | `used_special_discount` | 특별 할인 사용 여부 | 범주형 | Yes, No | 범주형 변수로 사용 |
| 19 | `offer_application_preference` | 혜택 적용 선호 여부 | 범주형 | Yes, No | 범주형 변수로 사용 |
| 20 | `past_complaint` | 과거 불만 접수 여부 | 범주형 | Yes, No | 범주형 변수로 사용 |
| 21 | `complaint_status` | 불만 처리 상태 | 범주형 | Not Applicable, Unsolved, Solved, Solved in Follow-up, No Information Available | 범주형 변수로 사용 |
| 22 | `feedback` | 고객 피드백 유형 | 범주형 | Poor Product Quality, No reason specified, Too many ads 등 | 긍정 피드백 여부(`positive_feedback`) 파생 |
| 23 | `churn_risk_score` | 이탈 위험 점수 | 이진형 | 0, 1 | 타깃 변수 / 1 = 이탈, 0 = 유지 |

### 전처리 참고사항

- `security_no`, `referral_id`는 식별자이므로 분석에서 제외합니다.
- `?`, `NaN`, `Error`는 결측값으로 처리합니다.
- 범주형 변수의 결측값은 `Unknown`으로 통일합니다.
- `churn_risk_score`의 비정상 값(-1 등)은 제거합니다.
- `age_group`, `tenure_days`, `visit_hour`, `visit_period`, `positive_feedback` 등의 파생 변수를 생성할 수 있습니다.
- 수치형 도메인은 실제 데이터에서 확인한 최솟값과 최댓값입니다.
- 단위 및 정의가 명확하지 않은 변수는 컬럼명을 바탕으로 추정했으며, 정확한 정의는 확인이 필요합니다.

## 3. 분석 과정
### 3-1. 이탈 고객 특성 (EDA)
#### 이탈 고객 분포
<img src="https://github.com/user-attachments/assets/dfd45902-f86e-45e3-b982-bb4bcf97b914" width="400">

#### 수치형 컬럼 분포
<img width="2400" height="960" alt="02_numeric_distribution" src="https://github.com/user-attachments/assets/4a9d4324-7c8d-413f-9589-aa112e10ec2a" />

회원의 나이(age), 마지막 로그인 후 경과일(days_since_last_login), 평균 이용 시간(avg_time_spent), 
가입 후 경과일(tenure_days), 방문 시간대(visit_hour): 
- 고객 이탈 여부가 분포에 영향을 미치지 않음


평균 거래 금액(avg_transaction_value):
- 이탈한 고객(1)의 비율이 이탈하지 않은 고객보다 0-6000 사이의 값에서 더 높음  
- 이탈한 고객(1)의 분포는 6000 초과에서 보이지 않음.


평균 로그인 빈도 관련 지표(avg_frequency_login_days):
- 20-30 부분에서 이탈한 고객(1)의 비율이 그렇지 않은 고객의 비율보다 약 2배 정도 높음


보유 포인트(points_in_wallet):
- 이탈한 고객(1)의 보유 포인트(points_in_wallet)이 이탈하지 않은 고객(0)보다 중앙값과 최빈값이 작음.

#### 수치형 변수들끼리의 상관 분석(heatmap)
<img width="965" height="884" alt="image" src="https://github.com/user-attachments/assets/fed597cd-e005-4432-b951-54c964318d17" />

수치형 변수들 간의 상관성은 미미한 수준이므로 다중공산성을 걱정할 필요가 없다.

# 범주형 고객 분포
<img width="2190" height="1790" alt="image" src="https://github.com/user-attachments/assets/ff386552-cf15-4770-b380-998c9dfcd447" />

- 다른 컬럼들에선 고객 이탈율 속한 값에 따라 큰 차이를 안보였지만 멤버십 등급(membership_category), 고객 피드백 유형(feedback)에선 속한 값에 따라 이탈율이 확연히 다름.



## 4. 모델링
- 사용 모델: Keras MLP
- 사용한 변수:<br>고객 멤버십(membership_category), 포인트(points_in_wallet),평균 거래금액(avg_transaction_value)
- 사용한 파생 변수: <br> 피드백 그룹(feedback_group): 범주형 변수 feedback 값들을 <br>긍정(Products always in Stock, Quality Customer Care, Reasonable Price, User Friendly Website), <br>부정(Poor Product Quality, Poor Customer Service, Poor Website, Too many ads), <br>없음(No reason specified) 범주로 묶음


## 5. 결과와 해석
- 성능 비교표 / ROC / 변수 중요도<br>
  <img width="592" height="382" alt="image" src="https://github.com/user-attachments/assets/50da9d04-ae64-4435-94b1-f20ecbc4b137" />

평가 데이터에서 정확도 0.938, 정밀도 0.945, 재현율 0.940, F1 0.942, ROC-AUC 0.976을 기록했다. 학습 정확도(0.934)와 평가 정확도의 차이가 -0.003으로 과적합은 관찰되지 않았으며, 유지와 이탈 두 클래스 모두 F1 0.93~0.94로 한쪽에 치우치지 않은 성능을 보였다.

<img width="622" height="393" alt="image" src="https://github.com/user-attachments/assets/e9ca10cb-c31f-4998-92e1-3d2515121706" />

학습 손실(BCE)은 첫 에폭 약 0.22에서 2에폭 만에 약 0.17로 급감했고, 이후 완만히 감소하여 30에폭 이후에는 약 0.153 부근에서 수렴했다. 무작위 예측의 손실(약 0.693)보다 크게 낮은 값으로, 모델이 이탈 패턴을 안정적으로 학습했음을 보여준다.

<img width="493" height="385" alt="image" src="https://github.com/user-attachments/assets/4397de4a-d12f-4efd-92bb-cca3f69c7914" />

평가 데이터 7,399명에 대한 혼동행렬에서 유지 3,175명, 이탈 3,762명을 올바르게 분류했고, 유지 고객 221명을 이탈로 오분류(거짓 양성)하고 이탈 고객 241명을 놓쳤다(거짓 음성). 두 오류의 규모가 비슷해 한쪽 클래스에 치우치지 않은 균형 잡힌 분류 성능을 보였다.


- 비즈니스 인사이트와 제안

## 6. Streamlit 웹사이트
- 페이지별 설명과 캡처<br>
  🔗 **데모**: (Streamlit 링크https://github.com/user-attachments/assets/4043fc17-cd72-482e-bf20-d71b68677dad)  
- 실행 방법

## 7. 한계와 개선 방향

## 8. 프로젝트 구조와 실행 방법
```
churn_app/
├─ Home.py
├─ pages/
├─ models/
├─ notebooks/
├─ images/
└─ requirements.txt
```
```bash
pip install -r requirements.txt
streamlit run Home.py
```

## 9. 소감
-전윤호

-이승훈

이번 프로젝트는 나에게 단순히 하나의 서비스를 만드는 과정이 아니라, 그동안 배웠던 Python과 머신러닝, 딥러닝, 데이터 분석 등의 개념을 실제 문제에 적용해 보는 첫 경험이었다.

프로젝트를 시작할 당시에는 고객 이탈 예측이라는 주제 자체가 익숙하지 않았다. 특히 다양한 머신러닝 모델 중 어떤 모델이 우리가 보유한 데이터에 가장 적합할지 판단하는 것부터 쉽지 않았다. 의사결정나무(Decision Tree)를 시작으로 MLP 등 여러 모델을 직접 학습시키고 결과를 비교하면서 적합한 모델을 찾아가는 과정이 필요했다. 익숙하지 않은 개념과 기술이 많았던 만큼 AI의 도움을 받아 새로운 내용을 하나씩 이해하고 실제 프로젝트에 적용해 나갔다.

프로젝트를 진행하면서 특히 인상 깊었던 부분은 특성 중요도 분석이었다. 분석 결과, 보유 포인트(points_in_wallet)와 회원 등급(membership_category) 두 가지 변수가 모델의 예측에 매우 높은 비중을 차지하고 있었다. 심지어 이 두 변수만으로도 상당히 높은 예측 성능을 얻을 수 있다는 점이 인상적이었다.

처음에는 높은 정확도가 긍정적으로 느껴졌지만, 다시 생각해 보니 오히려 주의 깊게 살펴봐야 할 부분이었다. 모델이 고객의 다양한 행동 특성을 종합적으로 학습한 것이 아니라, 특정 두 변수에 포함된 패턴에 지나치게 의존해 이탈 위험을 판단하고 있을 가능성이 있었기 때문이다.

이러한 문제를 확인하기 위해 입력 변수 구성을 달리하여 모델을 학습하고 성능을 비교했다. 기존 22개 변수를 사용하는 방식뿐만 아니라 9개, 5개 변수만 사용하는 방식도 실험하면서 변수 구성에 따라 정확도와 재현율 등 평가 지표가 어떻게 달라지는지 살펴보았다.

이 과정을 통해 단순히 정확도가 높은 모델을 선택하는 것보다 모델이 어떤 변수를 기반으로 예측하고 있는지, 특정 변수에 지나치게 의존하고 있지는 않은지 확인하는 과정이 중요하다는 점을 배웠다. 또한 높은 성능을 얻는 것만큼이나 그 결과를 신뢰할 수 있는지 검증하는 과정도 중요하다는 것을 느꼈다.

마지막으로 이번 프로젝트를 성공적으로 마무리할 수 있었던 것은 함께한 팀원들의 역할이 컸다고 생각한다. 전윤호 님, 신지호 님, 김지원 님, 문진호 님, 김보영 님과 함께 각자의 역할을 수행하며 한마음으로 프로젝트를 진행했기에 지금과 같은 결과물을 만들어 낼 수 있었다.

모두가 처음 접하는 프로젝트였고 서로 처음 만난 사이였지만, 초반부터 원활하게 의사소통이 이루어졌다는 점이 특히 의미 있었다. 서로의 의견을 나누고 진행 방향을 함께 고민하면서 큰 혼란 없이 프로젝트를 마무리할 수 있었다. 개인적으로는 결과물 자체도 의미 있었지만, 처음 만난 사람들과 하나의 목표를 향해 협력하고 함께 완성해 나갔다는 경험이 더욱 값지게 느껴졌다.

이번 프로젝트는 아직 부족한 부분이 많다는 것을 깨닫는 동시에, 새로운 기술을 배우고 직접 적용해 볼 수 있었던 소중한 경험이었다.

-신지호

-문진호

-김지원
