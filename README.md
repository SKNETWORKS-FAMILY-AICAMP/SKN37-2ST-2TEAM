# SKN37-2ST-2TEAM
# 고객 이탈 예측 분석 및 Streamlit 대시보드

> 한 줄 소개: 이커머스 멤버십 고객 데이터를 분석해 이탈 위험 고객을 예측하고, 실무자가 쓸 수 있는 대시보드로 구현한 프로젝트



## 👥 팀원 소개

| 전윤호 | 이승훈 | 신지호 | 문진호 | 김지원 | 김보영 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/46d9cec5-3aa2-4f30-9154-644edad9ccfe" alt="전윤호" width="110" height="110"> | <img src="https://github.com/user-attachments/assets/c7a442ab-13b1-4a8f-a0d0-32143ed21f7f" alt="이승훈" width="110" height="110"> | <img src="https://github.com/user-attachments/assets/5165d87d-0818-4f3c-abba-d284bf303345" alt="신지호" width="110" height="110"> | <img src="https://github.com/user-attachments/assets/6688888a-d473-47c7-8b45-1325829ac953" alt="문진호" width="110" height="110"> | <img width="110" height="110" alt="김지원" src="https://github.com/user-attachments/assets/588a22aa-792f-46a6-a6b2-cf607155ba8d">|<img width="110"  height="110" alt="김보영" src="https://github.com/user-attachments/assets/0bc195a2-4462-4b50-9b5c-f9e36ed5dd85"> |


## 🎯 핵심 결과
- churn_risk_score를 이진 타깃으로 사용하여, 1을 이탈, 0을 유지 라벨로 정의
- 이탈 핵심 요인:  멤버십, 보유 포인트, 평균 이용금액, 피드백
- 최종 모델: PyTorch MLP, 정확도 93.8%, 정밀도 94.5%, 재현율 94.0%, F1-score 94.2%, ROC-AUC 97.6%
- 구현: 전체 이탈 라벨 현황, 멤버십·포인트 구간별 분석, 고객별 예측 결과와 모델 평가 지표 대시보드




## 1. 프로젝트 배경과 목표

### 프로젝트 배경
#### 이커머스 시장의 성장
eMarketer의 2025년 전망에 따르면 전 세계 소매판매에서 이커머스가 차지하는 비중은 20%를 넘어설 것으로 예상된다. 국내 역시 2025년 상반기 온라인쇼핑 거래액이 약 131.5조 원으로, 전체 소매판매의 41.1%를 차지하며 온라인 소비가 주요 구매 방식으로 자리 잡고 있다.

<img width="600" alt="국내 소매판매에서 온라인쇼핑이 차지하는 비중" src="https://github.com/user-attachments/assets/34ea9bfe-70de-47df-a1d0-4b3e0949c42b" />

<br>

#### 고객 유지의 중요성
이커머스 시장의 성장과 경쟁 심화로 고객 유지의 중요성이 커지고 있다. 제공된 자료에 따르면 국내 온라인 쇼핑몰 폐업 건수는 2020년 4만 1,119건에서 2024년 9만 4,850건으로 증가했다. 폐업의 원인을 고객 이탈만으로 설명할 수는 없지만, 기존 고객을 유지하는 일은 안정적인 매출 확보와 마케팅 비용 부담 완화에 도움이 될 것으로 보인다. Columbia University의 「Valuing Customers」(2004)에서도 고객 유지율이 1% 개선될 때 기업 가치가 약 5% 높아질 수 있다고 분석했다. 이에 본 프로젝트는 고객 이탈 위험을 미리 파악하고, 고객 유지 방안을 마련하는 데 참고할 수 있는 예측 모델을 구현하고자 한다.

<br>
<img width="500" height="401" alt="image" src="https://github.com/user-attachments/assets/0fe99443-ded9-461f-87f4-4bbd80c569fa" />


### 프로젝트 목표
1. 데이터의 고객 특성과 이탈 라벨 간 관계를 EDA로 파악
2. 멤버십, 보유 포인트, 평균 거래 금액, 피드백 그룹을 사용해 PyTorch MLP 이진분류 모델을 학습
3. Accuracy, Precision, Recall, F1-score, ROC-AUC를 이용해 모델 성능 측정
4. 학습된 모델과 전처리기를 Streamlit에 연결해 고객 정보 입력과 예측 결과 확인이 가능한 대시보드 구현

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

- `security_no`, `referral_id`는 식별자이므로 분석에서 제외한다.
- `?`, `NaN`, `Error`는 결측값으로 처리한다.
- 범주형 변수의 결측값은 `Unknown`으로 통일한다.
- `churn_risk_score`의 비정상 값(-1 등)은 제거한다.
- `age_group`, `tenure_days`, `visit_hour`, `visit_period`, `positive_feedback` 등의 파생 변수를 생성할 수 한다.
- 수치형 도메인은 실제 데이터에서 확인한 최솟값과 최댓값이다.
- 단위 및 정의가 명확하지 않은 변수는 컬럼명을 바탕으로 추정했으며, 정확한 정의는 확인이 필요하다.

## 3. 분석 과정
### 3-1. 이탈 고객 특성 (EDA)
#### 이탈 고객 분포
<img src="https://github.com/user-attachments/assets/dfd45902-f86e-45e3-b982-bb4bcf97b914" width="400">

#### 수치형 컬럼 분포
<img width="2400" height="960" alt="02_numeric_distribution" src="https://github.com/user-attachments/assets/4a9d4324-7c8d-413f-9589-aa112e10ec2a" />


평균 거래 금액(avg_transaction_value):
- 이탈한 고객(1)의 비율이 이탈하지 않은 고객보다 0-6000 사이의 값에서 더 높다.  
- 이탈한 고객(1)의 분포는 50000 초과에서 보이지 않는다.

<br>

평균 로그인 빈도 관련 지표(avg_frequency_login_days):
- 20-30 부분에서 이탈한 고객(1)의 비율이 그렇지 않은 고객의 비율보다 약 1.5배 정도 높다.

<br>

보유 포인트(points_in_wallet):
- 이탈한 고객(1)의 보유 포인트(points_in_wallet)이 이탈하지 않은 고객(0)보다 중앙값과 최빈값이 작다.

<br>

회원의 나이(age), 마지막 로그인 후 경과일(days_since_last_login), 평균 이용 시간(avg_time_spent), 
가입 후 경과일(tenure_days), 방문 시간대(visit_hour) 컬럼들은 고객 이탈 여부가 분포에 유의미하게 영향을 미치지 않는다.

#### 수치형 변수들끼리의 상관 분석(heatmap)
<img width="965" height="884" alt="image" src="https://github.com/user-attachments/assets/fed597cd-e005-4432-b951-54c964318d17" />

수치형 변수들 간의 상관성은 미미한 수준이므로 다중공선성을 걱정할 필요가 없다.

# 범주형 고객 분포
<img width="2190" height="1790" alt="image" src="https://github.com/user-attachments/assets/ff386552-cf15-4770-b380-998c9dfcd447" />

- 다른 컬럼들에선 고객 이탈율 속한 값에 따라 큰 차이를 안보였지만 멤버십 등급(membership_category), 고객 피드백 유형(feedback)에선 속한 값에 따라 이탈율이 확연히 다르다.



## 4. 모델링
- 사용 모델: PyTorch MLP

<br>

- 사용한 변수:<br>고객 멤버십(membership_category), 포인트(points_in_wallet),평균 거래금액(avg_transaction_value)

<br>

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

<h3><핵심지표 페이지></h3>
<img width="900"  alt="image" src="https://github.com/user-attachments/assets/fe6802d2-632b-433a-88ba-e2094ddd4f0a" />
<br>
<h3><고객 이탈 예측 페이지></h3> 
<img width="900"  alt="image" src="https://github.com/user-attachments/assets/9f0ebff2-aa95-4412-9154-34a3f25a1dfa" />
 <br>
 <h3><이탈 원인 분석 페이지></h3>
<img width="900"  alt="image" src="https://github.com/user-attachments/assets/4c25e43e-828d-493a-85ed-f854aac7fca7" />
 <br>
<h3><모델 성능 페이지></h3>
<img width="900"  alt="image" src="https://github.com/user-attachments/assets/ef25bacf-d6cf-40c1-9b46-df4197d5b96d" />
<br>
<br>
  🔗 **데모**: (Streamlit 링크https://github.com/user-attachments/assets/4043fc17-cd72-482e-bf20-d71b68677dad)  


## 7. 기대 효와 개선 방향
### 기대 효과
ShopInsight는 고객의 이용 특성과 회원 정보를 분석하여 이탈 위험을 예측하고, 기업의 고객 관리와 의사결정을 지원하는 것을 목표로 한다.

첫째, 이탈 위험 고객의 사전 식별
이탈 가능성이 높은 고객을 조기에 파악하여 고객 유지 활동의 우선순위를 설정할 수 있다.

둘째, 맞춤형 고객 관리 전략 수립
회원 등급, 보유 포인트, 이용 금액 등의 특성을 바탕으로 고객별 맞춤 혜택과 리텐션 마케팅 전략을 수립할 수 있다.

셋째, 비즈니스 의사결정 지원
고객 특성과 이탈 위험을 종합적으로 분석하여 마케팅 효율을 높이고, 기존 고객과의 장기적인 관계 형성에 기여할 수 있다.

궁극적으로 ShopInsight를 단순한 이탈 예측 도구를 넘어 고객의 가치를 발견하고 기업의 성장을 지원하는 고객 분석 솔루션으로 발전시키고자 한다.

### 향후 개선 방향
이번 프로젝트를 통해 모델의 예측 성능뿐만 아니라 실제 환경에서의 신뢰성과 활용 가능성을 높이기 위한 개선 과제를 확인했다.

첫째, 특정 변수에 대한 모델 의존성 개선
보유 포인트(points_in_wallet)와 회원 등급(membership_category)에 집중된 변수 중요도를 추가로 검증하고, 변수 구성을 달리한 모델 비교를 통해 예측의 안정성을 높이고자 한다. 또한 타깃 라벨 생성 기준과 정보 누출 가능성도 점검할 필요가 있다.

둘째, 고객 행동 분석 범위 확대
최근 구매일, 구매 빈도 변화, 재방문 여부, 장바구니 이용 패턴 등 시간에 따른 행동 변수를 추가하여 고객의 이탈 신호를 더욱 정밀하게 파악하고자 한다.

셋째, 모델 검증 및 서비스 고도화
새로운 기간의 데이터를 활용한 검증과 다양한 성능 지표 비교를 통해 모델의 일반화 성능을 확인하고, 예측 결과가 실제 고객 유지율과 재구매율 개선으로 이어지는지 검증하고자 한다.

궁극적으로 고객의 이탈을 예측하는 수준을 넘어, 고객별 맞춤형 관리 방안까지 제안할 수 있는 실질적인 비즈니스 솔루션으로 발전시키는 것이 목표다.


## 8. 소감
<table>
  <thead>
    <tr>
      <th width="80"align="center">이름</th>
      <th>소감</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="80"align="center">전윤호</td>
      <td>
        실습에서 배운 내용을 프로젝트 전반에 적용하며 복습할 수 있어 뜻깊었습니다. 의견이 필요할 때마다 팀원들이 적극적으로 의견을 나눠주셨고, 여러 의견을 하나로 모으는 과정도 원활하게 이루어져 프로젝트를 큰 어려움 없이 차근차근 진행할 수 있었습니다.<br>
        부족한 팀장이었지만 적극적으로 함께해주신 모든 팀원분께 진심으로 감사드립니다.
      </td>
    </tr>
    <tr>
      <td width="80"align="center">이승훈</td>
      <td>
        이번 프로젝트는 단순히 하나의 서비스를 만드는 과정이 아니라, 그동안 배웠던 Python과 머신러닝, 딥러닝, 데이터 분석 등의 개념을 실제 문제에 적용해 보는 첫 경험이었습니다.<br><br>
        프로젝트를 시작할 당시에는 고객 이탈 예측이라는 주제 자체가 익숙하지 않았습니다. 다양한 머신러닝 모델 중 어떤 모델이 보유 데이터에 적합할지 판단하는 것도 쉽지 않았습니다. 의사결정나무(Decision Tree)를 시작으로 MLP 등 여러 모델을 직접 학습시키고 결과를 비교하면서 적합한 모델을 찾아갔습니다. 익숙하지 않은 개념과 기술이 많았던 만큼 AI의 도움을 받아 새로운 내용을 하나씩 이해하고 프로젝트에 적용했습니다.<br><br>
        특히 인상 깊었던 부분은 특성 중요도 분석이었습니다. 분석 결과, 보유 포인트(points_in_wallet)와 회원 등급(membership_category) 두 변수가 모델의 예측에 높은 비중을 차지했습니다. 이 두 변수만으로도 상당히 높은 예측 성능을 얻을 수 있다는 점이 인상적이었습니다.<br><br>
        처음에는 높은 정확도가 긍정적으로 느껴졌지만, 모델이 고객의 다양한 행동 특성을 종합적으로 학습한 것이 아니라 특정 두 변수의 패턴에 지나치게 의존하고 있을 가능성도 있다고 생각했습니다.<br><br>
        이를 확인하기 위해 입력 변수 구성을 달리해 모델을 학습하고 성능을 비교했습니다. 기존 22개 변수를 사용하는 방식뿐만 아니라 9개, 5개 변수를 사용하는 방식도 실험하며 변수 구성에 따라 정확도와 재현율 등 평가 지표가 어떻게 달라지는지 살펴보았습니다.<br><br>
        이 과정을 통해 정확도가 높은 모델을 선택하는 것뿐 아니라, 모델이 어떤 변수를 기반으로 예측하는지와 특정 변수에 지나치게 의존하지 않는지도 확인해야 한다는 점을 배웠습니다. 높은 성능을 얻는 것만큼 결과를 신뢰할 수 있는지 검증하는 과정도 중요하다는 것을 느꼈습니다.<br><br>
        이번 프로젝트를 성공적으로 마무리할 수 있었던 것은 함께한 팀원들의 역할이 컸다고 생각합니다. 전윤호 님, 신지호 님, 김지원 님, 문진호 님, 김보영 님과 각자의 역할을 수행하며 한마음으로 프로젝트를 진행했기에 결과물을 만들어 낼 수 있었습니다.<br><br>
        모두가 처음 접하는 프로젝트였고 서로 처음 만난 사이였지만, 초반부터 원활하게 의사소통이 이루어졌다는 점이 특히 의미 있었습니다. 서로 의견을 나누고 진행 방향을 함께 고민하면서 큰 혼란 없이 프로젝트를 마무리할 수 있었습니다. 결과물도 의미 있었지만, 처음 만난 사람들과 하나의 목표를 향해 협력하고 함께 완성해 나간 경험이 더욱 값지게 느껴졌습니다.<br><br>
        이번 프로젝트는 아직 부족한 부분이 많다는 것을 깨닫는 동시에, 새로운 기술을 배우고 직접 적용해 볼 수 있었던 소중한 경험이었습니다.
      </td>
    </tr>
    <tr>
      <td width="80"align="center">신지호</td>
      <td>
        직접 데이터를 찾아보고 어떤 데이터가 프로젝트에 더 적합한지, 어떤 모델로 학습했을 때 성능이 더 좋았는지 비교해보는 등 배운 내용을 실제로 적용해 보는 좋은 경험이었습니다. 팀원들이 적극적으로 참여해주신 덕분에 함께 의견을 나누고 공유하면서 크게 막히는 부분 없이 프로젝트를 마무리할 수 있었습니다.
      </td>
    </tr>
    <tr>
      <td width="80"align="center">문진호</td>
      <td>
        이번 팀 프로젝트는 팀원들(전윤호, 김지원, 신지호, 이승훈, 문진호, 김보영)과 함께 처음 접한 AI의 기초 개념을 다지고, 머신러닝과 딥러닝의 핵심 원리까지 폭넓게 이해할 수 있었던 뜻깊은 시간이었습니다.<br><br>
        먼저 필요한 데이터를 직접 수집한 뒤, 데이터 전처리로 품질을 높이고 다각도의 데이터 분석(EDA)을 수행해 모델 학습에 유의미한 특성과 인사이트를 도출했습니다.<br><br>
        이후 팀원들과 긴밀히 소통하고 AI 도구를 활용하여 정제된 데이터를 바탕으로 다층 퍼셉트론(MLP) 모델을 직접 학습했습니다.<br><br>
        마지막으로 분석 결과와 모델을 Streamlit으로 시각화해 웹페이지로 구현함으로써, 데이터 처리부터 모델링과 구현까지 AI 개발 과정을 함께 경험할 수 있었습니다.
      </td>
    </tr>
    <tr>
      <td width="80"align="center">김지원</td>
      <td>
        이번 프로젝트를 통해 뚜렷한 결과물을 만들어 낼 수 있어 뿌듯했습니다. 팀원들과 함께 진행하면서 업무 처리 속도가 빨라졌고, 각 과정도 더욱 원활하게 진행될 수 있었습니다. 또한 실무적인 데이터 분석을 경험하며 데이터 분석이 실제 업무에서 어떻게 활용되고 기업에 어떤 방식으로 기여할 수 있는지 체감할 수 있어 뜻깊은 시간이었습니다.
      </td>
    </tr>
  </tbody>
</table>


