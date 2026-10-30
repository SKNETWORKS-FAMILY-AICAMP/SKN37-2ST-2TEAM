"""ShopInsight · 고객 이탈 예측 대시보드

기준 디자인: 사용자가 선택한 shopinsight_churn_ui HTML
Streamlit 기준 코드: 사용자가 제공한 app(9).py
데이터: data/[2nd] customer_churn_ecommerce_cleaned.csv
MLP 모델: models/mlp_churn_4features.pth (PyTorch)
전처리기: models/mlp_churn_preprocessor.pkl (scikit-learn 1.6.1)

실행: python -m streamlit run app.py
"""
from __future__ import annotations

import html as html_lib
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ShopInsight · 고객 이탈 예측",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / "data" / "[2nd] customer_churn_ecommerce_cleaned.csv",
    BASE / "[2nd] customer_churn_ecommerce_cleaned.csv",
    BASE / "data" / "customer_churn_ecommerce_cleaned.csv",
]
MODEL_CANDIDATES = [
    BASE / "models" / "mlp_churn_4features.pth",
    BASE / "mlp_churn_4features.pth",
]
PREPROCESSOR_CANDIDATES = [
    BASE / "models" / "mlp_churn_preprocessor.pkl",
    BASE / "mlp_churn_preprocessor.pkl",
]
POSITIVE_FEEDBACK = {
    "Products always in Stock",
    "Quality Customer Care",
    "Reasonable Price",
    "User Friendly Website",
}
MEMBERSHIP_OPTIONS = [
    "No Membership", "Basic Membership", "Silver Membership",
    "Gold Membership", "Platinum Membership", "Premium Membership",
]
FEEDBACK_OPTIONS = [
    "Poor Product Quality", "Poor Customer Service", "Poor Website",
    "Too many ads", "No reason specified", "Reasonable Price",
    "User Friendly Website", "Products always in Stock", "Quality Customer Care",
]
PAGES = ["핵심 지표", "고객 이탈 예측", "이탈 원인 분석", "모델 성능"]
MANWON = 10_000  # 만원 -> 원
# 화면 선택값 -> 모델의 feedback_group 값
FEEDBACK_GROUP_OPTIONS = {"긍정": "positive", "부정": "negative", "없음": "none"}
REQUIRED_DATA_COLUMNS = {
    "membership_category", "points_in_wallet", "feedback",
    "avg_transaction_value", "avg_frequency_login_days",
    "avg_time_spent", "churn_risk_score",
}

(st.html if hasattr(st, "html") else lambda c: st.markdown(c, unsafe_allow_html=True))("""
<style>
:root{--bg:#f7f5f1;--side:#fbfaf7;--paper:#fff;--ink:#1b2234;--muted:#7d8697;--line:#ebe8e2;
--blue:#2f62f0;--blue-2:#5a7cff;--blue-soft:#edf2ff;--purple:#7b6cf6;--green:#2f9e6e;
--green-soft:#e6f5ec;--rose:#e2617f;--shadow:0 10px 30px rgba(40,52,90,.06);}
html,body,[class*="css"]{font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;}
/* 본문 영역: HTML 시안 본문과 같은 여백·너비 (Streamlit 기본은 최대 1400px + 좌우 큰 여백이라 짧고 좁아 보임) */
.block-container{padding:30px clamp(20px,3.4vw,48px) 48px !important;max-width:none !important;}
/* 글꼴: Streamlit 기본 글꼴(Source Sans) 대신 시안과 같은 글꼴, 줄 간격도 시안처럼 기본값으로 */
.hero,.card,.kpi,.crumb,.footer,.info-band,.side-card,.side-meta,.side-status,.brand,.form-h,.row-grid{
 font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;line-height:normal;}
.hero *,.card *,.kpi *,.side-card *,.brand *{font-family:inherit;}
.row-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:16px;align-items:stretch;margin-bottom:2px;}
.row-grid > .card,.row-grid > .kpis{margin:0;}
.row-grid > .kpis > .kpi{height:100%;}
section[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--line);
 width:256px !important;min-width:256px !important;max-width:256px !important;}
/* HTML 시안과 같은 사이드바: 너비 256px, 로고를 맨 위(여백 30px)에 */
[data-testid="stSidebarHeader"]{display:none !important;}
[data-testid="stSidebarContent"]{padding:0 !important;scrollbar-gutter:auto !important;}
[data-testid="stSidebarUserContent"]{padding:30px 22px 22px !important;margin:0 !important;width:auto !important;}
[data-testid="stSidebarResizeHandle"]{display:none !important;}

/* sidebar */
.brand{display:flex;gap:12px;align-items:center;padding:0 4px 18px;border-bottom:1px solid var(--line);margin-bottom:20px;}
.brand-mark{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;
 background:linear-gradient(145deg,#6aa5ff,#3a63f0);box-shadow:0 6px 14px rgba(47,98,240,.28);}
.brand-name{font-size:17px;font-weight:800;letter-spacing:-.02em;color:var(--ink);}
.brand-sub{font-size:10px;color:var(--muted);}
/* 사이드바 메뉴 버튼 (key="nav_0"~"nav_3"): 테마와 무관하게 색 고정 */
section[data-testid="stSidebar"] [class*="st-key-nav_"]{margin:0 0 -11px;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button{
 width:auto !important;min-height:0 !important;padding:11px 14px !important;border:0 !important;border-radius:12px !important;
 background:transparent !important;box-shadow:none !important;justify-content:flex-start !important;gap:11px;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button::before{content:"";width:12px;height:12px;flex:none;
 border-radius:50%;border:1.5px solid #c3c8d2;background:#fff;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button *{color:#3a4254 !important;font-size:14px !important;font-weight:500 !important;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover{background:#f0eee9 !important;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"]{
 background:linear-gradient(90deg,var(--blue),var(--blue-2)) !important;box-shadow:0 6px 16px rgba(47,98,240,.3) !important;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"]::before{
 border-color:#fff;background:radial-gradient(circle,#fff 35%,transparent 40%);}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] *{color:#fff !important;font-weight:700 !important;}
.stApp p,.stApp label{color:var(--ink);}
.side-card{margin-top:28px;padding:20px 18px;border-radius:16px;background:linear-gradient(180deg,#f4f6ff,#fff);box-shadow:var(--shadow);}
.side-card strong{display:block;font-size:15px;line-height:1.45;margin:12px 0 6px;color:var(--ink);}
.side-card p{margin:0;font-size:11px;line-height:1.65;color:var(--muted);}
.side-meta{font-size:10px;color:#9aa1ae;margin:16px 4px 10px;line-height:1.6;}
.side-status{padding:12px 14px;border-radius:10px;font-size:12px;font-weight:600;}
.st-off{background:#fff4e3;color:#9a6a24;} .st-on{background:var(--green-soft);color:var(--green);}

/* main */
.crumb{color:#6f7889;font-size:11px;font-weight:700;letter-spacing:.14em;margin:12px 0 -2px;}
.hero{position:relative;overflow:hidden;border-radius:22px;min-height:250px;box-shadow:var(--shadow);
 background:radial-gradient(circle at 78% 30%,rgba(122,150,255,.35),transparent 42%),
 radial-gradient(circle at 96% 90%,rgba(255,196,140,.45),transparent 38%),
 linear-gradient(100deg,#fff 0%,#fff 42%,#eef2ff 70%,#fdf1e6 100%);
 display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);margin-bottom:22px;}
.hero-text{padding:34px;display:flex;flex-direction:column;justify-content:center;}
.hero-eyebrow{color:var(--blue);font-size:10px;font-weight:800;letter-spacing:.16em;}
.hero h1{margin:0 0 16px;font-size:clamp(28px,3.2vw,40px);line-height:1.2;letter-spacing:-.05em;font-weight:900;
 background:linear-gradient(90deg,#2257ea,#3d6cf5);-webkit-background-clip:text;background-clip:text;color:transparent;padding:0;}
.hero p{margin:0;color:#5a6274;font-size:13px;line-height:1.8;}
.hero-art{position:relative;min-height:250px;}
.hero-illust{position:absolute;right:8px;bottom:4px;width:100%;max-width:590px;height:auto;}
.glass{position:absolute;background:rgba(255,255,255,.72);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.9);
 border-radius:18px;box-shadow:0 14px 30px rgba(60,80,160,.14);padding:16px 18px;}
.glass small{display:block;font-size:11px;color:#5b6478;}
.glass b{display:block;font-size:24px;letter-spacing:-.03em;margin-top:3px;}
.g1{left:6%;top:22%;transform:rotate(-6deg);} .g1 b{color:var(--blue);}
.g2{right:6%;top:34%;transform:rotate(5deg);} .g2 b{color:#1d2a55;}
.g3{left:22%;bottom:10%;width:210px;transform:rotate(-2deg);}
.mini-bars{display:flex;align-items:flex-end;gap:7px;height:56px;margin-top:8px;}
.mini-bars span{flex:1;border-radius:5px 5px 2px 2px;background:linear-gradient(180deg,#7d98ff,#3a63f0);}
.hero-ctrl{position:absolute;top:18px;right:22px;display:flex;gap:10px;z-index:2;align-items:center;}
.chip{background:rgba(255,255,255,.86);border:1px solid #e8e9f0;border-radius:12px;padding:8px 12px;font-size:12px;color:#3a4254;}
.avatar{width:36px;height:36px;border-radius:50%;display:grid;place-items:center;background:#1f3478;color:#fff;font-weight:700;}

.card,.kpi{background:var(--paper);border:1px solid #efece6;border-radius:18px;box-shadow:var(--shadow);}
.card{padding:22px;margin-bottom:16px;}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:18px;}
.kpis.two{grid-template-columns:repeat(2,minmax(0,1fr));}
.kpis.one{grid-template-columns:1fr;margin-bottom:0;}
.kpi{padding:20px;display:flex;gap:14px;align-items:center;min-height:108px;}
.kpi-ico{width:52px;height:52px;flex:none;border-radius:15px;display:grid;place-items:center;font-size:22px;}
.i-blue{background:#eaefff;color:var(--blue);} .i-rose{background:#fdecf1;color:var(--rose);}
.i-purple{background:#efedff;color:var(--purple);} .i-green{background:var(--green-soft);color:var(--green);}
.kpi-label{color:#4b5366;font-size:11px;font-weight:600;}
.kpi-value{font-size:25px;font-weight:850;letter-spacing:-.03em;margin:4px 0 3px;color:var(--ink);}
.kpi-foot{color:var(--green);font-size:10px;font-weight:600;}
.kpi-foot.gray{color:#98a0ad;font-weight:500;} .metric-empty{color:#a7aebb;}

.sec-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:18px;}
.sec-title{display:flex;gap:10px;align-items:center;font-size:17px;font-weight:800;letter-spacing:-.03em;color:var(--ink);}
.sec-ico{width:28px;height:28px;border-radius:8px;background:var(--blue-soft);color:var(--blue);display:grid;place-items:center;font-size:13px;}
.sec-help{color:var(--muted);font-size:11px;margin-top:6px;line-height:1.55;}
.pill{color:#4d63c9;background:var(--blue-soft);border-radius:99px;padding:6px 10px;font-size:10px;font-weight:600;white-space:nowrap;}
.pill.warn{color:#9b6b2c;background:#fff2dc;}
.foot-note{color:var(--muted);font-size:11px;margin-top:14px;}

.bars{display:grid;gap:15px;}
.bar-row{display:grid;grid-template-columns:22px minmax(110px,160px) 1fr 50px;align-items:center;gap:10px;font-size:12px;}
.rank{width:22px;height:22px;border-radius:50%;background:var(--blue-soft);color:var(--blue);font-size:10px;font-weight:700;display:grid;place-items:center;}
.bar-label{color:#343c4e;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.bar-label em{font-style:normal;color:#9aa1ae;font-weight:500;font-size:10px;margin-left:4px;}
.track{background:#eef0f5;height:10px;border-radius:99px;overflow:hidden;}
.fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--blue),var(--purple));}
.fill.green{background:linear-gradient(90deg,#5cc89a,var(--green));}
.bar-value{text-align:right;color:#566074;font-size:11px;font-weight:700;}

.column-chart{height:210px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;align-items:end;border-bottom:1px solid #eceae4;}
.column-group{height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:7px;}
.column-pct{color:#4b5366;font-size:11px;font-weight:700;}
.column-bar{width:min(52px,72%);border-radius:9px 9px 3px 3px;background:linear-gradient(180deg,#7d98ff,var(--blue));}
.column-labels{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
.column-labels span{color:#6f7889;font-size:10px;text-align:center;padding:8px 0 4px;}

.heatmap{display:grid;grid-template-columns:96px repeat(4,1fr);gap:6px;font-size:11px;}
.heatmap div{padding:12px 6px;text-align:center;border-radius:8px;font-weight:600;}
.heat-h{color:#6f7889;background:#f6f4ef;font-weight:700;} .heat-r{background:#f6f4ef;color:#3d4557;}
.h1{background:#e9eeff;color:#3554c4;} .h2{background:#c9d5ff;color:#24409f;}
.h3{background:#8ea5fb;color:#fff;} .h4{background:#4a6cf0;color:#fff;}

.matrix{border-collapse:separate;border-spacing:6px;width:100%;font-size:12px;}
.matrix th,.matrix td{padding:16px 10px;text-align:center;border-radius:10px;}
.matrix th{background:#f6f4ef;color:#4b5366;}
.matrix td{background:#fafaf8;color:#a7aebb;border:1px dashed #dedbd3;}
.matrix td.filled{background:#edf2ff;color:#24409f;border:0;font-weight:800;font-size:15px;}

.result-placeholder{min-height:250px;display:grid;place-items:center;text-align:center;padding:24px;color:var(--muted);}
.result-placeholder .icon{width:62px;height:62px;border-radius:18px;margin:0 auto 14px;display:grid;place-items:center;
 background:linear-gradient(145deg,#6aa5ff,#3a63f0);box-shadow:0 10px 22px rgba(47,98,240,.25);}
.result-placeholder strong{color:#3a4254;display:block;margin-bottom:7px;font-size:14px;}
.result-placeholder p{font-size:11px;line-height:1.65;margin:0 auto;max-width:290px;}
.result-big{font-size:46px;font-weight:900;letter-spacing:-.04em;margin:6px 0;}
.badge{display:inline-block;padding:6px 14px;border-radius:99px;font-size:12px;font-weight:700;}
.badge.risk{background:#fdecf1;color:var(--rose);} .badge.safe{background:var(--green-soft);color:var(--green);}
.info-band{margin-top:14px;padding:14px 16px;border-radius:12px;background:#eaf1ff;color:#2b4fb8;font-size:12px;line-height:1.65;}
.footer{color:#9aa1ae;font-size:10px;margin-top:28px;}

/* 입력 위젯: 다크 테마여도 밝은 입력창 유지 */
[data-testid="stWidgetLabel"] p{color:#3d4557 !important;font-size:12px !important;font-weight:600;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#9aa1ae !important;font-size:11px;}
[data-testid="stSelectbox"] [role="group"],[data-testid="stNumberInputContainer"],div[data-baseweb="select"] > div,div[data-baseweb="input"]{
 background:#fff !important;border:1px solid #e3e0d9 !important;border-radius:10px !important;box-shadow:none !important;}
[data-testid="stSelectbox"] [role="group"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within{border-color:var(--blue-2) !important;box-shadow:0 0 0 3px #e9eeff !important;}
[data-testid="stSelectbox"] input,[data-testid="stNumberInput"] input,div[data-baseweb="select"] *{
 background:transparent !important;color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important;}
[data-testid="stSelectbox"] input::placeholder,[data-testid="stNumberInput"] input::placeholder{color:#a7aebb !important;-webkit-text-fill-color:#a7aebb !important;}
[data-testid="stSelectbox"] button,[data-testid="stNumberInput"] button{background:transparent !important;color:#7d8697 !important;border:0 !important;}
[data-testid="stSelectbox"] button svg,[data-testid="stNumberInput"] button svg{color:#7d8697 !important;fill:#7d8697 !important;}
[data-testid="stNumberInput"] button:hover{background:var(--blue-soft) !important;}
[role="listbox"],[role="listbox"] [role="option"],div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{
 background:#fff !important;color:var(--ink) !important;}
[role="listbox"] [role="option"] *{color:var(--ink) !important;}
[role="listbox"] [role="option"]:hover,[role="listbox"] [aria-selected="true"]{background:var(--blue-soft) !important;}

/* 페이지 이동 중 이전 페이지 잔상 숨기기: 새로 그려지기 전의 이전 요소(stale)를 표시하지 않음 */
[data-testid="stMain"] [data-stale="true"]{display:none !important;}
/* 입력칸 클릭 시 나오는 'Press Enter to submit form' 안내 숨기기 */
[data-testid="InputInstructions"]{display:none !important;}
.side-sub{font-size:10px;font-weight:500;opacity:.75;margin-top:3px;}

/* ---------- 차트 애니메이션: 0에서 실제 값까지 서서히 ---------- */
@keyframes si-grow-w{from{width:0;}}
@keyframes si-grow-h{from{height:0;}}
@keyframes si-draw{to{stroke-dashoffset:0;}}
@keyframes si-fade{from{opacity:0;}}
@keyframes si-pop{from{opacity:0;transform:scale(.85);}}
.fill{animation:si-grow-w 1.6s cubic-bezier(.45,.05,.25,1) .25s both;}
.bar-row:nth-child(2) .fill{animation-delay:.37s;} .bar-row:nth-child(3) .fill{animation-delay:.49s;}
.bar-row:nth-child(4) .fill{animation-delay:.61s;} .bar-row:nth-child(5) .fill{animation-delay:.73s;}
.bar-row:nth-child(6) .fill{animation-delay:.85s;}
.bar-value,.column-pct{animation:si-fade 1s ease .6s both;}
.column-bar{animation:si-grow-h 1.6s cubic-bezier(.45,.05,.25,1) .25s both;}
.column-group:nth-child(2) .column-bar{animation-delay:.4s;} .column-group:nth-child(3) .column-bar{animation-delay:.55s;}
.column-group:nth-child(4) .column-bar{animation-delay:.7s;}
.anim-draw{animation:si-draw 1.8s cubic-bezier(.4,0,.2,1) .25s forwards;}
.anim-fade{animation:si-fade 1s ease .5s both;}
.anim-fade-late{animation:si-fade .6s ease 1.8s both;}
.heatmap > div{animation:si-pop .6s cubic-bezier(.45,.05,.25,1) both;}
.heatmap > div:nth-child(n+6){animation-delay:.1s;} .heatmap > div:nth-child(n+11){animation-delay:.2s;}
.heatmap > div:nth-child(n+16){animation-delay:.3s;} .heatmap > div:nth-child(n+21){animation-delay:.4s;}
.matrix td.filled{animation:si-pop .6s cubic-bezier(.45,.05,.25,1) .15s both;}

/* 고객 정보 입력 카드: 제목 + 폼을 하나의 흰 카드로 */
.st-key-predict_card{background:var(--paper);border:1px solid #efece6;border-radius:18px;box-shadow:var(--shadow);
 padding:22px;gap:0 !important;}
.card-head-only .sec-head{margin-bottom:18px;}
.card-head-only{font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;line-height:normal;}

/* 예측 폼 섹션 제목 */
.form-h{font-size:12px;font-weight:800;color:var(--ink);margin:2px 0 4px;}
.form-h.sep{border-top:1px solid var(--line);padding-top:14px;margin-top:6px;}

/* form widgets */
div[data-testid="stForm"]{border:1px solid var(--line);border-radius:14px;background:#fdfcfa;padding:18px;}
.stButton>button,.stFormSubmitButton>button{width:100%;border:0;color:#fff;font-weight:700;border-radius:12px;padding:12px;
 background:linear-gradient(90deg,var(--blue),var(--blue-2));box-shadow:0 8px 18px rgba(47,98,240,.28);}
.stButton>button:hover,.stFormSubmitButton>button:hover{color:#fff;filter:brightness(1.05);border:0;}
.stButton>button p,.stFormSubmitButton>button p{color:#fff !important;font-weight:700;}

@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,1fr);}.hero{grid-template-columns:1fr;}.hero-art{display:none;}}
@media (max-width:720px){.kpis{grid-template-columns:1fr;}.bar-row{grid-template-columns:22px 100px 1fr 44px;}}
</style>
""")


BAG_SVG = (
    '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none">'
    '<path d="M6 8h12l-1 12H7L6 8Z" fill="#fff"/>'
    '<path d="M9 8V6a3 3 0 0 1 6 0v2" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    '</svg>'
)


def html(content: str) -> None:
    """HTML을 그대로 출력.

    st.html은 마크다운 해석을 거치지 않아 st.markdown보다 훨씬 빠르게 그려진다.
    다만 st.html은 <svg>를 지워 버리므로, SVG(아이콘·꺾은선·ROC 그래프)가 든 블록만 st.markdown으로 출력한다.
    """
    if hasattr(st, "html") and "<svg" not in content:
        st.html(content)
    else:  # 구버전 Streamlit
        st.markdown(content, unsafe_allow_html=True)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


@st.cache_data(show_spinner=False)
def load_data(path_text: str, modified_time: float) -> pd.DataFrame:
    data = pd.read_csv(path_text)
    missing = REQUIRED_DATA_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
    return data


def register_legacy_sklearn_pickle_compatibility() -> None:
    """scikit-learn 1.6.1에서 저장된 ColumnTransformer pickle 로딩 호환 처리."""
    import sklearn.compose._column_transformer as column_transformer_module
    if not hasattr(column_transformer_module, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _RemainderColsList.__module__ = "sklearn.compose._column_transformer"
        column_transformer_module._RemainderColsList = _RemainderColsList


def build_mlp(input_dim: int, hidden_dims: list[int]):
    """학습 노트북과 같은 구조: Linear -> ReLU -> Linear -> ReLU -> Linear(로짓 1개)."""
    import torch.nn as nn

    layers, prev = [], input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class ChurnMLP:
    """PyTorch MLP(.pth) + sklearn 전처리기(.pkl)로 이탈 확률을 계산한다.

    전처리기는 scikit-learn 1.6.1에서 저장됐다. 다른 버전에서 transform()을 직접 호출하면
    내부 속성 차이로 오류가 나므로(예: SimpleImputer '_fill_dtype'), 학습 때 저장된
    중앙값·평균·표준편차·범주 목록만 꺼내 NumPy로 같은 변환을 수행한다.
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        import joblib
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = build_mlp(checkpoint["input_dim"], checkpoint.get("hidden_dims", [32, 16]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(checkpoint.get("threshold", 0.5))

        register_legacy_sklearn_pickle_compatibility()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artifact = joblib.load(preprocessor_path)
        self.feature_columns = list(artifact["feature_columns"])
        self.feedback_mapping = dict(artifact["feedback_group_mapping"])
        if self.feature_columns != list(checkpoint.get("feature_columns", self.feature_columns)):
            raise ValueError("모델과 전처리기의 입력 컬럼이 다릅니다. 같은 학습에서 저장한 파일인지 확인하세요.")

        transformer = artifact["preprocessor"]
        numeric = transformer.named_transformers_["numeric"]
        categorical = transformer.named_transformers_["categorical"]
        self.numeric_cols = list(artifact["numeric_cols"])
        self.categorical_cols = list(artifact["categorical_cols"])
        self.num_median = np.asarray(numeric.named_steps["imputer"].statistics_, dtype=np.float64)
        self.num_mean = np.asarray(numeric.named_steps["scaler"].mean_, dtype=np.float64)
        self.num_scale = np.asarray(numeric.named_steps["scaler"].scale_, dtype=np.float64)
        self.cat_fill = list(categorical.named_steps["imputer"].statistics_)
        self.cat_categories = [list(c) for c in categorical.named_steps["onehot"].categories_]

        input_dim = len(self.numeric_cols) + sum(len(c) for c in self.cat_categories)
        if input_dim != checkpoint["input_dim"]:
            raise ValueError(f"전처리 결과({input_dim}개)와 모델 입력({checkpoint['input_dim']}개) 크기가 다릅니다.")

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """ColumnTransformer(numeric: 중앙값 대체→표준화, categorical: 최빈값 대체→원-핫)과 같은 변환."""
        num = frame[self.numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        for i, median in enumerate(self.num_median):
            num[~np.isfinite(num[:, i]), i] = median
        num = (num - self.num_mean) / self.num_scale

        blocks = [num]
        for col, fill, categories in zip(self.categorical_cols, self.cat_fill, self.cat_categories):
            values = frame[col].astype(object).where(frame[col].notna(), fill).to_numpy()
            # handle_unknown="ignore": 학습 때 없던 값은 모든 열이 0
            blocks.append(np.column_stack([values == c for c in categories]).astype(np.float64))
        return np.column_stack(blocks).astype(np.float32)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        with torch.no_grad():
            return torch.sigmoid(self.model(torch.from_numpy(self.transform(frame)))).numpy().ravel()


def prepare_model_input(data: pd.DataFrame, feedback_mapping: dict) -> pd.DataFrame:
    """정제 CSV에서 모델 입력 4개 컬럼을 만든다. feedback -> feedback_group 변환 포함."""
    return pd.DataFrame({
        "membership_category": data["membership_category"].astype("object"),
        "points_in_wallet": pd.to_numeric(data["points_in_wallet"], errors="coerce"),
        "avg_transaction_value": pd.to_numeric(data["avg_transaction_value"], errors="coerce"),
        "feedback_group": data["feedback"].map(feedback_mapping),
    }, index=data.index)


@st.cache_resource(show_spinner=False)
def start_model_loader(model_path_text: str, model_modified_time: float,
                       preprocessor_path_text: str, preprocessor_modified_time: float) -> dict:
    """앱이 켜질 때 한 번만 백그라운드에서 모델을 불러온다.

    PyTorch import가 몇 초 걸리므로, 요약·데이터 분석 화면은 모델을 기다리지 않고 바로 보여주고
    예측·모델 성능 화면에서만 로딩이 끝날 때까지 기다린다.
    """
    state = {"model": None, "error": None, "done": threading.Event(), "seconds": None}

    def work():
        started = time.perf_counter()
        try:
            state["model"] = ChurnMLP(model_path_text, preprocessor_path_text)
        except ModuleNotFoundError as exc:
            state["error"] = f"패키지 미설치: {exc.name} → pip install {exc.name} 실행 후 앱을 다시 켜 주세요."
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["seconds"] = time.perf_counter() - started
            state["done"].set()

    threading.Thread(target=work, daemon=True).start()
    return state


@st.cache_data(show_spinner="모델 성능을 계산하는 중입니다...")
def evaluate_mlp(_mlp: ChurnMLP, model_key: str, data_path_text: str, data_modified_time: float) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, roc_curve,
    )

    data = pd.read_csv(data_path_text)
    x = prepare_model_input(data, _mlp.feedback_mapping)
    y = pd.to_numeric(data["churn_risk_score"], errors="raise").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    probabilities = _mlp.predict_proba(x_test)
    predictions = (probabilities >= _mlp.threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model_source": "mlp_churn_4features.pth + mlp_churn_preprocessor.pkl (입력: "
                        + ", ".join(_mlp.feature_columns) + ")",
        "evaluation_note": "모델 학습 당시의 분할 방식이 기록되어 있지 않아, 완전히 독립된 최종 테스트셋인지 추가 확인이 필요합니다.",
    }


@st.cache_data(show_spinner=False)
def cached_analysis(path_text: str, modified_time: float) -> dict:
    return create_analysis_tables(load_data(path_text, modified_time))


def rate_table(data: pd.DataFrame, group_col: str, label_col: str = "churn_risk_score") -> pd.DataFrame:
    result = data.groupby(group_col, dropna=False)[label_col].agg(customers="size", rate="mean").reset_index()
    result["rate_pct"] = result["rate"] * 100
    return result


def quartile_series(series: pd.Series, labels: list[str]) -> pd.Series:
    # 동일 값이 많은 데이터에서도 정확히 4개 표본 구간을 만들기 위해 순위 기반 qcut 사용
    ranks = pd.to_numeric(series, errors="coerce").rank(method="first")
    return pd.qcut(ranks, q=4, labels=labels)


def create_analysis_tables(data: pd.DataFrame) -> dict:
    target = "churn_risk_score"
    result: dict = {}

    membership = rate_table(data, "membership_category").sort_values("rate", ascending=False)
    result["membership"] = [
        (str(row.membership_category), int(row.customers), float(row.rate_pct))
        for row in membership.itertuples(index=False)
    ]

    point_labels = ["하위 25%", "중하위 25%", "중상위 25%", "상위 25%"]
    point_bins = quartile_series(data["points_in_wallet"], point_labels)
    points_df = pd.DataFrame({"구간": point_bins, target: data[target]})
    points_agg = points_df.groupby("구간", observed=False)[target].agg(customers="size", rate="mean").reset_index()
    result["points"] = [(str(r["구간"]), int(r["customers"]), float(r["rate"] * 100)) for _, r in points_agg.iterrows()]

    feedback_rates = rate_table(data, "feedback").sort_values("rate", ascending=False)
    positive_rows = feedback_rates[feedback_rates["feedback"].isin(POSITIVE_FEEDBACK)]
    negative_rows = feedback_rates[~feedback_rates["feedback"].isin(POSITIVE_FEEDBACK)]
    feedback_items = [
        (str(r.feedback), int(r.customers), float(r.rate_pct), False)
        for r in negative_rows.itertuples(index=False)
    ]
    if not positive_rows.empty:
        positive_count = int(positive_rows["customers"].sum())
        positive_rate = float((positive_rows["rate"] * positive_rows["customers"]).sum() / positive_count * 100)
        feedback_items.append(("긍정 피드백 전체", positive_count, positive_rate, True))
    feedback_items.sort(key=lambda item: item[2], reverse=True)
    result["feedback"] = feedback_items

    transaction_labels = ["하위 25%", "중하위 25%", "중상위 25%", "상위 25%"]
    transaction_bins = quartile_series(data["avg_transaction_value"], transaction_labels)
    trans_df = pd.DataFrame({"구간": transaction_bins, target: data[target]})
    trans_agg = trans_df.groupby("구간", observed=False)[target].agg(customers="size", rate="mean").reset_index()
    result["transaction"] = [(str(r["구간"]), int(r["customers"]), float(r["rate"] * 100)) for _, r in trans_agg.iterrows()]

    login_labels = ["1–4일", "5–8일", "9–14일", "15일 이상"]
    login_bins = pd.cut(
        pd.to_numeric(data["avg_frequency_login_days"], errors="coerce"),
        bins=[-np.inf, 4, 8, 14, np.inf], labels=login_labels,
    )
    login_df = pd.DataFrame({"구간": login_bins, target: data[target]})
    login_agg = login_df.groupby("구간", observed=False)[target].agg(customers="size", rate="mean").reset_index()
    result["login"] = [(str(r["구간"]), int(r["customers"]), float(r["rate"] * 100)) for _, r in login_agg.iterrows()]

    heat_df = pd.DataFrame({"login": login_bins, "transaction": transaction_bins, target: data[target]})
    heat = heat_df.pivot_table(index="login", columns="transaction", values=target, aggfunc="mean", observed=False)
    heat = heat.reindex(index=login_labels, columns=transaction_labels)
    heat_count = heat_df.pivot_table(index="login", columns="transaction", values=target, aggfunc="count", observed=False)
    heat_count = heat_count.reindex(index=login_labels, columns=transaction_labels)
    result["heat"] = heat.to_numpy(dtype=float) * 100
    result["heat_counts"] = heat_count.to_numpy(dtype=float)
    result["heat_rows"] = login_labels
    result["heat_cols"] = transaction_labels
    return result


def sec_head(icon: str, title: str, help_text: str, pill: str | None = None, warn: bool = False) -> str:
    pill_html = f'<span class="pill{" warn" if warn else ""}">{html_lib.escape(pill)}</span>' if pill else ""
    return (
        f'<div class="sec-head"><div><div class="sec-title"><span class="sec-ico">{icon}</span>{html_lib.escape(title)}</div>'
        f'<div class="sec-help">{html_lib.escape(help_text)}</div></div>{pill_html}</div>'
    )


def bar_rows(items, green: bool = False) -> str:
    rows = []
    for rank, item in enumerate(items, 1):
        if len(item) >= 4:
            label, count, value, is_green = item[:4]
        elif len(item) == 3:
            label, count, value = item
            is_green = green
        else:
            label, value = item
            count, is_green = None, green
        width = min(max(float(value), 1.5), 100.0)
        count_html = f'<em>{int(count):,}명</em>' if count is not None else ""
        color_class = " green" if is_green else ""
        rows.append(
            f'<div class="bar-row"><span class="rank">{rank}</span>'
            f'<span class="bar-label">{html_lib.escape(str(label))}{count_html}</span>'
            f'<div class="track"><div class="fill{color_class}" style="width:{width:.2f}%"></div></div>'
            f'<span class="bar-value">{float(value):.1f}%</span></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def kpi_grid(items, empty: bool = False) -> str:
    cells = []
    for icon, css_class, label, value, foot in items:
        value_class = "kpi-value metric-empty" if value == "--" else "kpi-value"
        foot_class = "kpi-foot gray" if value == "--" or empty else "kpi-foot"
        cells.append(
            f'<div class="kpi"><div class="kpi-ico {css_class}">{icon}</div><div>'
            f'<div class="kpi-label">{html_lib.escape(str(label))}</div>'
            f'<div class="{value_class}">{html_lib.escape(str(value))}</div>'
            f'<div class="{foot_class}">{html_lib.escape(str(foot))}</div></div></div>'
        )
    return '<div class="kpis">' + "".join(cells) + "</div>"


def heat_class(value: float) -> str:
    return "h4" if value >= 65 else "h3" if value >= 50 else "h2" if value >= 40 else "h1"


def column_chart(items) -> str:
    max_value = max([item[2] for item in items] + [1])
    groups = []
    labels = []
    for label, count, value in items:
        height = max(4.0, float(value) / max_value * 84)
        groups.append(
            f'<div class="column-group"><span class="column-pct">{value:.1f}%</span>'
            f'<div class="column-bar" style="height:{height:.1f}%"></div></div>'
        )
        labels.append(f'<span>{html_lib.escape(label)}</span>')
    return '<div class="column-chart">' + "".join(groups) + '</div><div class="column-labels">' + "".join(labels) + '</div>'


def polyline_length(coords) -> float:
    """선 그리기 애니메이션용: 꺾은선 전체 길이(stroke-dasharray 값)."""
    return float(sum(np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(coords, coords[1:]))) + 1


def login_svg(items) -> str:
    xs = [78, 207, 336, 465]
    values = [float(item[2]) for item in items]
    labels = [item[0] for item in items]
    def y(value):
        return 170 - min(max(value, 0), 100) * 1.35
    points = " ".join(f"{x},{y(v):.1f}" for x, v in zip(xs, values))
    markers = "".join(
        f'<circle cx="{x}" cy="{y(v):.1f}" r="5" fill="#fff" stroke="#2f62f0" stroke-width="3"/>'
        f'<text x="{x-16}" y="{y(v)-12:.1f}" fill="#3d4557" font-size="12" font-weight="700">{v:.1f}%</text>'
        f'<text x="{x-22}" y="198" fill="#8a92a1" font-size="10">{html_lib.escape(lbl)}</text>'
        for x, v, lbl in zip(xs, values, labels)
    )
    grid = "".join(
        f'<line x1="52" y1="{yy}" x2="500" y2="{yy}" stroke="#eeece7"/>'
        f'<text x="12" y="{yy+4}" fill="#8a92a1" font-size="10">{pct}%</text>'
        for yy, pct in [(35, 100), (80, 66), (125, 33), (170, 0)]
    )
    polygon = f"{points} 465,170 78,170"
    line_len = polyline_length([(x, y(v)) for x, v in zip(xs, values)])
    return (
        '<svg viewBox="0 0 520 220" style="width:100%;height:auto" role="img" aria-label="로그인 간격별 이탈 라벨 비율">'
        '<defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2f62f0" stop-opacity=".22"/>'
        '<stop offset="1" stop-color="#2f62f0" stop-opacity="0"/></linearGradient></defs>'
        f'{grid}<polygon class="anim-fade" points="{polygon}" fill="url(#ag)"/>'
        f'<polyline class="anim-draw" points="{points}" fill="none" stroke="#2f62f0" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="{line_len:.0f}" stroke-dashoffset="{line_len:.0f}"/>'
        f'<g class="anim-fade-late">{markers}</g></svg>'
    )


def heatmap_html(tables: dict) -> str:
    cells = '<div class="heat-h">로그인 간격</div>' + "".join(
        f'<div class="heat-h">{html_lib.escape(col)}</div>' for col in tables["heat_cols"]
    )
    heat = tables["heat"]
    counts = tables["heat_counts"]
    for row_index, row_label in enumerate(tables["heat_rows"]):
        cells += f'<div class="heat-r">{html_lib.escape(row_label)}</div>'
        for col_index in range(len(tables["heat_cols"])):
            value = heat[row_index, col_index]
            count = counts[row_index, col_index]
            if np.isnan(value) or count == 0:
                cells += '<div class="heat-h">자료 없음</div>'
            else:
                cells += f'<div class="{heat_class(float(value))}" title="표본 {int(count):,}명">{value:.1f}%</div>'
    return '<div class="heatmap">' + cells + '</div>'


def roc_svg(fpr, tpr) -> str:
    # 화면 표시용으로 ROC 포인트를 최대 70개로 균등 샘플링
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if len(fpr) > 70:
        indices = np.linspace(0, len(fpr) - 1, 70).astype(int)
        fpr, tpr = fpr[indices], tpr[indices]
    x0, x1, y0, y1 = 52, 500, 170, 35
    coords = [(x0 + x*(x1-x0), y0 - y*(y0-y1)) for x, y in zip(fpr, tpr)]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    line_len = polyline_length(coords)
    return (
        '<svg viewBox="0 0 520 220" style="width:100%;height:auto" role="img" aria-label="ROC curve">'
        '<line x1="52" y1="35" x2="52" y2="170" stroke="#dfe3ec"/>'
        '<line x1="52" y1="170" x2="500" y2="170" stroke="#dfe3ec"/>'
        '<line x1="52" y1="170" x2="500" y2="35" stroke="#c9ced8" stroke-dasharray="5 5"/>'
        '<text x="10" y="39" fill="#8a92a1" font-size="10">1.0</text>'
        '<text x="14" y="174" fill="#8a92a1" font-size="10">0</text>'
        '<text x="48" y="194" fill="#8a92a1" font-size="10">0</text>'
        '<text x="488" y="194" fill="#8a92a1" font-size="10">1.0</text>'
        f'<polyline class="anim-draw" points="{points}" fill="none" stroke="#2f62f0" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="{line_len:.0f}" stroke-dashoffset="{line_len:.0f}"/>'
        '</svg>'
    )


# ------------------------------------------------------------------
# 실제 데이터 및 모델 로드
# ------------------------------------------------------------------
data_path = first_existing(DATA_CANDIDATES)
model_path = first_existing(MODEL_CANDIDATES)
preprocessor_path = first_existing(PREPROCESSOR_CANDIDATES)

if data_path is None:
    st.error("정제 CSV를 찾지 못했습니다. data 폴더에 [2nd] customer_churn_ecommerce_cleaned.csv 파일을 넣어 주세요.")
    st.stop()
if model_path is None:
    st.error("MLP 모델 파일 mlp_churn_4features.pth를 models 폴더에 넣어 주세요.")
    st.stop()
if preprocessor_path is None:
    st.error("전처리 파일 mlp_churn_preprocessor.pkl을 models 폴더에 넣어 주세요.")
    st.stop()

try:
    data = load_data(str(data_path), data_path.stat().st_mtime)
except Exception as exc:
    st.error(f"데이터를 읽지 못했습니다: {type(exc).__name__}: {exc}")
    st.stop()

analysis = cached_analysis(str(data_path), data_path.stat().st_mtime)
row_count = len(data)
label_rate = float(data["churn_risk_score"].mean() * 100)
feature_count = len([column for column in data.columns if column != "churn_risk_score"])
missing_count = int(data.isna().sum().sum())

model_loader = start_model_loader(
    str(model_path), model_path.stat().st_mtime,
    str(preprocessor_path), preprocessor_path.stat().st_mtime,
)


def get_model():
    """모델이 필요한 화면에서만 호출. 아직 로딩 중이면 끝날 때까지 기다린다."""
    if not model_loader["done"].is_set():
        with st.spinner("MLP 모델을 불러오는 중입니다..."):
            model_loader["done"].wait()
    return model_loader["model"], model_loader["error"]


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
if "page" not in st.session_state:
    requested_page = st.query_params.get("page", PAGES[0])
    st.session_state.page = requested_page if requested_page in PAGES else PAGES[0]


def go_to(name: str) -> None:
    st.session_state.page = name
    st.query_params["page"] = name  # 새로고침해도 같은 페이지 유지


page = st.session_state.page

with st.sidebar:
    html(
        f'<div class="brand"><div class="brand-mark">{BAG_SVG.format(s=22)}</div>'
        '<div><div class="brand-name">ShopInsight</div><div class="brand-sub">E-Commerce Analytics</div></div></div>'
    )
    # 페이지 전체를 새로 불러오지 않도록 링크 대신 버튼 + session_state로 이동
    for i, name in enumerate(PAGES):
        st.button(name, key=f"nav_{i}", type="primary" if name == page else "secondary",
                  on_click=go_to, args=(name,))
    html(
        f'<div class="side-card"><div class="brand-mark" style="width:36px;height:36px;border-radius:10px">{BAG_SVG.format(s=18)}</div>'
        '<strong>데이터로 더 많은 고객을,<br>더 오래.</strong>'
        '<p>고객 행동 데이터에서 이탈 위험 신호를 찾아 유지 전략의 근거를 만듭니다.</p></div>'
        '<div class="side-meta">© ShopInsight · Customer Intelligence<br>실제 CSV 분석</div>'
    )
    if model_loader["done"].is_set():
        if model_loader["model"] is not None:
            html('<div class="side-status st-on">MLP 모델 연결됨'
                 f'<div class="side-sub">로딩 {model_loader["seconds"]:.1f}초</div></div>')
        else:
            html('<div class="side-status st-off">모델 연결 확인 필요</div>')
    else:
        # 로딩이 끝나면 페이지를 옮기지 않아도 표시가 바뀌도록 0.5초마다 이 부분만 다시 확인
        @st.fragment(run_every=0.5)
        def model_status_poller():
            if model_loader["done"].is_set():
                st.rerun()  # 전체를 한 번 다시 그려 '연결됨'으로 바꾸고 확인을 멈춤
            html('<div class="side-status st-off">MLP 모델 불러오는 중…</div>')

        model_status_poller()

html(f'<div class="crumb">SHOPINSIGHT / {html_lib.escape(page)}</div>')

# ------------------------------------------------------------------
# Page: 요약
# ------------------------------------------------------------------
if page == "핵심 지표":
    html(f"""
<div class="hero">
  <div class="hero-text">
    <h1>데이터가 이끄는<br>이커머스의 새로운 성장</h1>
    <p>고객의 행동을 정밀하게 분석하여 이탈을 방지하고<br>더 많은 비즈니스 기회를 확보합니다.</p>
  </div>
  <div class="hero-art"><svg class="hero-illust" viewBox="0 0 600 260" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"> <defs> <linearGradient id="hsWhite" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#e6e8f4"/></linearGradient> <linearGradient id="hsWhiteSide" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9dcee"/><stop offset="1" stop-color="#c4c8e0"/></linearGradient> <linearGradient id="hsTan" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fbe6c2"/><stop offset="1" stop-color="#ebc690"/></linearGradient> <linearGradient id="hsPink" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f6e2f4"/><stop offset="1" stop-color="#dcb8e6"/></linearGradient> <linearGradient id="hsPinkSide" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9b3e2"/><stop offset="1" stop-color="#c79fd6"/></linearGradient> <linearGradient id="hsLav" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#d3d7ff"/><stop offset="1" stop-color="#97a2f0"/></linearGradient> <linearGradient id="hsLav2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eceefe"/><stop offset="1" stop-color="#c3c9f8"/></linearGradient> <linearGradient id="hsChrome" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#eef0f6"/><stop offset=".45" stop-color="#9ea4b8"/><stop offset=".7" stop-color="#d9dce6"/><stop offset="1" stop-color="#7f8599"/></linearGradient> <linearGradient id="hsNavy" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4a4f8c"/><stop offset="1" stop-color="#23264a"/></linearGradient> <linearGradient id="hsBox" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f8e3c4"/><stop offset="1" stop-color="#e8c89a"/></linearGradient> <radialGradient id="hsShadow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#5b5f9c" stop-opacity=".28"/><stop offset="1" stop-color="#5b5f9c" stop-opacity="0"/></radialGradient> </defs> <ellipse cx="315" cy="238" rx="285" ry="14" fill="url(#hsShadow)"/> <path d="M58 152 C58 128, 96 128, 96 152" fill="none" stroke="#e5c6ee" stroke-width="4" stroke-linecap="round"/> <polygon points="44,150 112,150 112,232 44,232" fill="url(#hsPink)"/> <polygon points="112,150 126,156 126,228 112,232" fill="url(#hsPinkSide)"/> <path d="M104 112 C104 72, 150 72, 150 112" fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round"/> <polygon points="88,110 166,110 166,234 88,234" fill="url(#hsTan)"/> <polygon points="166,110 180,116 180,230 166,234" fill="#dcb47e"/> <path d="M170 78 C170 26, 224 26, 224 78" fill="none" stroke="#1f2440" stroke-width="5.5" stroke-linecap="round"/> <polygon points="142,76 252,76 252,237 142,237" fill="url(#hsWhite)"/> <polygon points="252,76 276,84 276,232 252,237" fill="url(#hsWhiteSide)"/> <line x1="142" y1="92" x2="252" y2="92" stroke="#e3e5f1" stroke-width="1.5"/> <g fill="#a7abbd" font-family="Pretendard, 'Malgun Gothic', sans-serif" font-size="11.5" font-weight="500" letter-spacing="1.6" text-anchor="middle"> <text x="197" y="128">A</text><text x="197" y="147">BETTER</text><text x="197" y="166">SHOPPING</text><text x="197" y="185">EXPERIENCE</text> </g> <polygon points="352,104 396,104 396,224 352,224" fill="url(#hsLav2)"/> <polygon points="300,96 362,96 362,226 300,226" fill="url(#hsLav)"/> <polygon points="362,96 374,101 374,222 362,226" fill="#8c97e6"/> <g stroke-linecap="round" stroke-linejoin="round" fill="none"> <path d="M286 104 L304 118" stroke="#262b45" stroke-width="7"/> <polygon points="304,118 458,108 442,190 320,193" stroke="url(#hsChrome)" stroke-width="3.2"/> <g stroke="#b7bccc" stroke-width="1.6"> <line x1="321.1" y1="116.9" x2="333.6" y2="192.7"/><line x1="338.2" y1="115.8" x2="347.1" y2="192.3"/> <line x1="355.3" y1="114.7" x2="360.7" y2="192.0"/><line x1="372.4" y1="113.6" x2="374.2" y2="191.7"/> <line x1="389.6" y1="112.4" x2="387.8" y2="191.3"/><line x1="406.7" y1="111.3" x2="401.3" y2="191.0"/> <line x1="423.8" y1="110.2" x2="414.9" y2="190.7"/><line x1="440.9" y1="109.1" x2="428.4" y2="190.3"/> <line x1="309.3" y1="143" x2="452.7" y2="135"/><line x1="314.7" y1="168" x2="447.3" y2="162"/> </g> <path d="M458 108 L466 90 L474 88" stroke="url(#hsChrome)" stroke-width="3.2"/> <path d="M320 193 L322 214 L446 214 L442 190" stroke="url(#hsChrome)" stroke-width="3"/> <path d="M330 204 L440 203" stroke="#c3c7d4" stroke-width="2"/> <path d="M336 214 L334 222 M366 214 L366 224 M410 214 L410 224 M438 214 L440 222" stroke="#9ea4b8" stroke-width="2.4"/> </g> <g fill="#272c46"><circle cx="333" cy="226" r="7"/><circle cx="366" cy="228" r="7"/><circle cx="410" cy="228" r="7"/><circle cx="441" cy="226" r="7"/></g> <g fill="#c3c8d8"><circle cx="333" cy="226" r="2.4"/><circle cx="366" cy="228" r="2.4"/><circle cx="410" cy="228" r="2.4"/><circle cx="441" cy="226" r="2.4"/></g> <path d="M478 154 C478 136, 502 136, 502 154" fill="none" stroke="#e5c6ee" stroke-width="3.5" stroke-linecap="round"/> <polygon points="468,152 508,152 508,232 468,232" fill="url(#hsPink)"/> <polygon points="508,152 518,157 518,228 508,232" fill="url(#hsPinkSide)"/> <rect x="540" y="172" width="54" height="62" rx="24" fill="url(#hsNavy)"/> <path d="M567 190 L570 200 L580 203 L570 206 L567 216 L564 206 L554 203 L564 200 Z" fill="#c9cdf6" opacity=".75"/> <rect x="506" y="192" width="50" height="42" rx="3" fill="url(#hsBox)"/> <rect x="503" y="186" width="56" height="10" rx="2" fill="#f3d7b0"/> <rect x="527" y="186" width="8" height="48" fill="#e9b9da"/> <rect x="511" y="170" width="40" height="16" rx="2" fill="#efd3f2"/> <rect x="527" y="170" width="8" height="16" fill="#dba8d6"/> </svg></div>
</div>""")

    kpis = [
        ("👥", "i-blue", "분석 고객 수", f"{row_count:,}명", "전처리 완료 데이터"),
        ("⚠", "i-rose", "이탈 라벨 비율", f"{label_rate:.1f}%", "churn_risk_score = 1 비율"),
    ]
    # HTML 시안과 똑같이 CSS grid 한 덩어리로 그림:
    #  - st.columns는 칸마다 높이가 따로 정해져 두 카드 높이가 어긋나고, 칸 사이 간격도 시안(16px)보다 넓음
    #  - grid는 같은 줄의 카드 높이를 자동으로 맞춰 줌(align-items: stretch)
    html('<div class="row-grid">' + "".join(kpi_grid([item]).replace('class="kpis"', 'class="kpis one"', 1)
                                            for item in kpis) + '</div>')
    html(
        '<div class="row-grid">'
        '<div class="card">' + sec_head(
            "▥", "회원 등급별 이탈률",
            "각 등급에서 churn_risk_score = 1인 비율 · 고객 수 함께 표시", "실제 CSV 집계"
        ) + bar_rows([(lbl if lbl == "No Membership" else lbl.replace(" Membership", ""), *rest)
                  for lbl, *rest in analysis["membership"]]) + "</div>"
        '<div class="card">' + sec_head(
            "◆", "포인트 구간별 이탈률",
            "보유 포인트 분포를 4개 분위로 나눈 결과", "실제 CSV 집계"
        ) + bar_rows(analysis["points"], green=True)
        + '<div class="foot-note">관찰된 라벨 비율이며, 포인트가 이탈의 원인이라는 의미는 아닙니다.</div></div>'
        '</div>'
    )

# ------------------------------------------------------------------
# Page: 고객 이탈 예측 (입력 5개: HTML 시안 기준)
# ------------------------------------------------------------------
elif page == "고객 이탈 예측":
    c1, c2 = st.columns([1.1, 0.9], gap="medium")
    # 제목과 입력 폼을 흰 카드 하나로 묶음 (HTML 시안과 같은 모양)
    with c1, st.container(key="predict_card"):
        html('<div class="card-head-only">' + sec_head(
            "◎", "고객 정보 입력", "멤버십, 포인트, 고객 평가와 거래 내역을 입력합니다.", "필수 정보 5개"
        ) + "</div>")
        with st.form("predict_form"):
            html('<div class="form-h">고객 정보</div>')
            f1, f2 = st.columns(2, vertical_alignment="bottom")
            membership = f1.selectbox("회원 등급", MEMBERSHIP_OPTIONS, index=None, placeholder="선택해 주세요")
            points = f2.number_input("보유 포인트 (만원)", value=None, placeholder="예: 720", step=0.01)
            html('<div class="form-h sep">고객 평가 및 거래 내역</div>')
            f1, f2 = st.columns(2, vertical_alignment="bottom")
            total_value = f1.number_input("누적 이용 금액 (만원)", min_value=0.0, value=None,
                                          placeholder="예: 25 (= 25만원)", step=0.1, format="%.1f")
            total_count = f2.number_input("누적 이용 횟수", min_value=1, value=None, placeholder="예: 10", step=1)
            f1, _ = st.columns(2)
            feedback_group = f1.selectbox("고객 평가", list(FEEDBACK_GROUP_OPTIONS), index=None,
                                          placeholder="선택해 주세요")
            st.caption("금액은 만원 단위로 입력합니다. 평균 거래 금액 = 누적 이용 금액 × 10,000 ÷ 누적 이용 횟수")
            submitted = st.form_submit_button("이탈 확률 확인", width="stretch")

    with c2:
        result_html = (
            '<div class="result-placeholder"><div>'
            f'<div class="icon">{BAG_SVG.format(s=28)}</div>'
            '<strong>고객의 다음 행동을 예측합니다</strong>'
            '<p>고객 정보를 입력하고 예측을 실행하면 결과가 이곳에 표시됩니다.</p>'
            '</div></div>'
        )
        if submitted:
            mlp_model, model_error = get_model()
            MODEL_READY = mlp_model is not None
            if any(v is None for v in [membership, points, feedback_group, total_value, total_count]):
                result_html = ('<div class="result-placeholder"><div><strong>입력값을 모두 채워 주세요</strong>'
                               '<p>5개 항목이 모두 있어야 예측할 수 있습니다.</p></div></div>')
            elif not MODEL_READY:
                result_html = ('<div class="result-placeholder"><div><strong>MLP 모델을 불러오지 못했습니다</strong>'
                               f'<p>{html_lib.escape(model_error or "")}</p></div></div>')
            else:
                # 누적 이용 금액은 만원 단위로 입력받으므로 원 단위로 바꾼 뒤 횟수로 나눈다
                avg_value = float(total_value) * MANWON / int(total_count)
                customer = pd.DataFrame([{
                    "membership_category": membership,
                    "points_in_wallet": float(points),
                    "avg_transaction_value": avg_value,
                    "feedback_group": FEEDBACK_GROUP_OPTIONS[feedback_group],
                }])
                try:
                    probability = float(mlp_model.predict_proba(customer)[0])
                    risk = probability >= mlp_model.threshold
                    color = "var(--rose)" if risk else "var(--green)"
                    badge = ('<span class="badge risk">이탈 위험 고객</span>' if risk
                             else '<span class="badge safe">유지 예상 고객</span>')
                    result_html = (
                        '<div class="result-placeholder"><div><strong>예측 이탈 확률</strong>'
                        f'<div class="result-big" style="color:{color}">{probability*100:.1f}%</div>{badge}'
                        f'<p style="margin-top:12px">계산된 평균 거래 금액 {avg_value:,.0f}원 · '
                        f'이탈 확률 {mlp_model.threshold:.0%} 이상이면 이탈 위험으로 판정합니다.</p></div></div>'
                    )
                except Exception as exc:
                    result_html = ('<div class="result-placeholder"><div><strong>예측 중 오류가 발생했습니다</strong>'
                                   f'<p>{html_lib.escape(type(exc).__name__ + ": " + str(exc))}</p></div></div>')
        html('<div class="card">' + sec_head("✦", "예측 결과", "입력한 고객 한 명에 대한 모델 결과", "MLP")
             + result_html + "</div>")
        html('<div class="info-band">예측은 churn_risk_score 라벨 패턴을 학습한 결과입니다. '
             '타깃 정의와 데이터 누수 여부를 검증한 뒤 업무 의사결정에 사용하세요.</div>')


# ------------------------------------------------------------------
# Page: 데이터 분석
# ------------------------------------------------------------------
elif page == "이탈 원인 분석":
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(
            '<div class="card">' + sec_head(
                "▤", "피드백 유형별 이탈률", "피드백 유형별 churn_risk_score = 1 비율과 고객 수", "실제 CSV 집계"
            ) + bar_rows(analysis["feedback"])
            + '<div class="foot-note">긍정 피드백 4개 유형은 하나의 그룹으로 합산했습니다.</div></div>'
        )
    with c2:
        html(
            '<div class="card">' + sec_head(
                "₩", "평균 거래 금액 구간별 이탈률", "평균 거래 금액을 4개 분위로 나눈 이탈 라벨 비율", "실제 CSV 집계"
            ) + column_chart(analysis["transaction"])
            + '<div class="foot-note">각 구간은 전체 고객을 기준으로 한 사분위 구간입니다.</div></div>'
        )
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        html(
            '<div class="card">' + sec_head(
                "⌁", "로그인 간격별 이탈률", "평균 로그인 간격 구간별 이탈 라벨 비율", "실제 CSV 집계"
            ) + login_svg(analysis["login"])
            + '<div class="foot-note">로그인 간격 구간: 1–4일, 5–8일, 9–14일, 15일 이상</div></div>'
        )
    with c4:
        html(
            '<div class="card">' + sec_head(
                "▦", "로그인 간격 × 거래 금액", "두 행동 변수를 함께 봤을 때의 조합별 이탈 라벨 비율", "실제 CSV 집계"
            ) + heatmap_html(analysis)
            + '<div class="foot-note">각 셀에는 해당 조합의 고객 표본 수가 툴팁으로 표시됩니다.</div></div>'
        )
    html('<div class="info-band">모든 그래프는 정제 CSV의 churn_risk_score 라벨을 집계한 결과입니다. 변수 간 관계를 보여주며 인과관계를 증명하지 않습니다.</div>')

# ------------------------------------------------------------------
# Page: 모델 성능
# ------------------------------------------------------------------
elif page == "모델 성능":
    mlp_model, model_error = get_model()
    model_metrics, metrics_error = None, None
    if mlp_model is not None:
        try:
            model_key = f"{model_path}:{model_path.stat().st_mtime}:{preprocessor_path.stat().st_mtime}"
            model_metrics = evaluate_mlp(mlp_model, model_key, str(data_path), data_path.stat().st_mtime)
        except Exception as exc:
            metrics_error = f"{type(exc).__name__}: {exc}"
    if mlp_model is None or model_metrics is None:
        html('<div class="info-band">MLP 성능을 계산할 수 없습니다. models/mlp_churn_4features.pth, models/mlp_churn_preprocessor.pkl, torch 설치 여부를 확인하세요.</div>')
        if model_error or metrics_error:
            st.code(model_error or metrics_error)
    else:
        metrics = model_metrics
        score_items = [
            ("◎", "i-blue", "Accuracy", f"{metrics['accuracy']*100:.2f}%", "현재 데이터 검증 정확도"),
            ("⚑", "i-rose", "Recall · 이탈", f"{metrics['recall']*100:.2f}%", "실제 1 라벨을 찾은 비율"),
            ("⚖", "i-purple", "F1-score · 이탈", f"{metrics['f1']*100:.2f}%", "Precision과 Recall의 균형"),
            ("∿", "i-green", "ROC-AUC", f"{metrics['roc_auc']*100:.2f}%", "분류 구분 능력"),
        ]
        html(kpi_grid(score_items))
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            cm = metrics["confusion_matrix"]
            table = (
                '<table class="matrix"><thead><tr><th></th><th>예측 유지</th><th>예측 이탈</th></tr></thead><tbody>'
                f'<tr><th>실제 유지</th><td class="filled">{cm[0][0]:,}</td><td class="filled">{cm[0][1]:,}</td></tr>'
                f'<tr><th>실제 이탈</th><td class="filled">{cm[1][0]:,}</td><td class="filled">{cm[1][1]:,}</td></tr>'
                '</tbody></table>'
            )
            html('<div class="card">' + sec_head("▦", "혼동 행렬", "현재 CSV 평가 분할의 실제값과 예측값") + table + "</div>")
        with c2:
            auc_text = f"ROC-AUC {metrics['roc_auc']:.3f}"
            html('<div class="card">' + sec_head("⌁", "ROC Curve", auc_text) + roc_svg(metrics["fpr"], metrics["tpr"]) + "</div>")
        html(
            f'<div class="info-band">검증 분할: 학습 {metrics["train_rows"]:,}명 / 평가 {metrics["test_rows"]:,}명 · '
            'Stratified 80:20, random_state=42. 저장된 전처리기와 MLP 가중치를 그대로 사용해 현재 CSV의 평가 분할을 계산했습니다. '
            f'모델 구성: {html_lib.escape(metrics.get("model_source", "PyTorch MLP"))}. '
            + html_lib.escape(metrics.get("evaluation_note", ""))
            + '</div>'
        )

html('<div class="footer">ShopInsight · Customer Intelligence · 실제 CSV 및 저장된 PyTorch MLP 기반</div>')
