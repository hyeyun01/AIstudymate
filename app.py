import streamlit as st
import numpy as np
import joblib  # pickle 대신 joblib 사용

# ---------------------------------------------
# 1) 페이지 기본 설정
# ---------------------------------------------
st.set_page_config(page_title="AI StudyMate - 학습 성향 분석", layout="wide")
st.title("🧠 AI StudyMate - 학습 성향 진단")
st.write("30문항 설문을 기반으로 학습 성향을 분석하고, 맞춤형 하브루타 파트너 유형과 학습법을 추천합니다.")
st.divider()

# ---------------------------------------------
# 2) 설문 문항 정의 (Q1~Q30)
# ---------------------------------------------
questions = [
    "나는 문제를 풀기 전에 계획을 세우는 편이다.",
    "새로운 내용을 배우면 스스로 정리해보는 편이다.",
    "어려운 문제가 나오면 바로 질문하기보다는 먼저 스스로 해결하려 한다.",
    "친구와 함께 공부하면 더 잘 이해된다.",
    "다른 사람의 생각을 듣고 비교하는 것을 좋아한다.",
    "토론 활동이 나에게 도움이 된다.",
    "나는 혼자 공부하는 것이 더 편하다.",
    "학습 목표를 스스로 설정하는 편이다.",
    "틀린 문제를 다시 분석하는 데 시간을 투자한다.",
    "내가 모르는 것을 솔직하게 말하는 편이다.",
    "친구가 질문하면 쉽게 설명해주는 편이다.",
    "팀 활동에서 의견 조율을 잘하는 편이다.",
    "문제를 다양하게 바꿔보며 탐구하는 편이다.",
    "새로운 방식으로 문제를 해결하는 것을 좋아한다.",
    "원리를 이해해야 안심된다.",
    "나의 학습 습관을 스스로 점검한다.",
    "학습 스케줄을 지키려고 노력한다.",
    "모르는 것이 있으면 바로 검색하거나 찾는다.",
    "친구와 아이디어를 주고받는 것을 좋아한다.",
    "서로의 풀이를 비교해보는 활동을 좋아한다.",
    "질문을 활발하게 하는 편이다.",
    "문제의 다양한 경우를 실험하는 편이다.",
    "복잡한 문제를 단계적으로 나누어 생각한다.",
    "내가 이해한 것을 정리해서 말할 수 있다.",
    "수업 중 발표나 설명을 잘하는 편이다.",
    "오답을 분석하여 공부 방향을 조정한다.",
    "모둠 활동에서 주도적으로 참여한다.",
    "어려운 내용을 반복해서 탐구해본다.",
    "배운 내용을 다른 사람에게 설명해본다.",
    "다른 사람과 학습할 때 동기부여가 된다."
]

CHOICES = ["① 전혀 아니다", "② 아니다", "③ 보통이다", "④ 그렇다", "⑤ 매우 그렇다"]
responses = {}

st.subheader("📘 학습 성향 설문 (30문항)")

for i, question in enumerate(questions, start=1):
    st.write(f"**Q{i}. {question}**")
    choice = st.radio(
        "",
        CHOICES,
        key=f"q_{i}",
        horizontal=True
    )
    responses[f"Q{i}"] = CHOICES.index(choice) + 1
    st.markdown("---")

# ---------------------------------------------
# 3) 역량 점수 계산
# ---------------------------------------------
if st.button("🧪 학습 성향 분석 시작"):
    responses_array = np.array(list(responses.values()))

    Analytical_idx = [0, 2, 8, 14, 22]
    Collaborative_idx = [3, 4, 10, 11, 18, 19, 25]
    SelfDirected_idx = [1, 6, 7, 15, 16, 26]
    Questioning_idx = [5, 12, 13, 20, 21, 27, 28]

    Analytical = responses_array[Analytical_idx].mean()
    Collaborative = responses_array[Collaborative_idx].mean()
    SelfDirected = responses_array[SelfDirected_idx].mean()
    Questioning = responses_array[Questioning_idx].mean()

    profile_vector = np.array([Analytical, Collaborative, SelfDirected, Questioning]).reshape(1, -1)

    # ---------------------------------------------
    # 4) 저장된 scaler + K-means 모델 불러오기
    # ---------------------------------------------
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    # 스케일링 후 예측
    profile_scaled = scaler.transform(profile_vector)
    cluster = kmeans.predict(profile_scaled)[0]

    # ---------------------------------------------
    # 5) 군집명 / 하브루타 추천
    # ---------------------------------------------
    cluster_name_map = {
        0: "병아리 탐험가 🐣",
        1: "논리왕 🤓",
        2: "친구왕 🦄",
        3: "문제 해결 마스터 🕵️‍♂️"
    }

    partner_recommendation_map = {
        0: "학습 루틴이 잘 잡혀 있는 '자기주도형' 친구와 함께하면 기본기 형성이 빠릅니다.",
        1: "협력·소통이 강한 '관계 중심형' 친구와 페어를 이루면 이해폭이 넓어집니다.",
        2: "'탐구형/분석형' 친구와 함께하면 사고력이 균형 있게 성장합니다.",
        3: "협력 중심형 친구와 함께 활동하면 설명력·소통력이 보완됩니다."
    }

    study_tips_map = {
        0: """**학습 스타일 분석:**  
- 기초 개념 이해와 반복 학습을 잘해요.  
- 간단한 문제를 단계적으로 푸는 활동을 좋아해요.  
- 복잡한 문제를 혼자 탐구하는 것이 부족해요.  
- 심화 문제를 해결하는 경험이 약해요.  

**이렇게 공부하면 좋아요:**  
- 노트에 개념과 예제 문제를 정리해보기  
- 문제 풀이 과정을 말로 설명하며 반복  
- 쉬운 문제부터 단계별로 연습하여 자신감 쌓기  

**친구와 함께 공부할 때 역할:**  
- 친구와 함께 공부할 때는 ‘탐험가’ 역할을 맡아, 활동 계획과 기본 개념을 제시하면 좋아요.
""",
        1: """**학습 스타일 분석:**  
- 논리적으로 문제를 분석하고 체계적으로 푸는 것을 잘해요.  
- 혼자서 단계별 문제 해결과 계획 세우기를 좋아해요.  
- 협력 학습이나 토론을 통한 이해는 부족해요.  
- 창의적 문제 접근 경험이 약해요.  

**이렇게 공부하면 좋아요:**  
- 문제 풀이 계획표 작성 후 혼자 풀이  
- 어려운 문제를 여러 방법으로 해결하며 사고력 확장  
- 풀이 과정을 글로 정리하여 논리 구조 점검  

**친구와 함께 공부할 때 역할:**  
- 친구와 함께 공부할 때는 ‘분석가’ 역할을 맡아, 문제 접근 방법과 전략을 제시하면 좋아요.
""",
        2: """**학습 스타일 분석:**  
- 친구와 함께 공부하고 토론하며 이해하는 것을 잘해요.  
- 그룹 활동과 발표를 좋아해요.  
- 혼자서 계획 세우고 문제를 분석하는 능력은 부족해요.  
- 자기주도적 학습 경험이 약해요.  

**이렇게 공부하면 좋아요:**  
- 그룹 토론과 발표를 통해 문제 풀이 공유  
- 문제를 서로 설명하고 역할 분담 후 결과 정리  
- 글쓰기나 말로 설명하기로 이해한 내용을 기록  

**친구와 함께 공부할 때 역할:**  
- 친구와 함께 공부할 때는 ‘설명가’ 역할을 맡아, 이해한 내용을 공유하면 좋아요.
""",
        3: """**학습 스타일 분석:**  
- 복잡한 문제를 다양한 방법으로 탐구하고 해결하는 것을 잘해요.  
- 심화 문제와 응용 활동을 좋아해요.  
- 기초 개념 반복 학습은 부족해요.  
- 학습 루틴 관리 경험이 약해요.  

**이렇게 공부하면 좋아요:**  
- 문제 변형 및 응용 문제를 스스로 풀어보기  
- 학습 내용을 글로 정리하거나 친구에게 설명  
- 기초 개념 복습과 실수 분석으로 약점 보완  

**친구와 함께 공부할 때 역할:**  
- 친구와 함께 공부할 때는 ‘문제 해결사’ 역할을 맡아, 어려운 문제를 시도하고 전략을 공유하면 좋아요.
"""
    }

    cluster_name = cluster_name_map[cluster]
    partner_recommendation = partner_recommendation_map[cluster]
    study_tips = study_tips_map[cluster]

    # ---------------------------------------------
    # 6) 결과 출력
    # ---------------------------------------------
    st.subheader("📌 분석 결과 요약")
    st.metric("예측된 학습자 유형", cluster_name)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🎯 나의 역량 점수")
        st.write(f"- **Analytical(분석성)**: {Analytical:.2f}/5.00")
        st.write(f"- **Collaborative(협력성)**: {Collaborative:.2f}/5.00")
        st.write(f"- **Self-Directed(자기주도)**: {SelfDirected:.2f}/5.00")
        st.write(f"- **Questioning(탐구·질문성)**: {Questioning:.2f}/5.00")

    with col2:
        st.write("### 🤝 추천 하브루타 파트너 유형")
        st.info(partner_recommendation)

    st.divider()

    st.subheader("📇 나의 Strength Profile 카드")

    # -----------------------------
    # 7) Strength Profile 카드 HTML로 섹션별 출력
    # -----------------------------
    # study_tips를 섹션별로 분리
    # 예: "**학습 스타일 분석:**", "**이렇게 공부하면 좋아요:**", "**친구와 함께 공부할 때 역할:**"
    sections = study_tips.split("**")[1:]  # 맨 앞 공백 제거

    for sec in sections:
        try:
            title, content = sec.split(":**", 1)
        except ValueError:
            continue
        # 줄바꿈 처리
        content_html = content.replace("\n", "<br>").strip()

        st.markdown(
            f"""
            <div style="
                background-color:#f0f4f8;
                padding:18px; 
                border-radius:12px;
                margin-bottom:10px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            ">
                <h4 style="color:#1f4e79;">{title}</h4>
                <p style="margin:5px 0;">{content_html}</p>
            </div>
            """, unsafe_allow_html=True
        )
