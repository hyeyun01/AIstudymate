import streamlit as st
import numpy as np
import joblib
from partner_matching import match_partners
import pandas as pd

# ---------------------------------------------
# 보완형 매핑 (0↔3, 1↔2)
# ---------------------------------------------
complement_map = {
    0: 3,
    3: 0,
    1: 2,
    2: 1
}

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
# 3) 역량 점수 계산 및 학습자 유형 예측
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

    # 모델 불러오기
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")

    profile_scaled = scaler.transform(profile_vector)
    cluster = int(kmeans.predict(profile_scaled)[0])

    # 군집 정보 저장
    st.session_state['Analytical'] = Analytical
    st.session_state['Collaborative'] = Collaborative
    st.session_state['SelfDirected'] = SelfDirected
    st.session_state['Questioning'] = Questioning
    st.session_state['cluster'] = cluster

    cluster_name_map = {
        0: "병아리 탐험가 🐣",
        1: "논리왕 🤓",
        2: "친구왕 🦄",
        3: "문제 해결 마스터 🕵️‍♂️"
    }

    partner_recommendation_map = {
        0: "학습 루틴이 잘 잡혀 있는 '문제 해결 마스터 🕵️‍♂️' 친구와 함께하면 좋아요.",
        1: "'친구왕 🦄' 친구와 함께하면 협력적 활동에 강점을 보완할 수 있어요.",
        2: "'논리왕 🤓' 친구와 함께하면 사고력이 균형 있게 성장해요.",
        3: "'병아리 탐험가 🐣' 친구와 함께하면 기초 개념 보완에 도움이 돼요."
    }

    st.session_state['cluster_name'] = cluster_name_map[cluster]
    st.session_state['partner_recommendation'] = partner_recommendation_map[cluster]

    # 학생 데이터 처리
    if 'students_processed' not in st.session_state:
        df_students_raw = pd.read_csv("real_students.csv")
        st.session_state['students_processed'] = match_partners(df_students_raw)


# ---------------------------------------------
# 6) 결과 출력
# ---------------------------------------------
if 'cluster' in st.session_state:

    Analytical = st.session_state['Analytical']
    Collaborative = st.session_state['Collaborative']
    SelfDirected = st.session_state['SelfDirected']
    Questioning = st.session_state['Questioning']
    cluster = st.session_state['cluster']

    st.subheader("📌 분석 결과 요약")
    st.metric("예측된 학습자 유형", st.session_state['cluster_name'])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🎯 나의 역량 점수")
        st.write(f"- **Analytical**: {Analytical:.2f}/5.00")
        st.write(f"- **Collaborative**: {Collaborative:.2f}/5.00")
        st.write(f"- **Self-Directed**: {SelfDirected:.2f}/5.00")
        st.write(f"- **Questioning**: {Questioning:.2f}/5.00")

    with col2:
        st.write("### 🤝 추천 하브루타 파트너 유형")
        st.info(st.session_state['partner_recommendation'])

    st.divider()


# ---------------------------------------------
# 학습 메이트 추천
# ---------------------------------------------
if 'cluster' in st.session_state:

    df_students = st.session_state['students_processed']

    st.subheader("🧑‍🤝‍🧑 학습 메이트 추천받기")

    col1, col2 = st.columns(2)

    # 버튼: 추천 모드 저장
    with col1:
        if st.button("💡 나의 단점을 보완해줄 학습 메이트"):
            st.session_state['show_recommendation_mode'] = "complement"

    with col2:
        if st.button("🤝 나와 비슷한 학습 메이트"):
            st.session_state['show_recommendation_mode'] = "similar"

    # 추천 결과 출력
    mode = st.session_state.get('show_recommendation_mode', None)
    if mode:

        cluster = st.session_state['cluster']

        if mode == "complement":
            target_cluster = complement_map[cluster]
            candidates = df_students[df_students['Cluster'] == target_cluster]
            result = candidates.head(3)[['ID', 'grade']]
            st.subheader("🎯 추천 학습 메이트 (보완형)")

        elif mode == "similar":
            candidates = df_students[df_students['Cluster'] == cluster]
            result = candidates.head(3)[['ID', 'grade']]
            st.subheader("🎯 추천 학습 메이트 (유사형)")

        st.dataframe(result.reset_index(drop=True))
