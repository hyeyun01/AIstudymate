import streamlit as st
import numpy as np
import pandas as pd
import joblib
from partner_matching import match_partners

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

st.subheader("📘 학습 성향 설문 (30문항)")
responses = {}
for i, question in enumerate(questions, start=1):
    choice = st.radio(
        f"**Q{i}. {question}**",
        CHOICES,
        key=f"q_{i}",
        horizontal=True
    )
    responses[f"Q{i}"] = CHOICES.index(choice) + 1
    st.markdown("---")

# ---------------------------------------------
# 3) 학습 성향 분석
# ---------------------------------------------
if st.button("🧪 학습 성향 분석 시작"):
    responses_list = [CHOICES.index(st.session_state[f"q_{i}"]) + 1 for i in range(1,31)]

    responses_array = np.array(responses_list)
    Analytical_idx = [0, 2, 8, 14, 22]
    Collaborative_idx = [3, 4, 10, 11, 18, 19, 25]
    SelfDirected_idx = [1, 6, 7, 15, 16, 26]
    Questioning_idx = [5, 12, 13, 20, 21, 27, 28]

    Analytical = responses_array[Analytical_idx].mean()
    Collaborative = responses_array[Collaborative_idx].mean()
    SelfDirected = responses_array[SelfDirected_idx].mean()
    Questioning = responses_array[Questioning_idx].mean()

    profile_vector = np.array([Analytical, Collaborative, SelfDirected, Questioning]).reshape(1,-1)

    # ---------------------------------------------
    # 4) 스케일러 + KMeans 불러오기
    # ---------------------------------------------
    try:
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans_model.pkl")
    except Exception as e:
        st.error(f"모델 불러오기 실패: {e}")
        st.stop()

    # feature 이름 맞추기
    df_profile = pd.DataFrame({
        'competency_label_1':[Analytical],
        'competency_label_2':[SelfDirected],
        'competency_label_3':[Collaborative],
        'competency_label_4':[Questioning]
    })
    X_scaled = scaler.transform(df_profile)
    cluster = int(kmeans.predict(X_scaled)[0])

    # ---------------------------------------------
    # 5) Strength Profile & 파트너 추천 정보
    # ---------------------------------------------
    cluster_name_map = {
        0: "병아리 탐험가 🐣",
        1: "논리왕 🤓",
        2: "친구왕 🦄",
        3: "문제 해결 마스터 🕵️‍♂️"
    }
    partner_recommendation_map = {
        0: "문제 해결 마스터 🕵️‍♂️ 친구와 함께하면 기본기 형성이 빠릅니다.",
        1: "친구왕 🦄 친구와 함께하면 협력 학습과 이해가 향상됩니다.",
        2: "논리왕 🤓 친구와 함께하면 사고력과 계획력이 강화됩니다.",
        3: "병아리 탐험가 🐣 친구와 함께 활동하면 개념 이해와 학습 루틴 형성에 도움됩니다."
    }

    # Strength Profile 예시
    strength_profile_map = {
        0: {"학습 스타일 분석":["기초 개념 이해와 반복 학습을 잘함"],"이렇게 공부하면 좋아요":["노트 정리"],"친구와 함께 공부할 때 역할":["탐험가 역할"]},
        1: {"학습 스타일 분석":["논리적 분석과 단계적 문제 해결"],"이렇게 공부하면 좋아요":["계획표 작성 후 풀이"],"친구와 함께 공부할 때 역할":["분석가 역할"]},
        2: {"학습 스타일 분석":["친구와 함께 토론 및 이해"],"이렇게 공부하면 좋아요":["그룹 토론"],"친구와 함께 공부할 때 역할":["설명가 역할"]},
        3: {"학습 스타일 분석":["문제 탐구와 응용 활동"],"이렇게 공부하면 좋아요":["문제 변형 풀이"],"친구와 함께 공부할 때 역할":["문제 해결사 역할"]},
    }

    # session_state 저장
    st.session_state['Analytical'] = Analytical
    st.session_state['Collaborative'] = Collaborative
    st.session_state['SelfDirected'] = SelfDirected
    st.session_state['Questioning'] = Questioning
    st.session_state['cluster'] = cluster
    st.session_state['cluster_name'] = cluster_name_map[cluster]
    st.session_state['partner_recommendation'] = partner_recommendation_map[cluster]
    st.session_state['strength_profile'] = strength_profile_map[cluster]

    # 학생 데이터 불러오기 + match_partners
    try:
        df_students_raw = pd.read_csv("real_students.csv")
        df_students_processed = match_partners(df_students_raw)
        st.session_state['students_processed'] = df_students_processed
    except Exception as e:
        st.warning(f"학생 데이터 처리 오류: {e}")
        st.session_state['students_processed'] = pd.DataFrame(columns=['ID','grade','Cluster'])

# ---------------------------------------------
# 6) 결과 출력 + 학습 메이트 추천 통합
# ---------------------------------------------
if 'cluster' in st.session_state:
    Analytical = st.session_state['Analytical']
    Collaborative = st.session_state['Collaborative']
    SelfDirected = st.session_state['SelfDirected']
    Questioning = st.session_state['Questioning']
    cluster = st.session_state['cluster']

    # 역량 카드
    st.subheader("📌 분석 결과 요약")
    st.metric("예측된 학습자 유형", st.session_state['cluster_name'])
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🎯 나의 역량 점수")
        st.write(f"- Analytical: {Analytical:.2f}/5.0")
        st.write(f"- Collaborative: {Collaborative:.2f}/5.0")
        st.write(f"- SelfDirected: {SelfDirected:.2f}/5.0")
        st.write(f"- Questioning: {Questioning:.2f}/5.0")
    with col2:
        st.write("### 🤝 추천 하브루타 파트너 유형")
        st.info(st.session_state['partner_recommendation'])

    # Strength Profile 카드
    st.divider()
    st.subheader("📇 나의 Strength Profile 카드")
    for title, points in st.session_state['strength_profile'].items():
        points_html = "".join([f"<p style='margin:5px 0;'>- {p}</p>" for p in points])
        st.markdown(
            f"""
            <div style="background-color:#f0f4f8;padding:18px;border-radius:12px;margin-bottom:12px;box-shadow:2px 2px 8px rgba(0,0,0,0.1);">
                <h4 style="color:#1f4e79;">{title}</h4>
                {points_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # 학습 메이트 추천
    complement_map = {0:3,1:2,2:1,3:0}
    df_students = st.session_state.get('students_processed', pd.DataFrame(columns=['ID','grade','Cluster']))

    st.divider()
    st.subheader("🧑‍🤝‍🧑 학습 메이트 추천")
    cluster_user = int(cluster)
    target_cluster = complement_map[cluster_user]

    recommended_complement = df_students[df_students['Cluster']==target_cluster][['ID','grade']].head(3)
    st.subheader("🎯 추천 학습 메이트 (보완형)")
    st.dataframe(recommended_complement.reset_index(drop=True))

    recommended_similar = df_students[df_students['Cluster']==cluster_user][['ID','grade']].head(3)
    st.subheader("🎯 추천 학습 메이트 (유사형)")
    st.dataframe(recommended_similar.reset_index(drop=True))
