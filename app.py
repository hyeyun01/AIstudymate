import streamlit as st
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------
# 1) 페이지 기본 설정
# ---------------------------------------------
st.set_page_config(page_title="AI StudyMate - 학습 성향 분석 데모", layout="wide")
st.title("🧠 AI StudyMate - 학습 성향 진단 데모")
st.write("30문항 설문을 기반으로 학습 성향을 분석하고, 맞춤형 하브루타 파트너 유형을 추천합니다.")
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

st.subheader("📘 학습 성향 설문 (30문항)")

CHOICES = ["① 전혀 아니다", "② 아니다", "③ 보통이다", "④ 그렇다", "⑤ 매우 그렇다"]
responses = {}

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
    # 4) K-means 군집 모델 (데모용)
    # ---------------------------------------------
    kmeans = KMeans(n_clusters=4, random_state=42)
    sample_data = np.random.rand(200, 4) * 5
    kmeans.fit(sample_data)
    cluster = kmeans.predict(profile_vector)[0]

    # ---------------------------------------------
    # 5) 군집명 매핑 (재밌는 이름 + 이모지)
    # ---------------------------------------------
    cluster_name_map = {
        0: "🐣 병아리 탐험가",
        1: "🤓 논리왕",
        2: "🦄 친구왕",
        3: "🕵️‍♂️ 문제 해결 마스터"
    }
    cluster_name = cluster_name_map.get(cluster, "Unknown")

    # ---------------------------------------------
    # 6) 하브루타 파트너 유형 추천
    # ---------------------------------------------
    partner_recommendation_map = {
        0: "🐣 '병아리 탐험가' 유형은 학습 루틴이 안정적인 '논리왕' 친구와 함께하면 기본기와 탐구력이 균형 있게 성장합니다.",
        1: "🤓 '논리왕' 유형은 창의적이고 협력적인 '친구왕' 친구와 페어를 이루면 사고력과 소통력이 극대화됩니다.",
        2: "🦄 '친구왕' 유형은 분석적이고 문제 해결 능력이 강한 '문제 해결 마스터' 친구와 함께하면 학습 효과가 상승합니다.",
        3: "🕵️‍♂️ '문제 해결 마스터' 유형은 협력적이고 유연한 '친구왕' 친구와 활동하면 설명력과 팀워크가 보완됩니다."

    }
    partner_recommendation = partner_recommendation_map[cluster]

    # ---------------------------------------------
    # 7) 결과 출력
    # ---------------------------------------------
    st.subheader("📌 분석 결과 요약")
    st.markdown(f"**예측된 학습자 유형:** <span style='color:#FF5733'>{cluster_name}</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🎯 나의 역량 점수")
        st.write(f"- **Analytical(분석성)**: {Analytical:.2f}")
        st.write(f"- **Collaborative(협력성)**: {Collaborative:.2f}")
        st.write(f"- **Self-Directed(자기주도)**: {SelfDirected:.2f}")
        st.write(f"- **Questioning(탐구·질문성)**: {Questioning:.2f}")

    with col2:
        st.write("### 🤝 추천 하브루타 파트너 유형")
        st.info(partner_recommendation)

    st.divider()

    # ---------------------------------------------
    # 8) 역량 카드 스타일 출력
    # ---------------------------------------------
    st.subheader("📇 나의 Strength Profile 카드")
    st.success(f"""
<b>{cluster_name}</b>

- 분석적 사고 수준: **{Analytical:.1f} / 5**
- 협력 성향: **{Collaborative:.1f} / 5**
- 자기주도: **{SelfDirected:.1f} / 5**
- 탐구·질문 수준: **{Questioning:.1f} / 5**

📌 *AI StudyMate는 이 프로필을 기반으로  
최적의 하브루타 파트너와 학습 그룹을 추천합니다.*
""", unsafe_allow_html=True)
