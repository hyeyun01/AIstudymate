import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

def match_partners(df_students):
    """
    학생 데이터를 받아 역량 계산 후, KMeans 군집과 거리 기반 파트너 추천
    """
    # 역량 계산 (CSV 컬럼 순서 기준)
    Analytical_idx = [0, 2, 8, 14, 22]
    Collaborative_idx = [3, 4, 10, 11, 18, 19, 25]
    SelfDirected_idx = [1, 6, 7, 15, 16, 26]
    Questioning_idx = [5, 12, 13, 20, 21, 27, 28]

    df_students['Analytical'] = df_students.iloc[:, Analytical_idx].mean(axis=1)
    df_students['Collaborative'] = df_students.iloc[:, Collaborative_idx].mean(axis=1)
    df_students['SelfDirected'] = df_students.iloc[:, SelfDirected_idx].mean(axis=1)
    df_students['Questioning'] = df_students.iloc[:, Questioning_idx].mean(axis=1)

    features = ['Analytical','Collaborative','SelfDirected','Questioning']

    # 스케일링 + KMeans 예측
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    X_scaled = scaler.transform(df_students[features])
    df_students['Cluster'] = kmeans.predict(X_scaled)

    # 유클리드 거리 기반 추천 (자신과 가장 가까운 학생 제외)
    distances = euclidean_distances(X_scaled)
    partners = []
    for i in range(len(df_students)):
        dist_row = distances[i].copy()
        dist_row[i] = np.inf
        partner_idx = np.argmin(dist_row)
        partners.append(df_students.iloc[partner_idx]['ID'])
    df_students['Recommended_Partner'] = partners

    # 군집 이름 매핑
    cluster_name_map = {
        0: "병아리 탐험가 🐣",
        1: "논리왕 🤓",
        2: "친구왕 🦄",
        3: "문제 해결 마스터 🕵️‍♂️"
    }
    df_students['Cluster_Name'] = df_students['Cluster'].map(cluster_name_map)

    return df_students[['ID','Cluster','Cluster_Name','Recommended_Partner']]
