from app.db import get_db_connection
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import warnings

from app.models import recommendation_model

class RecommendationService:
    def __init__(self, vehicle_limit=10000):
        self.conn = get_db_connection()
        self.vehicle_limit = vehicle_limit
        self.similarity_matrix = None
        self.vehicle_ids = None

    def close(self):
        if self.conn:
            self.conn.close()

    def load_interactions_summary(self):
        query = """
            SELECT 
                "UserId" AS user_id,
                "VehicleId" AS vehicle_id,
                "InteractionType" AS interaction_type,
                COUNT(*) AS count
            FROM "UserInteractions"
            GROUP BY "UserId", "VehicleId", "InteractionType"
            ORDER BY user_id, vehicle_id;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        return df

    def load_vehicle_features(self):
        query = f"""
            SELECT * FROM "Vehicles"
            ORDER BY "Id"
            LIMIT {self.vehicle_limit};
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        return df

    def prepare_data(self):
        interactions_df = self.load_interactions_summary()
        vehicle_features_df = self.load_vehicle_features()
        return interactions_df, vehicle_features_df

    def prepare_data_for_ml(self):
        interactions_df, vehicle_features_df = self.prepare_data()
        features_df = vehicle_features_df.copy()

        # Label encode categorical columns
        categorical_columns = ['Make', 'Model', 'Color', 'FuelType', 'Transmission', 'Status']
        for col in categorical_columns:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[col] = features_df[col].astype(str)
                features_df[col] = le.fit_transform(features_df[col])

        # Scale numerical columns
        numerical_columns = ['Year', 'Price', 'Mileage']
        for col in numerical_columns:
            if col in features_df.columns:
                scaler = StandardScaler()
                features_df[col] = scaler.fit_transform(features_df[[col]])

        return interactions_df, features_df

    def train_content_based_model(self, features_df):
        self.vehicle_ids = features_df["Id"].values

        columns_to_drop = ["Id", "Vin"]
        feature_matrix = features_df.drop(columns=columns_to_drop).values

        self.similarity_matrix = cosine_similarity(feature_matrix)

    def get_similar_vehicles(self, vehicle_id, n=5):
        if self.similarity_matrix is None or self.vehicle_ids is None:
            _, features_df = self.prepare_data_for_ml()
            self.train_content_based_model(features_df)

        idx = np.where(self.vehicle_ids == vehicle_id)[0]
        if idx.size == 0:
            return []
        idx = idx.item()

        sim_scores = self.similarity_matrix[idx]
        similar_indices = np.argsort(sim_scores)[::-1]
        similar_indices = similar_indices[similar_indices != idx]
        top_n_indices = similar_indices[:n]

        similar_vehicle_ids = self.vehicle_ids[top_n_indices]
        return similar_vehicle_ids.tolist()

    def train_collaborative_model(self, interactions_df):
        interaction_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='vehicle_id',
            values='count', 
            fill_value=0
        )
        sparse_matrix = csr_matrix(interaction_matrix.values)

        n_features = sparse_matrix.shape[1]
        n_components = min(50, n_features - 1)
        svd = TruncatedSVD(n_components=n_components)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            user_features = svd.fit_transform(sparse_matrix)

        vehicle_features = svd.components_.T

        return svd, user_features, vehicle_features, interaction_matrix

    def get_hybrid_recommendations(self, user_id, models, top_n=10):
        content_model, collaborative_model, interaction_matrix = models

        query = """
            SELECT "VehicleId" AS vehicle_id, COUNT(*) AS weight
            FROM "UserInteractions"
            WHERE "UserId" = %s
            GROUP BY "VehicleId"
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (user_id,))
            rows = cur.fetchall()

        if not rows:
            print(f"No interactions for user_id={user_id}")
            return []

        user_interactions = pd.DataFrame(rows)

        content_scores = {}
        for _, row in user_interactions.iterrows():
            vehicle_id = row['vehicle_id']
            weight = row['weight']

            similar_ids = content_model.get_similar_vehicles(vehicle_id, n=top_n * 5)
            for rank, similar_id in enumerate(similar_ids):
                score = weight * (1.0 / (rank + 1))
                content_scores[similar_id] = content_scores.get(similar_id, 0.0) + score

        collaborative_scores = {}
        if user_id in interaction_matrix.index:
            user_index = interaction_matrix.index.get_loc(user_id)
            user_vector = collaborative_model[1][user_index]
            vehicle_vectors = collaborative_model[2]
            scores = np.dot(vehicle_vectors, user_vector)
            vehicle_ids = interaction_matrix.columns.values

            for i, v_id in enumerate(vehicle_ids):
                collaborative_scores[v_id] = scores[i]

        hybrid_scores = {}
        for v_id in set(content_scores.keys()).union(collaborative_scores.keys()):
            content_score = content_scores.get(v_id, 0.0)
            collaborative_score = collaborative_scores.get(v_id, 0.0)

            alpha = 0.5
            hybrid_score = alpha * content_score + (1 - alpha) * collaborative_score
            hybrid_scores[v_id] = hybrid_score

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [int(v_id) for v_id, _ in ranked[:top_n]]

        return recommendations

    def train_models(self):
        interactions_df, features_df = self.prepare_data_for_ml()
        self.train_content_based_model(features_df)
        collaborative_model = self.train_collaborative_model(interactions_df)
        return collaborative_model

    def save_models(self, collaborative_model):
        recommendation_model.save_collaborative_model(collaborative_model)
        recommendation_model.save_content_model(self.similarity_matrix, self.vehicle_ids)

    def load_models(self):
        collaborative_model = recommendation_model.load_collaborative_model()
        self.similarity_matrix, self.vehicle_ids = recommendation_model.load_content_model()
        return collaborative_model
