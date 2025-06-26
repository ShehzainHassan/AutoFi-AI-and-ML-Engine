from app.db import get_db_connection
import pandas as pd
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import warnings
import json
from app.schemas import schemas
from app.models import recommendation_model
from joblib import Memory

memory = Memory(location=".cache", verbose=0)

class RecommendationService:
    feature_weights_vehicle = {
        "Horsepower": 5.0, "TorqueFtLbs": 5.0, "EngineSize": 4.5, "ZeroTo60MPH": 4.5,
        "DrivetrainType": 4.0, "CO2Emissions": 3.5, "Transmission": 3.5, "Price": 3.0,
        "Model": 2.5, "Make": 2.0, "Year": 1.5, "Color": 1.5, "FuelType": 1.0, 
        "CityMPG": 0.8, "Mileage": 0.5, "Status": 0.5
    }

    feature_weights_user = {
        "Price": 5.0, "FuelType": 4.5, "DrivetrainType": 4.25, "CO2Emissions": 4.0, "CityMPG": 4.0,
        "Horsepower": 3.5, "TorqueFtLbs": 3.5, "EngineSize": 3.0, "Color": 2.5,
        "Transmission": 2.5, "Mileage": 2.0, "ZeroTo60MPH": 1.5, "Status": 1.5, "Model": 1.0,
        "Make": 1.0, "Year": 0.8
    }

    interaction_weights = {
        "favorite-added": 5.0,
        "contacted-seller": 4.0,
        "share": 3.0,
        "view": 1.0,
    }

    def __init__(self, vehicle_limit=20000):
        self.conn = get_db_connection()
        self.vehicle_limit = vehicle_limit

        self.interactions_df = None
        self.vehicle_df = None

        self.collaborative_model = None
        self.vehicle_similarity_topk = None
        self.user_similarity_topk = None

        self.load_models()

    def close(self):
        if self.conn:
            self.conn.close()

    def load_interactions_summary(self):
        if self.interactions_df is not None:
            return self.interactions_df

        query = """
            SELECT "UserId" AS user_id, "VehicleId" AS vehicle_id,
                   "InteractionType" AS interaction_type, COUNT(*) AS count
            FROM "UserInteractions"
            GROUP BY "UserId", "VehicleId", "InteractionType"
            ORDER BY user_id, vehicle_id;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        df = df[df['interaction_type'].isin(self.interaction_weights.keys())]

        df['weighted_count'] = df.apply(
            lambda row: row['count'] * self.interaction_weights[row['interaction_type']], axis=1
        )

        self.interactions_df = df
        return self.interactions_df

    @memory.cache
    def load_vehicle_features_cached(vehicle_limit):
        conn = get_db_connection()
        query = f"""
            SELECT * FROM "Vehicles"
            ORDER BY "Id"
            LIMIT {vehicle_limit};
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)

        try:
            with open('app/data/car-features.json', 'r') as f:
                car_features_list = json.load(f)
        except Exception as e:
            print(f"Error loading car-features.json: {e}")
            car_features_list = []

        feature_lookup = {
            (item['make'], item['model'], item['year']): item for item in car_features_list
        }

        df['CO2Emissions'] = None
        df['CityMPG'] = None
        df['Horsepower'] = None
        df['TorqueFtLbs'] = None
        df['EngineSize'] = None
        df['ZeroTo60MPH'] = None
        df['DrivetrainType'] = None

        for i, row in df.iterrows():
            key = (row.get('Make'), row.get('Model'), row.get('Year'))
            feature_data = feature_lookup.get(key)

            if feature_data:
                fuel_economy = feature_data.get('features', {}).get('fuelEconomy', {})
                engine = feature_data.get('features', {}).get('engine', {})
                performance = feature_data.get('features', {}).get('performance', {})
                drivetrain = feature_data.get('features', {}).get('drivetrain', {})
                options = feature_data.get('features', {}).get('options', [])

                df.at[i, 'CO2Emissions'] = fuel_economy.get('CO2Emissions')
                df.at[i, 'CityMPG'] = fuel_economy.get('cityMPG')
                df.at[i, 'Horsepower'] = engine.get('horsepower')
                df.at[i, 'TorqueFtLbs'] = engine.get('torqueFtLBS')
                df.at[i, 'EngineSize'] = engine.get('size')
                df.at[i, 'ZeroTo60MPH'] = performance.get('ZeroTo60MPH')
                df.at[i, 'DrivetrainType'] = drivetrain.get('type')

        conn.close()
        return df
    
    def load_vehicle_features(self):
        if self.vehicle_df is None:
            self.vehicle_df = self.load_vehicle_features_cached(self.vehicle_limit)
        return self.vehicle_df
    
    def prepare_data(self):
        interactions_df = self.load_interactions_summary()
        vehicle_df = self.load_vehicle_features()
        features_df = vehicle_df.copy()

        cat_columns = ['Make', 'Model', 'Color', 'FuelType', 'Transmission', 'Status', 'DrivetrainType']
        for col in cat_columns:
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col].astype(str))

        if 'EngineSize' in features_df.columns:
            features_df['EngineSize'] = features_df['EngineSize'].astype(str).str.replace('L', '', regex=False)
            features_df['EngineSize'] = pd.to_numeric(features_df['EngineSize'], errors='coerce')

        num_columns = ['Year', 'Price', 'Mileage', 'CO2Emissions', 'CityMPG', 'Horsepower', 'TorqueFtLbs', 'EngineSize', 'ZeroTo60MPH']
        for col in num_columns:
            scaler = StandardScaler()
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df[col] = features_df[col].fillna(features_df[col].mean())
            features_df[col] = scaler.fit_transform(features_df[[col]])

        return interactions_df, features_df

    def train_content_based_model(self, features_df, weights, top_k=200):
        vehicle_ids = features_df["Id"].values
        feature_matrix = features_df.drop(columns=["Id", "Vin"])

        for col, weight in weights.items():
            if col in feature_matrix.columns:
                feature_matrix[col] *= weight

        feature_np = feature_matrix.values

        similarity_matrix = cosine_similarity(feature_np).astype(np.float32)

        top_k_similar = {}

        for i, v_id in enumerate(vehicle_ids):
            sim_scores = similarity_matrix[i]               

            top_indices = np.argsort(sim_scores)[::-1]
            top_indices = top_indices[top_indices != i][:top_k]

            top_vehicle_ids = vehicle_ids[top_indices]
            top_scores = sim_scores[top_indices]

            top_k_similar[int(v_id)] = [
                (int(other_id), float(score))
                for other_id, score in zip(top_vehicle_ids, top_scores)
            ]

        return top_k_similar

    def train_vehicle_similarity_model(self):
        _, features_df = self.prepare_data()
        self.vehicle_similarity_topk = self.train_content_based_model(features_df, self.feature_weights_vehicle, top_k=200)

    def train_user_similarity_model(self):
        _, features_df = self.prepare_data()
        self.user_similarity_topk = self.train_content_based_model(features_df, self.feature_weights_user, top_k=200)

    def train_collaborative_model(self):
        interactions_df, _ = self.prepare_data()
        interaction_matrix = interactions_df.pivot_table(
            index='user_id', columns='vehicle_id', values='weighted_count', fill_value=0
        )
        sparse_matrix = csr_matrix(interaction_matrix.values)

        n_features = sparse_matrix.shape[1]
        n_components = min(50, n_features - 1)
        svd = TruncatedSVD(n_components=n_components)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            user_features = svd.fit_transform(sparse_matrix)

        vehicle_features = svd.components_.T
        self.collaborative_model = (svd, user_features, vehicle_features, interaction_matrix)

    def train_all_models(self):
        self.train_collaborative_model()
        self.train_vehicle_similarity_model()
        self.train_user_similarity_model()

    def get_similar_vehicles(self, vehicle_id, top_n=5):
        if self.vehicle_similarity_topk is None:
            self.train_vehicle_similarity_model()

        top_k_similar = self.vehicle_similarity_topk.get(int(vehicle_id), [])
        selected = top_k_similar[:top_n]
        vehicle_df = self.load_vehicle_features()
        similar_vehicles = []

        for sim_id, score in selected:
            row = vehicle_df[vehicle_df["Id"] == sim_id].iloc[0]
            features = self.extract_vehicle_features(row)
            similar_vehicles.append(schemas.SimilarVehicle(
                vehicle_id=sim_id,
                similarity_score=score,
                features=features
            ))

        return similar_vehicles

    def get_hybrid_recommendations(self, user_id, top_n=10):
        if self.collaborative_model is None:
            self.train_collaborative_model()
        if self.user_similarity_topk is None:
            self.train_user_similarity_model()

        _, user_features, vehicle_features, interaction_matrix = self.collaborative_model

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
            return schemas.RecommendationResponse(recommendations=[], model_type="hybrid")

        user_interactions = pd.DataFrame(rows)

        content_scores = {}
        for _, row in user_interactions.iterrows():
            vehicle_id = row['vehicle_id']
            weight = row['weight']

            similar_vehicles = self.get_similar_vehicles_with_matrix(
                vehicle_id, self.user_similarity_topk, n=top_n * 5
            )

            for similar_vehicle in similar_vehicles:
                sim_id = similar_vehicle.vehicle_id
                sim_score = similar_vehicle.similarity_score 

                score = weight * sim_score
                content_scores[sim_id] = content_scores.get(sim_id, 0.0) + score

        collaborative_scores = {}
        if user_id in interaction_matrix.index:
            user_index = interaction_matrix.index.get_loc(user_id)
            user_vector = user_features[user_index]
            scores = np.dot(vehicle_features, user_vector)
            vehicle_ids = interaction_matrix.columns.values

            min_score = scores.min()
            max_score = scores.max()
            score_range = max_score - min_score if max_score != min_score else 1.0

            for i, v_id in enumerate(vehicle_ids):
                normalized_score = (scores[i] - min_score) / score_range
                collaborative_scores[v_id] = normalized_score

        hybrid_scores = {}
        for v_id in set(content_scores.keys()).union(collaborative_scores.keys()):
            content_score = content_scores.get(v_id, 0.0)
            collab_score = collaborative_scores.get(v_id, 0.0)

            hybrid_score = 0.5 * content_score + 0.5 * collab_score
            hybrid_scores[v_id] = hybrid_score

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top_vehicle_ids = [int(v_id) for v_id, _ in ranked[:top_n]]

        vehicle_df = self.load_vehicle_features()
        top_vehicles = vehicle_df[vehicle_df["Id"].isin(top_vehicle_ids)]

        recommendations = []
        for v_id, score in ranked[:top_n]:
            row = top_vehicles[top_vehicles["Id"] == int(v_id)].iloc[0]
            features = self.extract_vehicle_features(row)
            recommendations.append(schemas.VehicleRecommendation(
                vehicle_id=int(v_id),
                score=float(score),
                features=features
            ))

        return schemas.RecommendationResponse(
            recommendations=recommendations,
            model_type="hybrid"
        )

    def get_similar_vehicles_with_matrix(self, vehicle_id, similarity_topk, n=5):
        vehicle_df = self.load_vehicle_features()
        top_k_similar = similarity_topk.get(int(vehicle_id), [])
        selected = top_k_similar[:n]

        similar_vehicles = []
        for sim_id, score in selected:
            row = vehicle_df[vehicle_df["Id"] == sim_id].iloc[0]
            features = self.extract_vehicle_features(row)
            similar_vehicles.append(schemas.SimilarVehicle(
                vehicle_id=sim_id,
                similarity_score=score,
                features=features
            ))

        return similar_vehicles

    def save_models(self):
        print("[SAVE] Saving all trained models...")
        recommendation_model.save_collaborative_model(self.collaborative_model)
        recommendation_model.save_content_model(self.vehicle_similarity_topk)
        recommendation_model.save_user_content_model(self.user_similarity_topk)
        print("[SAVE] All models saved successfully.")

    def load_models(self):
        need_save = False

        print("[LOAD MODELS] Loading collaborative model...")
        self.collaborative_model = recommendation_model.load_collaborative_model()
        if self.collaborative_model is None:
            print("[TRAINING] Collaborative model not found, training new model...")
            self.train_collaborative_model()
            need_save = True

        print("[LOAD MODELS] Loading content-based model (vehicle)...")
        self.vehicle_similarity_topk = recommendation_model.load_content_model()
        if self.vehicle_similarity_topk is None:
            print("[TRAINING] Content-based model (vehicle) not found, training new model...")
            self.train_vehicle_similarity_model()
            need_save = True

        print("[LOAD MODELS] Loading content-based model (user)...")
        self.user_similarity_topk = recommendation_model.load_user_content_model()
        if self.user_similarity_topk is None:
            print("[TRAINING] Content-based model (user) not found, training new model...")
            self.train_user_similarity_model()
            need_save = True

        if need_save:
            print("[LOAD MODELS] Saving newly trained models...")
            self.save_models()

        print("[LOAD MODELS] All models ready.")

    def extract_vehicle_features(self, row):
        return {
            "Make": str(row.get("Make", "")),
            "Model": str(row.get("Model", "")),
            "Year": str(row.get("Year", "")),
            "Price": str(row.get("Price", "")),
            "Mileage": str(row.get("Mileage", "")),
            "Color": str(row.get("Color", "")),
            "FuelType": str(row.get("FuelType", "")),
            "Transmission": str(row.get("Transmission", "")),
            "Status": str(row.get("Status", "")),
            "CO2Emissions": str(row.get("CO2Emissions", "")),
            "CityMPG": str(row.get("CityMPG", "")),
            "Horsepower": str(row.get("Horsepower", "")),
            "TorqueFtLbs": str(row.get("TorqueFtLbs", "")),
            "EngineSize": str(row.get("EngineSize", "")),
            "ZeroTo60MPH": str(row.get("ZeroTo60MPH", "")),
            "DrivetrainType": str(row.get("DrivetrainType", "")),
        }
