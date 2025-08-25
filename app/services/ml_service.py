from __future__ import annotations
from typing import List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.ml_config import MLConfig
from app.models.model_persistance import (
    save_collaborative_model,
    save_content_model,
    save_user_content_model,
)
from app.repositories.user_repository import UserRepository
from app.repositories.vehicle_repository import VehicleRepository
from app.services.model_serving_service import ModelServingService

class MLModelService():
    """
    Handles feature prep, model training, inference, and model persistence.
    """
    def __init__(self, user_repo: UserRepository, vehicle_repo: VehicleRepository,
                 model_serving: ModelServingService, config: MLConfig = MLConfig()):
        self.user_repo = user_repo
        self.vehicle_repo = vehicle_repo
        self.model_serving = model_serving
        self.config = config
        self.vehicle_similarity_topk: Dict[int, List[Tuple[int, float]]] | None = None
        self.user_similarity_topk: Dict[int, List[Tuple[int, float]]] | None = None
        self.collaborative_model: Dict[str, object] | None = None
        self.models_loaded: bool = False

    async def prepare_data(self):
        """
        Returns:
          interactions_df with columns [user_id, vehicle_id, weighted_count]
          features_df numeric-encoded/scaled and includes 'Id'
        """
        interactions_raw = await self.user_repo.load_interactions_summary()
        vehicle_df = await self.vehicle_repo.load_vehicle_features()
        features_df = vehicle_df.copy()

        if "Id" not in features_df.columns:
            features_df = features_df.reset_index()

        weights = self.config.interaction_weights
        if not interactions_raw.empty:
            interactions_raw["weight"] = interactions_raw["interaction_type"].map(weights).fillna(0.0)
            interactions_raw["weighted"] = interactions_raw["count"].astype(float) * interactions_raw["weight"].astype(float)
            interactions_df = (
                interactions_raw.groupby(["user_id", "vehicle_id"], as_index=False)["weighted"].sum()
                .rename(columns={"weighted": "weighted_count"})
            )
        else:
            interactions_df = pd.DataFrame(columns=["user_id", "vehicle_id", "weighted_count"])

        cat_columns = ["Make", "Model", "Color", "FuelType", "Transmission", "Status", "DrivetrainType"]
        for col in cat_columns:
            if col not in features_df.columns:
                features_df[col] = ""
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col].astype(str))

        if "EngineSize" in features_df.columns:
            features_df["EngineSize"] = features_df["EngineSize"].astype(str).str.replace("L", "", regex=False)
            features_df["EngineSize"] = pd.to_numeric(features_df["EngineSize"], errors="coerce")

        num_columns = ["Year", "Price", "Mileage", "CO2Emissions", "CityMPG",
                       "Horsepower", "TorqueFtLbs", "EngineSize", "ZeroTo60MPH"]

        for col in num_columns:
            if col not in features_df.columns:
                features_df[col] = np.nan
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
            features_df[col] = features_df[col].fillna(features_df[col].mean())
            scaler = StandardScaler()
            features_df[col] = scaler.fit_transform(features_df[[col]])

        if "Id" not in features_df.columns:
            raise ValueError('Expected "Id" column in Vehicles table.')

        return interactions_df, features_df

    def _train_content_based_topk(self, features_df: pd.DataFrame, weights: Dict[str, float], top_k: int) -> Dict[int, List[Tuple[int, float]]]:
        vehicle_ids = features_df["Id"].values
        drop_cols = [c for c in ["Id", "Vin"] if c in features_df.columns]
        feature_matrix = features_df.drop(columns=drop_cols).copy()

        for col, w in (weights or {}).items():
            if col in feature_matrix.columns:
                feature_matrix[col] = feature_matrix[col] * float(w)

        feature_np = feature_matrix.values
        sim = cosine_similarity(feature_np).astype(np.float32)

        top_k_similar: Dict[int, List[Tuple[int, float]]] = {}
        for i, v_id in enumerate(vehicle_ids):
            sim_scores = sim[i]
            top_idx = np.argsort(sim_scores)[::-1]
            top_idx = top_idx[top_idx != i][:top_k]
            top_ids = vehicle_ids[top_idx]
            top_vals = sim_scores[top_idx]
            top_k_similar[int(v_id)] = [(int(oid), float(s)) for oid, s in zip(top_ids, top_vals)]
        return top_k_similar

    async def train_vehicle_similarity_model(self) -> None:
        _, features_df = await self.prepare_data()
        self.vehicle_similarity_topk = self._train_content_based_topk(features_df, self.config.vehicle_feature_weights, top_k=self.config.top_k_similar)
        save_content_model(self.vehicle_similarity_topk)
        self.models_loaded = True

    async def train_user_similarity_model(self) -> None:
        _, features_df = await self.prepare_data()
        self.user_similarity_topk = self._train_content_based_topk(features_df, self.config.user_feature_weights, top_k=self.config.top_k_similar)
        save_user_content_model(self.user_similarity_topk)
        self.models_loaded = True

    async def train_collaborative_model(self) -> None:
        interactions_df, _ = await self.prepare_data()

        if interactions_df.empty:
            self.collaborative_model = {
                "svd": None,
                "user_features": np.zeros((0, 0), dtype=np.float32),
                "vehicle_features": np.zeros((0, 0), dtype=np.float32),
                "interaction_matrix": pd.DataFrame(),
            }
            return

        interaction_matrix: pd.DataFrame = interactions_df.pivot_table(index="user_id", columns="vehicle_id", values="weighted_count", fill_value=0.0)
        sparse = csr_matrix(interaction_matrix.values)

        n_users, n_items = sparse.shape
        max_components = max(1, min(n_users, n_items) - 1)
        n_components = min(self.config.svd_components, max_components)

        svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_state, n_iter=self.config.max_iter)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            user_features = svd.fit_transform(sparse)

        vehicle_features = svd.components_.T

        self.collaborative_model = {
            "svd": svd,
            "user_features": user_features,
            "vehicle_features": vehicle_features,
            "interaction_matrix": interaction_matrix,
        }
        save_collaborative_model(self.collaborative_model)
        self.models_loaded = True
