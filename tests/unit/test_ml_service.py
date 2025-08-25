import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

from app.services.ml_service import MLModelService
from config.ml_config import MLConfig

@pytest.fixture
def mock_rec_service():
    service = MagicMock()
    service.load_interactions_summary = AsyncMock(return_value=pd.DataFrame({
        "user_id": [1, 1, 2],
        "vehicle_id": [101, 102, 101],
        "interaction_type": ["view", "click", "view"],
        "count": [5, 3, 2]
    }))
    service.load_vehicle_features = AsyncMock(return_value=pd.DataFrame({
        "Id": [101, 102],
        "Make": ["Toyota", "Honda"],
        "Model": ["Corolla", "Civic"],
        "Horsepower": [130, 158],
        "TorqueFtLbs": [128, 138],
        "EngineSize": ["1.8L", "2.0L"],
        "ZeroTo60MPH": [10.0, 8.5],
        "DrivetrainType": ["FWD", "FWD"],
        "CO2Emissions": [120, 140],
        "Transmission": ["Auto", "Manual"],
        "Price": [20000, 22000],
        "Year": [2020, 2021],
        "Color": ["Red", "Blue"],
        "FuelType": ["Gasoline", "Gasoline"],
        "CityMPG": [30, 32],
        "Mileage": [15000, 12000],
        "Status": ["Available", "Sold"]
    }))
    return service


@pytest.fixture
def ml_service(mock_rec_service):
    return MLModelService(
        user_repo=mock_rec_service,
        vehicle_repo=mock_rec_service,
        model_serving=MagicMock(),
        config=MLConfig()
    )


@pytest.mark.asyncio
async def test_prepare_data(ml_service):
    interactions_df, features_df = await ml_service.prepare_data()
    assert not interactions_df.empty
    assert "Id" in features_df.columns
    assert features_df.shape[0] == 2
    assert features_df["Horsepower"].dtype == np.float64


@pytest.mark.asyncio
@patch("app.models.model_persistance.joblib.dump")
async def test_train_vehicle_similarity_model(mock_dump, ml_service):
    await ml_service.train_vehicle_similarity_model()
    sim = ml_service.vehicle_similarity_topk
    assert isinstance(sim, dict)
    assert all(isinstance(v, list) for v in sim.values())
    mock_dump.assert_called_once()


@pytest.mark.asyncio
@patch("app.models.model_persistance.joblib.dump")
async def test_train_user_similarity_model(mock_dump, ml_service):
    await ml_service.train_user_similarity_model()
    sim = ml_service.user_similarity_topk
    assert isinstance(sim, dict)
    assert all(isinstance(v, list) for v in sim.values())
    mock_dump.assert_called_once()


@pytest.mark.asyncio
@patch("app.models.model_persistance.joblib.dump")
async def test_train_collaborative_model(mock_dump, ml_service):
    await ml_service.train_collaborative_model()
    model = ml_service.collaborative_model
    assert model is not None
    assert "svd" in model
    assert model["user_features"].shape[0] == 2
    assert model["vehicle_features"].shape[0] == 2
    mock_dump.assert_called_once()
