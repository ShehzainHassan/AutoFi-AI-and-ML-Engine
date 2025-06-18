import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BACKEND_API_URL = os.getenv("BACKEND_API_URL")

class RecommendationService:
    async def get_recommendations(self, user_id):
        return ["Car A", "Car B", "Car C"]

    async def get_similar_vehicles(self, vehicle_id):
        return ["Car X", "Car Y", "Car Z"]

    def load_interactions_summary(self):
        url = f"{BACKEND_API_URL}/user/get-interactions-summary"
        response = requests.get(url)
        response.raise_for_status()

        interactions = response.json()
        df = pd.DataFrame(interactions)
        return df
    
    def load_vehicle_features(self):
        url = f"{BACKEND_API_URL}/Vehicle?pageView=1000&offset=0"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        vehicles_list = data.get("vehicles", [])

        df = pd.DataFrame(vehicles_list)

        desired_columns = [
            "id",
            "vin",
            "make",
            "model",
            "year",
            "price",
            "mileage",
            "color",
            "fuelType",
            "transmission",
            "status"
        ]

        df_filtered = df[desired_columns]

        return df_filtered

    def prepare_data(self):
            interactions_df = self.load_interactions_summary()
            vehicle_features_df = self.load_vehicle_features()
            return interactions_df, vehicle_features_df