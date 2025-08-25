import pandas as pd
import asyncpg
import json
import asyncio
from typing import List, Dict, Optional, Any
import pickle


class VehicleRepository:
    """
    Encapsulates all vehicle-related DB access and in-memory caching.
    """
    VEHICLE_CACHE_KEY = 'vehicle_features'
    def __init__(self, pool: asyncpg.Pool, vehicle_limit: int = 20000, redis = None):
        self.pool = pool
        self.vehicle_limit = vehicle_limit
        self.redis = redis
        self._vehicle_cache: Optional[pd.DataFrame] = None
        self._vehicle_lookup: Dict[int, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def _read_car_features_json(self) -> List[Dict[str, Any]]:
        def _read():
            with open("app/data/car-features.json", "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            return await asyncio.to_thread(_read)
        except Exception as e:
            print(f"[WARN] Error loading car-features.json: {e}")
            return []

    async def load_vehicle_features(self) -> pd.DataFrame:
        async with self._lock:
            if self._vehicle_cache is not None:
                return self._vehicle_cache

            # Try Redis first
            redis = getattr(self, "redis", None)
            if redis:
                data = await redis.get(self.VEHICLE_CACHE_KEY)
                if data:
                    try:
                        df = pickle.loads(data)
                        self._vehicle_cache = df
                        self._vehicle_lookup = {int(row["Id"]): dict(row) for _, row in df.iterrows()}
                        return df
                    except Exception as e:
                        print(f"[WARN] Failed to load vehicles from Redis: {e}")

            # Fallback: query DB 
            async with self.pool.acquire() as conn:
                print("[DEBUG] starting DB fetch")
                rows = await conn.fetch(
                    f'SELECT * FROM "Vehicles" ORDER BY "Id" LIMIT {self.vehicle_limit}'
                )
                print("[DEBUG] fetched rows: ", len(rows))

            print("[DEBUG] building DataFrame")
            df = pd.DataFrame([dict(r) for r in rows])
            print("[DEBUG] built DataFrame")


            # Enrich with car-features.json 
            car_features_list = await self._read_car_features_json()
            feature_lookup = {
                (item['make'], item['model'], item['year']): item for item in car_features_list
            }

            # Add missing columns
            for col in ["CO2Emissions", "CityMPG", "Horsepower", "TorqueFtLbs",
                        "EngineSize", "ZeroTo60MPH", "DrivetrainType"]:
                df[col] = None

            print("[DEBUG] enrichment start")
            for i, row in df.iterrows():
                key = (row.get("Make"), row.get("Model"), row.get("Year"))
                feature_data = feature_lookup.get(key)
                if feature_data:
                    fuel = feature_data.get("features", {}).get("fuelEconomy", {})
                    engine = feature_data.get("features", {}).get("engine", {})
                    perf = feature_data.get("features", {}).get("performance", {})
                    drivetrain = feature_data.get("features", {}).get("drivetrain", {})

                    df.at[i, "CO2Emissions"] = fuel.get("CO2Emissions")
                    df.at[i, "CityMPG"] = fuel.get("cityMPG")
                    df.at[i, "Horsepower"] = engine.get("horsepower")
                    df.at[i, "TorqueFtLbs"] = engine.get("torqueFtLBS")
                    df.at[i, "EngineSize"] = engine.get("size")
                    df.at[i, "ZeroTo60MPH"] = perf.get("ZeroTo60MPH")
                    df.at[i, "DrivetrainType"] = drivetrain.get("type")

            print("[DEBUG] enrichment end")
            # Save in memory
            self._vehicle_lookup = {int(row["Id"]): dict(row) for _, row in df.iterrows()}
            self._vehicle_cache = df

            # Save to Redis 
            if redis:
                try:
                    await redis.set(self.VEHICLE_CACHE_KEY, pickle.dumps(df))
                except Exception as e:
                    print(f"[WARN] Failed to store vehicles in Redis: {e}")

            return df

    def get_vehicle_by_id(self, vehicle_id: int) -> Optional[Dict[str, Any]]:
        return self._vehicle_lookup.get(int(vehicle_id))

    @staticmethod
    def extract_vehicle_features(row: pd.Series | Dict[str, Any]) -> Dict[str, str]:
        getv = row.get if isinstance(row, dict) else row.get
        return {
            "Make": str(getv("Make", "")),
            "Model": str(getv("Model", "")),
            "Year": str(getv("Year", "")),
            "Price": str(getv("Price", "")),
            "Mileage": str(getv("Mileage", "")),
            "Color": str(getv("Color", "")),
            "FuelType": str(getv("FuelType", "")),
            "Transmission": str(getv("Transmission", "")),
            "Status": str(getv("Status", "")),
            "CO2Emissions": str(getv("CO2Emissions", "")),
            "CityMPG": str(getv("CityMPG", "")),
            "Horsepower": str(getv("Horsepower", "")),
            "TorqueFtLbs": str(getv("TorqueFtLbs", "")),
            "EngineSize": str(getv("EngineSize", "")),
            "ZeroTo60MPH": str(getv("ZeroTo60MPH", "")),
            "DrivetrainType": str(getv("DrivetrainType", "")),
        }