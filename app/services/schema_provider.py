import os
import json
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "car-features.json")

with open(DATA_PATH, "r") as f:
    CAR_FEATURES = json.load(f)

RELEVANT_TABLES = {
    "Vehicles": ["Id", "Make", "Model", "Year", "Price", "Mileage", "Color", "Transmission", "FuelType"],
    "Auctions": ["AuctionId", "VehicleId", "StartUtc", "EndUtc", "CurrentPrice", "Status", "ReservePrice"],
    "Bids": ["BidId", "AuctionId", "UserId", "Amount", "IsAuto", "CreatedUtc"],
    "AutoBids": ["UserId", "AuctionId", "MaxBidAmount", "CurrentBidAmount", "IsActive", "BidStrategyType"],
    "BidStrategies": ["UserId", "Type", "BidDelaySeconds", "MaxBidsPerMinute", "PreferredBidTiming"],
    "Users": ["Id", "Name", "Email"],
    "UserSavedSearches": ["UserId", "Search"],
    "UserInteractions": ["Id", "UserId", "VehicleId", "InteractionType", "CreatedAt"],
    "Watchlists": ["WatchlistId", "UserId", "AuctionId", "CreatedUtc"],
    "AuctionAnalytics": ["AuctionId", "TotalViews", "UniqueBidders", "TotalBids", "StartPrice", "FinalPrice", "EngagementScore"],
    "AnalyticsEvents": ["Id", "EventType", "UserId", "AuctionId", "EventData", "Source", "CreatedAt"],
}

# Enum references for AI understanding
ENUMS = {
    "BidStrategyType": ["Conservative", "Aggressive", "Incremental"]
}
@lru_cache(maxsize=1)
def get_schema_context() -> str:
    """
    Generate dynamic schema context for AI prompts.
    Includes table columns, sample vehicle features, and relevant enums.
    """
    schema_text = "Database schema and relevant tables:\n"
    for table, cols in RELEVANT_TABLES.items():
        schema_text += f"- {table}: columns {cols}\n"

    if ENUMS:
        schema_text += "\nEnums:\n"
        for enum_name, values in ENUMS.items():
            schema_text += f"- {enum_name}: {values}\n"

    schema_text += "\nSample vehicle features:\n"
    for vehicle in CAR_FEATURES[:3]:
        make = vehicle.get("make")
        model = vehicle.get("model")
        year = vehicle.get("year")
        features = vehicle.get("features", {})
        schema_text += f"- {year} {make} {model}: {features}\n"

    return schema_text