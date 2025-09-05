import os
import json
from functools import lru_cache
from app.services.column_metadata import COLUMN_METADATA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "car-features.json")

with open(DATA_PATH, "r") as f:
    CAR_FEATURES = json.load(f)

RELEVANT_TABLES = {
    "Vehicles": ["Id", "Make", "Model", "Year", "Price", "Mileage", "Color", "Transmission", "FuelType"],
    "Auctions": ["AuctionId", "VehicleId", "StartUtc", "EndUtc", "StartingPrice", "CurrentPrice", "Status", "CreatedUtc", "UpdatedUtc", "ScheduledStartTime", "PreviewStartTime", "IsReserveMet"],
    "Bids": ["BidId", "AuctionId", "UserId", "Amount", "IsAuto", "CreatedUtc"],
    "AutoBids": ["Id","UserId", "AuctionId", "MaxBidAmount", "CurrentBidAmount", "IsActive", "BidStrategyType", "CreatedAt", "UpdatedAt", "ExecutedAt"],
    "BidStrategies": ["AuctionId", "UserId", "Type", "BidDelaySeconds", "MaxBidsPerMinute", "MaxSpreadBids", "PreferredBidTiming", "CreatedAt", "UpdatedAt"],
    "Users": ["Id", "Name", "Email", "CreatedUtc", "LastLoggedIn", "Password"],
    "UserSavedSearches": ["UserId", "Search"],
    "UserInteractions": ["Id", "UserId", "VehicleId", "InteractionType", "CreatedAt"],
    "Watchlists": ["WatchlistId", "UserId", "AuctionId", "CreatedUtc"],
    "AuctionAnalytics": ["Id", "AuctionId", "TotalViews", "UniqueBidders", "TotalBids", "CompletionStatus", "StartPrice", "FinalPrice", "EngagementScore", "UpdatedAt"],
    "AnalyticsEvents": ["Id", "EventType", "UserId", "AuctionId", "EventData", "Source", "CreatedAt"],
}

# Enum references for AI understanding
ENUMS = {
    "BidStrategyType": ["Conservative", "Aggressive", "Incremental"],
    "PreferredBidTiming": ["Immediate", "LastMinute", "SpreadEvenly"],
    "AnalyticsEventType": ["AuctionView", "BidPlaced", "AuctionCompleted", "PaymentCompleted"],
    "AnalyticsSource": ["Web", "Mobile", "API"],
    "AuctionStatus": ["Scheduled", "PreviewMode", "Active", "Ended", "Cancelled"]
}

@lru_cache(maxsize=1)
def get_schema_context() -> str:
    schema_text = "Database schema and relevant tables:\n"
    for table, cols in RELEVANT_TABLES.items():
        schema_text += f"- {table}: columns {cols}\n"
        for col in cols:
            description = COLUMN_METADATA.get(table, {}).get(col, "No description available")
            schema_text += f"   - {col}: {description}\n"

    if ENUMS:
        schema_text += "\nEnums:\n"
        for enum_name, values in ENUMS.items():
            schema_text += f"- {enum_name}: {values}\n"

    schema_text += "\nVehicle features data (from car-features.json):\n"
    schema_text += (
        "Each vehicle has a 'make', 'model', 'year', and 'features'. Features include: "
        "drivetrain (type, transmission), engine (type, size, horsepower, torqueFtLBS, torqueRPM, valves, camType), "
        "fuelEconomy (fuelTankSize, combinedMPG, cityMPG, highwayMPG, CO2Emissions), "
        "performance (horsepower, torqueFtLBS, drivetrain, ZeroTo60MPH), "
        "measurements (doors, maximumSeating, heightInches, widthInches, lengthInches, wheelbaseInches, groundClearance, "
        "cargoCapacityCuFt, curbWeightLBS), and options (list of strings like 'Alloy wheels', 'Leather seats').\n"
    )    
    return schema_text