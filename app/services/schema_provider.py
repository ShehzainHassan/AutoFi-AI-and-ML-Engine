import os
import json
from functools import lru_cache
from app.services.column_metadata import COLUMN_METADATA
from typing import Optional

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
    "Users": ["Id", "Name", "Email", "CreatedUtc", "LastLoggedIn"],
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
def get_schema_context(query_type: Optional[str] = None) -> str:
    schema_text = "Database schema and relevant tables:\n"
    if query_type == "VEHICLE_SEARCH":
        allowed_tables = {"Vehicles", "Auctions", "Bids"}
    elif query_type == "AUCTION_SEARCH":
        allowed_tables = {"Auctions", "Bids", "AutoBids", "BidStrategies"}
    elif query_type == "USER_SPECIFIC":
        allowed_tables = set(RELEVANT_TABLES.keys())
    else:
        allowed_tables = set(RELEVANT_TABLES.keys())

    for table in sorted(allowed_tables):
        cols = RELEVANT_TABLES.get(table, [])
        schema_text += f"- {table}: columns {cols}\n"
        for col in cols:
            description = COLUMN_METADATA.get(table, {}).get(col, "No description available")
            schema_text += f"   - {col}: {description}\n"

    if query_type in {"VEHICLE_SEARCH", "AUCTION_SEARCH", "USER_SPECIFIC"}:
        schema_text += "\nEnums:\n"
        for enum_name, values in ENUMS.items():
            schema_text += f"- {enum_name}: {values}\n"

    if query_type == "VEHICLE_SEARCH":
        schema_text += "\nVehicle features data (from car-features.json):\n"
        schema_text += (
            "Each vehicle includes basic attributes like 'make', 'model', and 'year', along with a rich set of features:\n"
            "- drivetrain: type, transmission\n"
            "- engine: type, size, horsepower, torqueFtLBS, torqueRPM, valves, camType\n"
            "- fuelEconomy: fuelTankSize, combinedMPG, cityMPG, highwayMPG, CO2Emissions\n"
            "- performance: horsepower, torqueFtLBS, drivetrain, ZeroTo60MPH\n"
            "- measurements: doors, maximumSeating, heightInches, widthInches, lengthInches, wheelbaseInches, groundClearance, cargoCapacityCuFt, curbWeightLBS\n"
            "- options: list of available extras like 'Alloy wheels', 'Leather seats', etc.\n"
        )

    column_table_map = "\n".join(
        f"- {col} → {table}"
        for table in sorted(allowed_tables)
        for col in RELEVANT_TABLES.get(table, [])
    )
    schema_text += "\nColumn-to-table map:\n" + column_table_map

    return schema_text
