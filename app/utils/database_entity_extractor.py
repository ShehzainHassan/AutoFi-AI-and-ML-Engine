from typing import Dict, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from functools import lru_cache

@dataclass
class DatabaseEntities:
    """Extracted database entities from user query for schema optimization"""
    tables_needed: Set[str]
    columns_needed: Dict[str, Set[str]]
    relationships_needed: Set[str]
    confidence_scores: Dict[str, float]

class DatabaseEntityExtractor:
    """Extract database-specific entities from user queries based on schema"""
    
    def __init__(self):
        # Load tiny model once at startup (~20MB)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._build_schema_embeddings()
    
    @lru_cache(maxsize=1)
    def _build_schema_embeddings(self):
        """Build embeddings for database schema elements"""
        
        # Core database tables
        self.SCHEMA_TABLES = {
            "Vehicles": {
                "description": "vehicle information make model year price mileage color fuel transmission",
                "columns": ["Id", "Vin", "Make", "Model", "Year", "Price", "Mileage", "Color", "FuelType", "Transmission", "Status"],
                "relationships": ["has Auction", "has UserInteractions", "has Drivetrain", "has Engine", "has FuelEconomy"]
            },
            "Auctions": {
                "description": "auction bidding live active ended reserve price starting current",
                "columns": ["AuctionId", "VehicleId", "StartUtc", "EndUtc", "StartingPrice", "CurrentPrice", "ReservePrice", "IsReserveMet", "Status", "ScheduledStartTime", "PreviewStartTime"],
                "relationships": ["belongs to Vehicle", "has many Bids", "has many Watchlists", "has many AutoBids"]
            },
            "Bids": {
                "description": "bid bidding amount user auction manual automatic",
                "columns": [],
                "relationships": []
            },
            "AutoBids": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "Users": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "UserInteractions": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "Watchlists": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "UserSavedSearches": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "AnalyticsEvents": {
                "description": "",
                "columns": [],
                "relationships": []
            },
            "VehicleFeatures": {
                "description": "vehicle features specifications engine drivetrain fuel economy performance options measurements",
                "columns": ["Make", "Model", "Drivetrain", "Engine", "FuelEconomy", "Performance", "Measurements", "Options"],
                "relationships": ["belongs to Vehicle"]
            }
        }
        
        # Build embeddings for each table's description
        self.table_embeddings = {}
        for table_name, table_info in self.SCHEMA_TABLES.items():
            embedding = self.model.encode([table_info["description"]])
            self.table_embeddings[table_name] = embedding
        
        # Common query patterns and their associated tables
        self.QUERY_PATTERNS = {
            "vehicle_search": {
                "embedding": self.model.encode(["find cars vehicles SUV sedan truck Toyota Honda BMW price under over"]),
                "primary_tables": ["Vehicles"],
                "secondary_tables": ["Auctions", "UserInteractions", "VehicleFeatures"]
            },
            "auction_queries": {
                "embedding": self.model.encode(["auction bidding live active ended reserve current starting price"]),
                "primary_tables": ["Auctions", "Vehicles"],
                "secondary_tables": ["Bids", "Watchlists", "AutoBids"]
            },
            "user_specific": {
                "embedding": self.model.encode(["my mine I me user account history saved viewed purchased owned my recent interactions my activity history things I clicked on"]),                
                "primary_tables": ["Users"],
                "secondary_tables": ["UserInteractions", "Bids", "Watchlists", "UserSavedSearches"]
            },
            "bidding_activity": {
                "embedding": self.model.encode(["bid bidding placed won lost maximum automatic manual strategy"]),
                "primary_tables": ["Bids", "AutoBids"],
                "secondary_tables": ["Auctions", "Users"]
            },
            "financial_queries": {
                "embedding": self.model.encode(["price cost budget payment loan finance affordable expensive"]),
                "primary_tables": ["Vehicles"],
                "secondary_tables": ["Auctions", "Bids", "VehicleFeatures"]
            }
        }

    def extract_query_entities(self, user_query: str) -> DatabaseEntities:
        """
        Extract database entities from user query for schema optimization
        
        Args:
            user_query: Raw user input query
            
        Returns:
            DatabaseEntities: Tables, columns, and relationships needed for the query
        """
        query_embedding = self.model.encode([user_query])
        
        # Calculate similarity scores for each query pattern
        pattern_scores = {}
        for pattern_name, pattern_info in self.QUERY_PATTERNS.items():
            similarity = F.cosine_similarity(
                torch.tensor(query_embedding), 
                torch.tensor(pattern_info["embedding"]), 
                dim=1
            ).item()
            pattern_scores[pattern_name] = similarity
        
        # Calculate similarity scores for each table
        table_scores = {}
        for table_name, embedding in self.table_embeddings.items():
            similarity = F.cosine_similarity(
                torch.tensor(query_embedding), 
                torch.tensor(embedding), 
                dim=1
            ).item()
            table_scores[table_name] = similarity
        
        # Determine needed tables based on highest scoring patterns and direct table matches
        tables_needed = set()
        columns_needed = {}
        relationships_needed = set()
        
        # Add tables from high-scoring patterns
        for pattern_name, score in pattern_scores.items():
            if score > 0.3:  # Threshold for pattern relevance
                pattern_info = self.QUERY_PATTERNS[pattern_name]
                tables_needed.update(pattern_info["primary_tables"])
                if score > 0.5:  # Higher threshold for secondary tables
                    tables_needed.update(pattern_info["secondary_tables"])
        
        # Add tables with high direct similarity
        for table_name, score in table_scores.items():
            if score > 0.4:  # Threshold for direct table relevance
                tables_needed.add(table_name)
        
        # Build columns and relationships for needed tables
        for table_name in tables_needed:
            if table_name in self.SCHEMA_TABLES:
                table_info = self.SCHEMA_TABLES[table_name]
                columns_needed[table_name] = set(table_info["columns"])
                relationships_needed.update(table_info["relationships"])
        
        # Optimize columns based on query content
        columns_needed = self._optimize_columns_for_query(user_query, columns_needed, pattern_scores)
        
        return DatabaseEntities(
            tables_needed=tables_needed,
            columns_needed=columns_needed,
            relationships_needed=relationships_needed,
            confidence_scores={**pattern_scores, **table_scores}
        )
    
    def _optimize_columns_for_query(self, query: str, columns_needed: Dict[str, Set[str]], pattern_scores: Dict[str, float]) -> Dict[str, Set[str]]:
        """Optimize column selection based on query content"""
        query_lower = query.lower()
        optimized_columns = {}
        
        for table_name, columns in columns_needed.items():
            essential_columns = set()
            
            if table_name == "Vehicles":
                # Always include basic vehicle identification
                essential_columns.update(["Id", "Make", "Model", "Year"])
                
                # Add price if financial query
                if any(word in query_lower for word in ["price", "cost", "budget", "under", "over", "$", "expensive", "cheap"]):
                    essential_columns.add("Price")
                
                # Add specs if mentioned
                if any(word in query_lower for word in ["mileage", "miles", "color", "fuel", "transmission", "gas", "electric"]):
                    essential_columns.update(["Mileage", "Color", "FuelType", "Transmission"])
                
            elif table_name == "Auctions":
                # Always include basic auction info
                essential_columns.update(["AuctionId", "VehicleId", "Status"])
                
                # Add pricing if relevant
                if pattern_scores.get("auction_queries", 0) > 0.4:
                    essential_columns.update(["StartingPrice", "CurrentPrice", "ReservePrice", "IsReserveMet"])
                
                # Add timing if time-related query
                if any(word in query_lower for word in ["time", "when", "start", "end", "live", "active", "ended"]):
                    essential_columns.update(["StartUtc", "EndUtc", "ScheduledStartTime", "PreviewStartTime"])
                
            elif table_name == "Bids":
                essential_columns.update(["BidId", "AuctionId", "UserId", "Amount", "CreatedUtc"])
                
                if "auto" in query_lower or "automatic" in query_lower:
                    essential_columns.add("IsAuto")
                    
            elif table_name == "Users":
                essential_columns.update(["Id", "Name", "Email"])
                
                if any(word in query_lower for word in ["created", "joined", "login", "last"]):
                    essential_columns.update(["CreatedUtc", "LastLoggedIn"])
            
            elif table_name == "VehicleFeatures":
                essential_columns.update(["Make", "Model", "Drivetrain", "Engine", "FuelEconomy"])

                if any(word in query_lower for word in ["performance", "hp", "horsepower", "torque"]):
                    essential_columns.add("Performance")

                if any(word in query_lower for word in ["dimensions", "size", "length", "width", "height", "weight"]):
                    essential_columns.add("Measurements")

                if any(word in query_lower for word in ["options", "features", "trim", "package"]):
                    essential_columns.add("Options")

            else:
                # For other tables, include all columns by default
                essential_columns = columns
            
            optimized_columns[table_name] = essential_columns
        
        return optimized_columns

# Global instance for use in services
database_entity_extractor = DatabaseEntityExtractor()

def extract_query_entities(user_query: str) -> DatabaseEntities:
    """
    Main function to extract database entities from user query
    
    Usage in Smart Context Builder:
        entities = extract_query_entities(user_query)
        schema_context = get_targeted_database_schema(query_type, entities)
    """
    return database_entity_extractor.extract_query_entities(user_query)