from app.utils.database_entity_extractor import extract_query_entities, DatabaseEntities
from typing import Optional
from collections import Counter

def format_context_for_prompt(context: Optional[dict]) -> str:
    """Format user context into labeled sections + compact JSON fallback."""
    if not context:
        return ""

    parts = []
    dotnet_ctx = context.get("dotnet_context", {})
    ml_ctx = context.get("ml_context", {})
    parts.append("User Context Summary: \n")
    # .NET Context Summary
    if dotnet_ctx:
        auction_history = dotnet_ctx.get("auction_history", [])
        auto_bids = dotnet_ctx.get("auto_bid_settings", [])
        saved_searches = dotnet_ctx.get("saved_searches", [])

        won_count = sum(1 for a in auction_history if a.get("is_winner"))
        parts.append(
            f"- Auctions participated: {len(auction_history)}\n"
            f"- Auctions won: {won_count}\n"
            f"- Auto-bid configurations: {len(auto_bids)}\n"
            f"- Saved searches: {len(saved_searches)}\n"
            f"(Includes bidding history, auto-bid strategies, and personalized search preferences.)"
        )

    # ML Context Summary
    if ml_ctx:
        user_id = ml_ctx.get("user_id")
        user_name = ml_ctx.get("user_name")
        user_email = ml_ctx.get("user_email")
        interactions = ml_ctx.get("user_interactions", [])
        analytics_events = ml_ctx.get("analytics_events", [])

        interaction_counts = Counter(i.get("InteractionType") for i in interactions)
        event_counts = Counter(e.get("EventType") for e in analytics_events)

        parts.append(
            f"- User ID: {user_id}\n"
            f"- Name: {user_name}\n"
            f"- Email: {user_email}\n"
            f"- Recent interactions: {len(interactions)} (by type: {', '.join(f'{k}: {v}' for k, v in interaction_counts.items())})\n"
            f"- Recent behavioral events: {len(analytics_events)} (by type: {', '.join(f'{k}: {v}' for k, v in event_counts.items())})\n"
            f"(Reflects the user's latest engagement patterns for personalization.)"
        )
    return "\n".join(parts)

def build_optimized_context(query_type: str, user_query: str, user_id: int, context: dict) -> dict:
    """Build minimal, targeted context based on actual database entities"""
    
    # Extract database entities (~15ms, no API calls, all-MiniLM-L6-v2 model is loaded locally)
    entities = extract_query_entities(user_query)
    
    context_parts = {
        "query_type": query_type,
        "user_id": user_id,
        "schema_context": get_targeted_database_schema(query_type, entities),
        "user_context": format_context_for_prompt(context) if query_type == "USER_SPECIFIC" else ""
    }
    
    return context_parts

def get_targeted_database_schema(query_type: str, entities: DatabaseEntities) -> str:
    """Return only relevant AutoFiCore schema parts based on extracted entities"""
    
    if query_type == "GENERAL" or query_type == "FINANCE_CALC":
        return "No database access required - use general knowledge"
    
    schema_parts = []
    
    # Build schema based on actual needed tables and columns
    for table_name in entities.tables_needed:
        if table_name in entities.columns_needed:
            columns = list(entities.columns_needed[table_name])
            schema_parts.append(f"{table_name}: {', '.join(columns)}")
            
            # Add column descriptions for key fields
            if table_name == "Vehicles" and "Price" in columns:
                schema_parts.append("- Price: Vehicle listing price in USD")
            elif table_name == "Auctions" and "Status" in columns:
                schema_parts.append("- Status: Scheduled|PreviewMode|Active|Ended|Cancelled")
            elif table_name == "Bids" and "IsAuto" in columns:
                schema_parts.append("- IsAuto: true if placed by auto-bidding system")
    
    # Add relationships if multiple tables needed
    if len(entities.tables_needed) > 1:
        schema_parts.append("\nKey Relationships:")
        if "Vehicles" in entities.tables_needed and "Auctions" in entities.tables_needed:
            schema_parts.append("- Auctions.VehicleId -> Vehicles.Id")
        if "Auctions" in entities.tables_needed and "Bids" in entities.tables_needed:
            schema_parts.append("- Bids.AuctionId -> Auctions.AuctionId")
        if "VehicleFeatures" in entities.tables_needed and "Vehicles" in entities.tables_needed:
            schema_parts.append("- VehicleFeatures.Make/Model -> Vehicles.Make/Model/")
        if "Users" in entities.tables_needed:
            schema_parts.append("- Bids.UserId -> Users.Id")
    
    return "\n".join(schema_parts)