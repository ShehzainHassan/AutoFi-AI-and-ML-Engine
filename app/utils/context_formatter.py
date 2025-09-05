import simplejson as json
from datetime import datetime
from typing import Optional
from collections import Counter
 
def json_safe(obj):
    """Safely serialize datetime and other non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def format_context_for_prompt(context: Optional[dict]) -> str:
    """Format user context into labeled sections + compact JSON fallback."""
    if not context:
        return ""

    parts = []
    dotnet_ctx = context.get("dotnet_context", {})
    ml_ctx = context.get("ml_context", {})

    # .NET Context Summary
    if dotnet_ctx:
        auction_history = dotnet_ctx.get("auction_history", [])
        auto_bids = dotnet_ctx.get("auto_bid_settings", [])
        saved_searches = dotnet_ctx.get("saved_searches", [])

        won_count = sum(1 for a in auction_history if a.get("is_winner"))
        parts.append(
            f".NET Context Summary:\n"
            f"- Auctions participated: {len(auction_history)}\n"
            f"- Auctions won: {won_count}\n"
            f"- Auto-bid configurations: {len(auto_bids)}\n"
            f"- Saved searches: {len(saved_searches)}\n"
            f"(Includes bidding history, auto-bid strategies, and personalized search preferences.)"
        )

    # ML Context Summary
    if ml_ctx:
        interactions = ml_ctx.get("user_interactions", [])
        analytics_events = ml_ctx.get("analytics_events", [])

    # Interaction summary
    interaction_counts = Counter(i.get("InteractionType") for i in interactions)
    parts.append(
        f"ML Context Summary (last {min(5,len(interactions))} interactions/events):\n"
        f"- Recent interactions: {len(interactions)} (by type: {', '.join(f'{k}: {v}' for k, v in interaction_counts.items())})\n"
        f"- Recent behavioral events: {len(analytics_events)} "
        f"(by type: {', '.join(f'{k}: {v}' for k, v in Counter(e.get('EventType') for e in analytics_events).items())})\n"
        "(Reflects the user's latest engagement patterns for personalization.)"
    )


    compact_json = json.dumps(context, separators=(",", ":"), default=json_safe)

    return "\n".join(parts) + "\nRaw Context (compact JSON, for reference only):\n" + compact_json