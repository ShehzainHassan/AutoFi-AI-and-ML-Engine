from app.services.recommendation_service import RecommendationService

if __name__ == "__main__":
    service = RecommendationService(vehicle_limit=20000)

    # Content-based recommendations
    vehicle_id = 1
    similar_vehicles = service.get_similar_vehicles(vehicle_id, top_n=5)
    print(f"\nTop 5 similar vehicles to vehicle_id={vehicle_id} (Content-Based): {similar_vehicles}")

    # Hybrid recommendations
    user_id = 15
    hybrid_recommendations = service.get_hybrid_recommendations(user_id, top_n=5)
    print(f"\nTop 5 hybrid recommendations for user_id={user_id}: {hybrid_recommendations}")

    # Close DB connection
    service.close()
