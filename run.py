from app.services.recommendation_service import RecommendationService

if __name__ == "__main__":
    service = RecommendationService(vehicle_limit=10000)

    # Run full training pipeline
    collaborative_model = service.load_models()

    if collaborative_model is None:
        print("\nNo saved model found. Training new model...")
        collaborative_model = service.train_models()

        # Save models after training
        service.save_models(collaborative_model)
    else:
        print("\nLoaded saved models successfully.")

    # Content-based recommendations
    vehicle_id = 10
    similar_vehicles = service.get_similar_vehicles(vehicle_id, n=5)
    print(f"\nTop 5 similar vehicles to vehicle_id={vehicle_id} (Content-Based): {similar_vehicles}")

    # Hybrid recommendations
    user_id = 15
    models = (service, collaborative_model, collaborative_model[3]) 
    hybrid_recommendations = service.get_hybrid_recommendations(user_id, models, top_n=5)
    print(f"\nTop 5 hybrid recommendations for user_id={user_id}: {hybrid_recommendations}")

    # Close DB connection
    service.close()
