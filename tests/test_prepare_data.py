from app.services.recommendation_service import RecommendationService

service = RecommendationService()

interactions_df, vehicle_features_df = service.prepare_data()

print("Interactions Summary:")
print(interactions_df.head())

print("\nVehicle Features:")
print(vehicle_features_df.head())
