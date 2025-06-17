class RecommendationService:
    async def get_recommendations(self, user_id):
        return ["Car A", "Car B", "Car C"]

    async def get_similar_vehicles(self, vehicle_id):
        return ["Car X", "Car Y", "Car Z"]