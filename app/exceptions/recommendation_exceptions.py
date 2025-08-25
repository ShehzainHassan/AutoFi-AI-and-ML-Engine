class RecommendationServiceError(Exception):
    """Base exception for recommendation service errors."""
    def __init__(self, message: str, error_code: str = "RECOMMENDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class UserNotFoundError(RecommendationServiceError):
    """Raised when a user is not found in the system."""
    def __init__(self, user_id: int):
        super().__init__(
            message=f"User {user_id} not found",
            error_code="USER_NOT_FOUND"
        )

class ModelNotAvailableError(RecommendationServiceError):
    """Raised when the requested model is not available."""
    def __init__(self, model_type: str):
        super().__init__(
            message=f"Model '{model_type}' not available",
            error_code="MODEL_NOT_AVAILABLE"
        )

class InsufficientDataError(RecommendationServiceError):
    """Raised when there's not enough data to generate recommendations."""
    def __init__(self, user_id: int):
        super().__init__(
            message=f"Insufficient interaction data for user {user_id}",
            error_code="INSUFFICIENT_DATA"
        )

class VehicleNotFoundError(Exception):
    def __init__(self, vehicle_id: int):
        self.vehicle_id = vehicle_id
        self.message = f"Vehicle with ID {vehicle_id} not found."
        super().__init__(self.message)