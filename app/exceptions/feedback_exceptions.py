class FeedbackServiceError(Exception):
    """Base exception for feedback service errors."""
    def __init__(self, message: str, error_code: str = "FEEDBACK_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class MessageNotFoundError(FeedbackServiceError):
    """Raised when a message is not found."""
    def __init__(self, message_id: int):
        super().__init__(
            message=f"Message {message_id} not found",
            error_code="MESSAGE_NOT_FOUND"
        )
