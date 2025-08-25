from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone
from pydantic import ValidationError
from fastapi.exceptions import RequestValidationError
from app.schemas.schemas import ErrorResponse, ErrorDetail
import json

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except (ValidationError, RequestValidationError) as e:
            error_response = ErrorResponse(
                error=ErrorDetail(code="VALIDATION_ERROR", message=str(e)),
                request_id=request.headers.get("X-Request-ID", "unknown"),
                timestamp=datetime.now(timezone.utc)
            )
            return JSONResponse(
                status_code=400,
                content=json.loads(error_response.model_dump_json())
            )
        except Exception:
            error_response = ErrorResponse(
                error=ErrorDetail(code="INTERNAL_ERROR", message="Something went wrong"),
                request_id=request.headers.get("X-Request-ID", "unknown"),
                timestamp=datetime.now(timezone.utc)
            )
            return JSONResponse(
                status_code=500,
                content=json.loads(error_response.model_dump_json())
            )