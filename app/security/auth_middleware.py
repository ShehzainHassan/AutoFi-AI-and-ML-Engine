from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime
from config.app_config import settings

security = HTTPBearer()

class AuthService:
    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256", jwt_audience: str | None = None):
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_audience = jwt_audience or settings.JWT_AUDIENCE

    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                audience=self.jwt_audience,
            )

            exp = payload.get("exp")
            now = datetime.utcnow().timestamp()

            if exp and now > exp:
                print("Token has expired")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )

            user_id = int(payload.get("sub") or payload.get("nameid"))
            payload["user_id"] = user_id
            print(f"[AuthService] Authenticated user_id: {user_id}")
            return payload

        except jwt.ExpiredSignatureError:
            print("Token signature has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError as e:
            print(f"Invalid token error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )