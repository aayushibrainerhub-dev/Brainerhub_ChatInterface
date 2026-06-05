# import os

# from datetime import datetime, timedelta, timezone
# from typing import Any

# import jwt
# from dotenv import load_dotenv

# load_dotenv()
# JWT_SECRET = os.getenv("JWT_SECRET")
# if not JWT_SECRET:
#     raise ValueError("JWT_SECRET not found in .env")


# def generate_jwt(user: Any) -> str:
#     expiration_time = datetime.now(timezone.utc) + timedelta(minutes=15)
#     payload = {
#         "user_id": user.id,
#         "email": user.email,
#         "tenant_id": user.tenant_id,
#         "role": user.role,
#         "exp": expiration_time,
#     }
#     return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# def generate_token_for_user(user: Any) -> str:

#     return generate_jwt(user)


