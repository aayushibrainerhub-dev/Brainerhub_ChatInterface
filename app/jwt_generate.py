import os
from dotenv import load_dotenv
import jwt
from datetime import datatime, timedelta, timezone

load_dotenv()

def generate_jwt(user):
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise ValueError("JWT_SECRET not found in .env")

    expiration_time = datetime.now(timezone.utc) + timedelta(minutes = 15)
    payload = {
        "user_id": user.id,
        "email": user.email,
        "tenant_id": user.tenant_id,
        "role": user.role,
        "exp" : expiration_time,
    }

    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

if __name__ == "__main__":
    print(generate_jwt())


    # need to add validation here