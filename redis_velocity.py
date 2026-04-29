import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

def _velocity_key(user_id: str, window: int) -> str:
    return f"velocity:txn_count:{window}s:user:{user_id}"