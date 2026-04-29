import redis
import uuid
import time

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

def _velocity_key(user_id: str, window: int) -> str:
    return f"velocity:txn_count:{window}s:user:{user_id}"

def incr_txn_count(user_id: str, window:int = 60) -> int:
    key = _velocity_key(user_id, window)

    now = time.time()
    txn_event_id = str(uuid.uuid4())
    redis_client.zadd(key, {txn_event_id: now})
    cutoff = now - window
    redis_client.zremrangebyscore(key, 0, cutoff)
    redis_client.expire(key, window)
    return redis_client.zcard(key)

    # count = redis_client.incr(key)
    # if count == 1:
    #     redis_client.expire(key, window)
    # return count

def get_txn_count(user_id: str, window: int = 60) -> str:
    key = _velocity_key(user_id=user_id, window=window)

    now = time.time()
    cutoff = now - window
    redis_client.zremrangebyscore(key, 0, cutoff)
    return redis_client.zcard(key)

    # value = redis_client.get(key)
    # if value is None:
    #     return 0
    # return int(value)