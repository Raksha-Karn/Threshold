import json
from collections import defaultdict


class FakePipeline:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.commands = []

    def get(self, key):
        self.commands.append(("get", key))
        return self

    def setex(self, key, ttl, value):
        self.commands.append(("setex", key, ttl, value))
        return self

    def set(self, key, value, ex=None):
        self.commands.append(("set", key, value, ex))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def zremrangebyscore(self, key, min, max):
        self.commands.append(("zremrangebyscore", key, min, max))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def execute(self):
        results = []
        for command in self.commands:
            name = command[0]
            if name == "get":
                results.append(self.redis_client.get(command[1]))
            elif name == "setex":
                _, key, ttl, value = command
                self.redis_client.setex(key, ttl, value)
                results.append(True)
            elif name == "set":
                _, key, value, ex = command
                self.redis_client.set(key, value, ex=ex)
                results.append(True)
            elif name == "zadd":
                _, key, mapping = command
                self.redis_client.zadd(key, mapping)
                results.append(True)
            elif name == "zremrangebyscore":
                _, key, min_score, max_score = command
                self.redis_client.zremrangebyscore(key, min_score, max_score)
                results.append(True)
            elif name == "expire":
                results.append(True)
        self.commands = []
        return results


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sorted_sets = defaultdict(list)

    def pipeline(self):
        return FakePipeline(self)

    def setex(self, key, ttl, value):
        self.values[key] = value
        return True

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def exists(self, key):
        return int(key in self.values)

    def ttl(self, key):
        return -1 if key in self.values else -2

    def expire(self, key, ttl):
        return True

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.sorted_sets.pop(key, None)
        return len(keys)

    def zadd(self, key, mapping):
        for member, score in mapping.items():
            self.sorted_sets[key].append((float(score), member))
        return len(mapping)

    def zrangebyscore(self, key, min, max):
        min_score = float(min)
        max_score = float(max)
        return [
            member
            for score, member in self.sorted_sets.get(key, [])
            if min_score <= score <= max_score
        ]

    def zremrangebyscore(self, key, min, max):
        min_score = float(min)
        max_score = float(max)
        existing = self.sorted_sets.get(key, [])
        self.sorted_sets[key] = [
            (score, member)
            for score, member in existing
            if not min_score <= score <= max_score
        ]

    def dump_json(self, key):
        raw = self.get(key)
        return json.loads(raw) if raw else None
