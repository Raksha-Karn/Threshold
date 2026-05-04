import math
from collections import defaultdict, deque
from datetime import datetime, timezone


class GeoAgent:
    def __init__(
        self,
        max_history: int = 5,
        impossible_speed_kmh: float = 900.0,
        home_distance_high_risk_km: float = 1000.0,
        centroid_distance_high_risk_km: float = 500.0,
    ):
        self.max_history = max_history
        self.impossible_speed_kmh = impossible_speed_kmh
        self.home_distance_high_risk_km = home_distance_high_risk_km
        self.centroid_distance_high_risk_km = centroid_distance_high_risk_km

        self.user_locations = defaultdict(lambda: deque(maxlen=self.max_history))

    @staticmethod
    def haversine_distance_km(lat1, lon1, lat2, lon2) -> float:
        earth_radius_km = 6371.0

        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad)
            * math.cos(lat2_rad)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.asin(math.sqrt(a))

        return earth_radius_km * c

    @staticmethod
    def _parse_timestamp(timestamp):
        if timestamp is None:
            return datetime.now(timezone.utc)

        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp

        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    def _centroid_distance(self, user_id: str, lat: float, lon: float) -> float:
        history = self.user_locations[user_id]

        if not history:
            return 0.0

        centroid_lat = sum(item["lat"] for item in history) / len(history)
        centroid_lon = sum(item["lon"] for item in history) / len(history)

        return self.haversine_distance_km(
            centroid_lat,
            centroid_lon,
            lat,
            lon,
        )

    def _speed_from_last_location(self, user_id: str, lat: float, lon: float, timestamp) -> float:
        history = self.user_locations[user_id]

        if not history:
            return 0.0

        last_location = history[-1]

        distance_km = self.haversine_distance_km(
            last_location["lat"],
            last_location["lon"],
            lat,
            lon,
        )

        current_time = self._parse_timestamp(timestamp)
        previous_time = self._parse_timestamp(last_location["timestamp"])

        hours_elapsed = (current_time - previous_time).total_seconds() / 3600

        if hours_elapsed <= 0:
            return 0.0

        return distance_km / hours_elapsed

    def score(self, transaction: dict, update_history: bool = True) -> float:
        required_fields = ["user_id", "lat", "lon"]

        missing_fields = [
            field for field in required_fields
            if field not in transaction
        ]

        if missing_fields:
            raise ValueError(f"Missing geo fields: {missing_fields}")

        user_id = str(transaction["user_id"])
        lat = float(transaction["lat"])
        lon = float(transaction["lon"])

        timestamp = transaction.get("timestamp")

        home_lat = transaction.get("home_lat")
        home_lon = transaction.get("home_lon")

        if home_lat is not None and home_lon is not None:
            home_distance_km = self.haversine_distance_km(
                home_lat,
                home_lon,
                lat,
                lon,
            )
        else:
            home_distance_km = 0.0

        centroid_distance_km = self._centroid_distance(user_id, lat, lon)

        speed_kmh = self._speed_from_last_location(
            user_id=user_id,
            lat=lat,
            lon=lon,
            timestamp=timestamp,
        )

        home_score = min(
            home_distance_km / self.home_distance_high_risk_km,
            1.0,
        )

        centroid_score = min(
            centroid_distance_km / self.centroid_distance_high_risk_km,
            1.0,
        )

        speed_score = 1.0 if speed_kmh > self.impossible_speed_kmh else 0.0

        final_score = max(home_score, centroid_score, speed_score)

        if update_history:
            self.user_locations[user_id].append(
                {
                    "lat": lat,
                    "lon": lon,
                    "timestamp": self._parse_timestamp(timestamp).isoformat(),
                }
            )

        return float(final_score)

    def is_suspicious(self, transaction: dict, threshold: float = 0.5) -> bool:
        return self.score(transaction) >= threshold