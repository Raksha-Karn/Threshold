# Feature Schemas

## AnomalyAgent

Input fields:

- amount
- hour_of_day
- day_of_week
- amount_zscore
- user_mean_amount
- amount_vs_user_mean
- txn_count_in_last_1h
- txn_count_in_last_24h
- is_new_merchant

Output:

- score(transaction) -> float between 0 and 1

## BehaviourAgent

Input fields are loaded from:

`artifacts/preprocessors/sequence_metadata.json`

The LSTM uses the last N scaled transaction feature vectors for each user.

Output:

- score(user_id, transaction) -> weighted MSE behavior anomaly score

## VelocityRulesAgent

Model features:

- txns_last_1h
- txns_last_24h
- amount_sum_last_1h
- amount_sum_last_24h
- amount_mean_last_24h
- amount_max_last_24h
- unique_merchants_last_24h
- unique_devices_last_24h
- is_new_device
- is_new_city
- geo_distance_from_home
- card_present_flag
- is_international
- amount_round_number

Redis live fields:

- user_id
- timestamp_epoch
- amount
- merchant_id
- device_id
- city
- geo_distance_from_home
- card_present_flag
- is_international

Output:

- score(transaction) -> fraud probability float

## GraphAgent

Node features:

- linked_device_count
- shared_ip_count
- ring_density
- avg_geo_distance_from_home
- max_geo_distance_from_home
- is_international_rate
- txn_count_last_24h
- amount_sum_last_24h
- fraud_score_mean
- anomaly_score_mean
- unique_merchants
- unique_devices
- new_device_rate
- new_city_rate
- card_present_rate
- merchant_risk_mean

Output:

- score(user_id) -> fraud probability float

## GeoAgent

Input fields:

- user_id
- lat
- lon
- home_lat
- home_lon
- timestamp

Logic:

- Haversine distance from home
- Distance from last 5-location centroid
- Impossible travel speed flag if speed > 900 km/h

Output:

- score(transaction) -> float between 0 and 1