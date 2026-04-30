CREATE INDEX transaction_timestamp IF NOT EXISTS
FOR (t:Transaction) ON (t.timestamp);

CREATE INDEX transaction_amount IF NOT EXISTS
FOR (t:Transaction) ON (t.amount);

CREATE INDEX user_email IF NOT EXISTS
FOR (u:User) ON (u.email);

CREATE INDEX device_fingerprint IF NOT EXISTS
FOR (d:Device) ON (d.fingerprint);
