

## 1. CREATE KEYSPACE

```
CREATE KEYSPACE cycling
WITH REPLICATION = {
	'class' : 'SimpleStrategy',
	'replication_factor' : 1
};
```


## 2. CREATE TABLE

```
CREATE TABLE cycling.cyclist_category (
	category text,
	points int,
	id UUID,
	lastname text,
	PRIMARY KEY (category, points)
) WITH CLUSTERING ORDER BY (points DESC);
```