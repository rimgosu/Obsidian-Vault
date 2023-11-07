## 1. 키스페이스 생성


```
CREATE KEYSPACE test_keyspace
WITH REPLICATION = {
	'class' : 'SimpleStrategy',
	'replication_factor' : 1
};
```

### 1-1. 키스페이스 확인

```
DESC test_keyspace;
```

## 2. 테이블 생성

```
CREATE TABLE test_keyspace.test_table ( 
	user_id UUID, 
	similarity float, 
	target_id UUID,
	PRIMARY KEY ((user_id), similarity)
);
```

### 2-1. 생성한 테이블 확인

```
DESCRIBE TABLE test_keyspace.test_table;
```