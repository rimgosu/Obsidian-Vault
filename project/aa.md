```
@Override
	public <T> List<T> findAllByColumnValues(DriverConfigLoader loader, Class<T> classType, Map<String, Object> columnValues) {
	List<T> entities = new ArrayList<>();
	CqlSession session = CassandraSessionManager.getSession(loader);
	try {
		// 엔티티 클래스의 모든 필드 이름을 추출
		String fields = Arrays.stream(classType.getDeclaredFields())
							  .map(Field::getName)
							  .collect(Collectors.joining(", "));

		// WHERE 절 동적 생성
		StringBuilder whereClause = new StringBuilder();
		List<Object> bindValues = new ArrayList<>();
		for (Map.Entry<String, Object> entry : columnValues.entrySet()) {
			if (whereClause.length() > 0) {
				whereClause.append(" AND ");
			}
			whereClause.append(entry.getKey()).append(" = ?");
			bindValues.add(entry.getValue());
		}

		// SELECT 절에 필드 이름 포함
		String cql = String.format("SELECT %s FROM %s WHERE %s", 
								   fields, "member." + classType.getSimpleName().toLowerCase(), whereClause.toString());
		cql += " ALLOW FILTERING";

		System.out.println("[execute cql] " + cql);

		PreparedStatement preparedStatement = session.prepare(cql);
		// 바인딩된 값 추가
		BoundStatement boundStatement = preparedStatement.bind(bindValues.toArray());
		ResultSet resultSet = session.execute(boundStatement);
		for (Row row : resultSet) {
			T entity = classType.getDeclaredConstructor().newInstance();

			for (Field field : classType.getDeclaredFields()) {
				field.setAccessible(true); // 필드 접근 허용

				try {
					setFieldValue(field, entity, row);
				} catch (IllegalAccessException e) {
					System.out.println("Reflection error: " + e.getMessage());
					// 적절한 예외 처리
				}
			}
			
			entities.add(entity);
		}

	} catch (Exception e) {
		// 오류 처리 로직
		System.out.println("Error: " + e);
	}

	return entities;
}
```

```
@Override
public <T> List<T> findAll(DriverConfigLoader loader, Class<T> classType) {
	List<T> entities = new ArrayList<>();
	CqlSession session = CassandraSessionManager.getSession(loader);
	try {

		String cql = String.format("SELECT * FROM %s", "member." + classType.getSimpleName().toLowerCase());

		PreparedStatement preparedStatement = session.prepare(cql);
		ResultSet resultSet = session.execute(preparedStatement.bind());
		for (Row row : resultSet) {
			T entity = classType.getDeclaredConstructor().newInstance();

			for (Field field : classType.getDeclaredFields()) {
				field.setAccessible(true); // 필드 접근 허용

				try {
					setFieldValue(field, entity, row);
				} catch (IllegalAccessException e) {
					System.out.println("Reflection error: " + e.getMessage());
					// 적절한 예외 처리
				}
			}

			entities.add(entity);
		}

	} catch (Exception e) {
		// 오류 처리 로직
		System.out.println("Error: " + e);
	}

	return entities;
}
```