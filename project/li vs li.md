활성화된 li

```
<li class="p-2 border-bottom" style="background-color: #eee;">
  <a href="#!" class="chat-link d-flex justify-content-between" data-room-uuid="73cb3f6b-6b98-4908-b229-289aece6d113">
	<div class="d-flex flex-row">
	  <img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-8.webp" alt="avatar" class="rounded-circle d-flex align-self-center me-3 shadow-1-strong" width="60">
	  <div class="pt-1">
		<p class="fw-bold mb-0">b@gmail.com</p>
		<p class="small text-muted">Hello, Are you there?</p>
	  </div>
	</div>
	<div class="pt-1 chatRoomTime">
	  <span id="timeAgo"></span>
	  <span class="badge bg-danger float-end">1</span>
	</div>
  </a>
</li>
```

비활성화된 li

```
<li class="p-2 border-bottom">
	<a href="#!" class="d-flex justify-content-between">
	  <div class="d-flex flex-row">
		<img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-1.webp" alt="avatar" class="rounded-circle d-flex align-self-center me-3 shadow-1-strong" width="60">
		<div class="pt-1">
		  <p class="fw-bold mb-0">Danny Smith</p>
		  <p class="small text-muted">Lorem ipsum dolor sit.</p>
		</div>
	  </div>
	  <div class="pt-1">
		<p class="small text-muted mb-1">5 mins ago</p>
	  </div>
	</a>
</li>
```


chattingChatting(chat_uuid=c5d7d743-a54a-4926-b2e7-1f689f66b6e9, chatted_at=2023-11-16T06:39:02.783914500Z, room_uuid=e493b575-4065-4d7c-9c6a-2c7affbc8135, chat_chatter=a@gmail.com, chat_content=첫번째 채팅입니다., read_status=true, chat_emoticon=null)
 
[DBServiceImpl][save]

```
[cql:]INSERT INTO member.Chatting (chat_chatter, chat_content, chat_emoticon, chat_uuid, chatted_at, read_status, room_uuid) VALUES (?, ?, ?, ?, ?, ?, ?)
[result.values:]
Value: a@gmail.com, Class: java.lang.String
Value: 첫번째 채팅입니다., Class: java.lang.String
Value: null (The type cannot be determined)
Value: c82a56b4-f1d5-47e5-a76a-c4c0f7953c17, Class: java.util.UUID
Value: 2023-11-16T06:52:58.555199800Z, Class: java.time.Instant
Value: e493b575-4065-4d7c-9c6a-2c7affbc8135, Class: java.util.UUID
Value: a@gmail.com, Class: String
Value: 첫번째 채팅입니다., Class: String
Value: null
Value: c82a56b4-f1d5-47e5-a76a-c4c0f7953c17, Class: UUID
Value: 2023-11-16T06:52:58.555199800Z, Class: Instant
Value: e493b575-4065-4d7c-9c6a-2c7affbc8135, Class: UUID
save Error: com.datastax.oss.driver.api.core.type.codec.CodecNotFoundException: Codec not found for requested operation: [BOOLEAN <-> java.util.UUID]
```



```
@Override
	public <T> List<T> findAll(DriverConfigLoader loader, Class<T> classType) {
	    List<T> entities = new ArrayList<>();
	    try (CqlSession session = CqlSession.builder()
	            .withConfigLoader(loader)
	            .build()) {

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

```
@Override
	public <T> List<T> findAllByColumnValues(DriverConfigLoader loader, Class<T> classType, Map<String, Object> columnValues) {
	    List<T> entities = new ArrayList<>();
	    try (CqlSession session = CqlSession.builder()
	            .withConfigLoader(loader)
	            .build()) {

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

	        String cql = String.format("SELECT * FROM %s WHERE %s", 
	                "member." + classType.getSimpleName().toLowerCase(), whereClause.toString());
	        cql += " ALLOW FILTERING";
	        System.out.println("[execute cql : ]" + cql);

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
	            
	            System.out.println(entity.toString());

	            entities.add(entity);
	        }

	    } catch (Exception e) {
	        // 오류 처리 로직
	        System.out.println("Error: " + e);
	    }

	    return entities;
	}
```



## test

```
SELECT * FROM member.interaction WHERE from_to = 'from' AND my_username = 'a@gmail.com' AND type = 'chatting' ALLOW FILTERING
```

|interaction_uuid|from_to|type|interaction_regdate|last_checked|my_username|opponent_username|type_uuid|
|---|---|---|---|---|---|---|---|



## project

```
<li class="d-flex justify-content-between mb-4">
<img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-6.webp" alt="avatar" class="rounded-circle d-flex align-self-start me-3 shadow-1-strong" width="60">
<div class="card">
  <div class="card-header d-flex justify-content-between p-3">
	<p class="fw-bold mb-0">\${chat.chat_chatter}</p>
	<p class="text-muted small mb-0"><i class="far fa-clock"></i> \${new Date(chat.chatted_at).toLocaleString()}</p>
  </div>
  <div class="card-body">
	<p class="mb-0">
	  \${chat.chat_content} 
	</p>
  </div>
</div>
</li>
```

```
<li>
<strong>\${chat.chat_chatter}</strong>: 
\${chat.chat_content} 
<small>(\${new Date(chat.chatted_at).toLocaleString()})</small>
</li>
```


lili

```
<li class="d-flex justify-content-between mb-4">
		            <img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-6.webp" alt="avatar" class="rounded-circle d-flex align-self-start me-3 shadow-1-strong" width="60">
		            <div class="card">
		              <div class="card-header d-flex justify-content-between p-3">
		                <p class="fw-bold mb-0">Brad Pitt</p>
		                <p class="text-muted small mb-0"><i class="far fa-clock"></i> 12 mins ago</p>
		              </div>
		              <div class="card-body">
		                <p class="mb-0">
		                  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
		                  labore et dolore magna aliqua.
		                </p>
		              </div>
		            </div>
		          </li>
```
```
<li class="d-flex justify-content-between mb-4">
		            <div class="card w-100">
		              <div class="card-header d-flex justify-content-between p-3">
		                <p class="fw-bold mb-0">Lara Croft</p>
		                <p class="text-muted small mb-0"><i class="far fa-clock"></i> 13 mins ago</p>
		              </div>
		              <div class="card-body">
		                <p class="mb-0">
		                  Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque
		                  laudantium.
		                </p>
		              </div>
		            </div>
		            <img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-5.webp" alt="avatar" class="rounded-circle d-flex align-self-start ms-3 shadow-1-strong" width="60">
		          </li>
```



