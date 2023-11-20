
```
<form class="form-inline">
	<li class="bg-white mb-3" style="border-radius: 15px;">
	  <div class="form-outline" style="width: 100%;">
		<textarea class="form-control" id="textarea-value" rows="4"></textarea>
	  </div>
	</li>
	
	<button type="submit" id="send" class="btn btn-info btn-rounded float-end">Send</button>
</form>
```

```
<form class="form-inline">
	<div class="form-group">
		<label for="name">What is your name?</label>
		<input type="text" id="name" class="form-control" placeholder="Your name here...">
	</div>
	<button id="send" class="btn btn-default" type="submit">Send</button>
</form>
```



## 기존 코드 (textarea)

```
chattingListHtml += `
          	 <form class="form-inline">
             	<li class="bg-white mb-3" style="border-radius: 15px;">
	              <div class="form-outline" style="width: 100%;">
	                <textarea class="form-control" id="textarea-value" rows="4"></textarea>
	              </div>
	            </li>
             `;
```


```
[GreetingController][@MessageMapping("/hello")][chatting]Chatting(chat_uuid=null, room_uuid=null, chat_chatter=null, chatted_at=null, chat_content=1234, chat_emoticon=null, read_status=null)
```

```
SELECT chat_uuid, room_uuid, chat_chatter, chatted_at, chat_content, chat_emoticon, read_status FROM member.chatting order by room_uuid ALLOW FILTERING;
```
![[Pasted image 20231120191051.png]]


```
chattingListHtml += `
 <li class="d-flex justify-content-between mb-4">
	 <img src="https://mdbcdn.b-cdn.net/img/Photos/Avatars/avatar-6.webp" alt="avatar" class="rounded-circle d-flex align-self-start me-3 shadow-1-strong" width="60">
	 <div class="card">
		 <div class="card-header d-flex justify-content-between p-3">
			 <p class="fw-bold mb-0">\${chat.chat_chatter}</p>
			 <p class="text-muted small mb-0"><i class="far fa-clock"></i>\${new Date(chat.chatted_at).toLocaleString()}</p>
		 </div>
		 <div class="card-body">
			 <p class="mb-0">\${chat.chat_content}</p>
		 </div>
	 </div>
 </li>`;
```