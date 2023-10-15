[[메일 관련 서비스 개요]]
[[sendmail 주요 설정 파일 개요]]

# /etc/mail/virtusertable
- 가상의 메일 계정으로 들어오는 메일을 특정 계정으로 전달하는 정보를 설정한다
- `makemap hash /etc/virtusertable < /etc/mail/virtusertable`와 같은 명령으로 `/etc/mail/virtusertable.db`에 적용된다.

```
info@foo.com               foo-info
info@bar.com               bar-info
```

