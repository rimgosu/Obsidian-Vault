[[11번]]
[[sendmail 주요 설정 파일 개요]]
[[메일 관련 서비스 개요]]

## /etc/aliases
- 메일의 별칭 혹은 특정 계정으로 수신한 이메일을 다른 게정으로 전달하도록 설정하며, 보통 여러 사람에게 전달할 때 사용된다
- `[수신계정]` `[전달계정]`의 형식을 따르며, :include:`[파일이름]`으로 사용자 이름이 저장된 파일을 설정할 수 있다
- sendmail이 참조하는 파일은 /etc/aliases.db이므로 /etc/aliases를 수정한 후 newaliases나 sendmail -bi 명령으로 적용한다.

```
webmaster: ihduser, kaituser
admin: :include:/etc/mail_admin
```


