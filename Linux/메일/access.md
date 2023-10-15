[[메일 관련 서비스 개요]]
[[sendmail 주요 설정 파일 개요]]

# /etc/mail/access
- 메일 서버에 접속하는 호스트의 접근을 제어하는 설정 파일로 스팸 메일 방지 등에 사용할 수 있다
- 설정 방법은 `[정책대상]` `[정책]`의 형식을  사용한다.
	- 정책대상은 `도메인명, IP, 메일주소`를 사용
	- 정책은 `릴레이허용(RELAY), 거부(DENY), 거부후메시지전송(REJECT), DNS조회 실패 시에도 허용(OK)`을 지정할 수 있다
	- `makemap hash /etc/mail/access < /etc/mail/access`와 같은 명령으로 `/etc/mail/access.db`에 적용한다
```
Connect:127.0.0.1              OK
From:abnormal@young.com        REJECT
To:young.com                   RELAY
```