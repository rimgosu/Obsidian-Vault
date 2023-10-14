[[시스템로그]]
[기술블로그](https://m.blog.naver.com/chunsan89/221511250171)

## rsyslog 관련 파일
#### /etc/rc.d/init.d/rsyslog
- 시스템 시작 시 rsyslogd 데몬을 실행하는 스크립트
#### /etc/rsyslog.conf
- rsyslogd 데몬 환경설정 파일
- v3 이후 이거씀
###### /etc/sysconfig/rsyslog
- rsyslog 데몬 실행할 때 옵션 설정
- v3 이전에 이거 썼음 
#### /sbin/rsyslogd
- rsyslogd 데몬 파일 경로


## /etc/rsyslog.conf

#### HOW TO USE
- `*` : 전체
- 시설.수준;시설.수준 폴더
- 특정 파일만 선택 : =
- 특정 파일만 선택X : !=


**Action의 종류**

|**번호**|**Action**|**설명**|
|---|---|---|
|1|file|지정한 파일에 로그 기록|
|2|@host|지정한 호스트로 메시지 전달|
|3|user|지정한 사용자가 로그인 한 경우, 해당 사용자의 터미널로 전달|
|4|`*`| 현재 로그인 되어 있는 모든 사용자의 화면으로 전달|
|5|콘솔 또는 터미널|지정한 터미널로 메시지 전달|

---

**Facility의 종류**

|**번호**|**Facility**|**설명**|
|---|---|---|
|1|cron|cron, at과 같은 스케쥴링|
|2|auth, security|login과 같은 인증|
|3|authpriv|ssh와 같이 인증이 필요한 곳|
|4|daemon|telnet, ftp 등과 같은 데몬|
|5|kern|커널|
|6|lpr|프린트|
|7|mail|메일|
|8|mark|syslogd에 의해 만들어지는 날짜 유형|
|9|user|사용자 프로세스|

---

**Priority의 종류**

|**번호**|**Priority**|**설명**|
|---|---|---|
|1|none|지정한 facility를 제외|
|2|debug|프로그램 디버깅|
|3|info|통계, 기본 정보 메시지|
|4|notice|특별히 주의를 요하나, 에러는 아님|
|5|warning, warn|주의가 필요한 경고 메시지|
|6|error, err|에러 발생|
|7|crit|크게 급하지는 않지만 시스템에 문제가 생김|
|8|alert|즉각 조치 필요|
|9|emerg, panic|모든 사용자들에게 전달해야 할 위험한 상황|


### 기출 풀어보기
[[제1801회 리눅스마스터 1급 2차 기출문제]]

7. 다음은 시스템 로그 관련 설정을 하는 과정이다. 조건에 맞게 ( 괄호 ) 안에 알맞은 내용을 적으시오.

|   |
|---|
|# vi ①   <br>②  <br>③|

■ 조건  
- ①번은 관련 파일명을 절대 경로로 기재한다.  
- ②번은 ssh와 같은 인증을 필요로 하는 프로그램 유형이 발생한 메시지는 /var/log/sshlog에기록하고 info 수준의 로그는 제외한다.  
- ③번은 모든 facility가 발생하는 crit 수준의 메시지만 /var/log/critical 파일에 기록한다.  
- ②과 ③번은 조건과 관련된 내용 한 줄만을 기재한다.  

### 정답
1. /etc/rsyslog.conf
2. authpriv.`*`;uathpriv.!=info /var/log/sshlog
3. `*`.=crit /var/log/critical