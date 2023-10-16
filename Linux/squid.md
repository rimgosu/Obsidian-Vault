
### 1. Squid: 개요

- **정의**: Squid는 고성능, 사용자 친화적인 프록시 서버로, 웹 콘텐츠를 캐시하며 클라이언트의 웹 요청을 대신하여 인터넷에서 데이터를 가져옵니다.

### 2. 설정 및 관리 (squid.conf)

- Squid의 주 설정 파일은 일반적으로 `/etc/squid/squid.conf`에 위치합니다.
    
- 주요 설정 옵션에는 포트 번호, 캐시 크기, ACL 정의 및 다른 관련 설정이 포함됩니다.
    
- Squid의 상태 및 성능은 `squidclient` 유틸리티를 사용하여 모니터링할 수 있습니다.


---
### GPT 설명 squid

### 1. **기본 설정**

- **http_port**: Squid 서버가 리스닝할 포트를 지정합니다. 기본값은 3128입니다.
    
    
    `http_port 3128`
    
- **cache_dir**: 디스크 캐시 저장소를 지정합니다. 기본적인 설정은 `ufs` (UNIX 파일 시스템)을 사용하는데, 다음과 같이 설정합니다.
    
    
    `cache_dir ufs /var/spool/squid 100 16 256`
    
    여기서 `100`은 캐시 크기(MB), `16`은 디렉토리의 수, `256`은 각 디렉토리 당 하위 디렉토리의 수를 나타냅니다.
    

### 2. **액세스 제어 (ACL)**
- 별칭을 정의하고, 액션합니다.78

ACL을 사용하면 특정 요청에 대해 어떤 동작을 할지 결정할 수 있습니다.

- **ACL 정의**: `acl` 키워드를 사용하여 규칙을 정의합니다.
    
    `acl localnet src 192.168.0.0/16  # localnet라는 이름의 규칙을 정의, 192.168.0.0/16 대역의 IP 주소로부터 오는 요청을 나타냅니다.`
    
- **ACL 액션**: 정의한 규칙에 액션을 적용합니다.
    
    
    `http_access allow localnet      # localnet에서의 요청을 허용 http_access deny all            # 그 외 모든 요청은 거부`
    

### 3. **로그 설정**

- **access_log**: 액세스 로그의 위치와 포맷을 지정합니다.
    
    bashCopy code
    
    `access_log /var/log/squid/access.log squid`
    

### 4. **기타 설정**

- **refresh_pattern**: 캐시된 객체의 유효기간을 설정합니다.
    
    `refresh_pattern ^ftp:           1440    20%     10080 refresh_pattern ^gopher:        1440    0%      1440 refresh_pattern -i (/cgi-bin/|\?) 0     0%      0 refresh_pattern .               0       20%     4320`
    

### 5. **squidclient**

Squid의 상태와 성능을 모니터링하려면 `squidclient` 유틸리티를 사용합니다.

`squidclient mgr:info`