[[14번]]

### DNS 설정 파일


```
man named.conf
```

다음 명령어로 named.conf 의 설정 확인할 수 있다.



`named.conf`는 BIND (Berkeley Internet Name Domain) DNS 서버의 주요 설정 파일입니다. 리눅스에서 DNS 서버를 설정 및 운영하기 위해 반드시 알아야 하는 내용들을 간략히 정리해 보겠습니다.

1. **Configuration Clauses**:
    
    - `options`: 전역적인 DNS 서버 설정을 포함합니다.
    - `zone`: 특정 도메인에 대한 정보를 포함하며, 각 도메인마다 설정이 필요합니다.
    - `logging`: 로깅에 관한 설정을 포함합니다.
2. **Options Clause**:
    
    - `directory`: zone 파일이 위치한 디렉터리를 지정합니다.
    - `allow-query`: DNS 쿼리를 허용할 IP 주소나 네트워크를 지정합니다.
    - `forwarders`: 다른 DNS 서버로 쿼리를 전달할 주소를 지정합니다.
    - `forward`: 포워딩 방식을 설정합니다. (예: `first`, `only`)
    - **forward only** : 짬처리
    - **forward first** : 응답이 없을 경우 자신이 응답하도록 설정
    - `listen-on`: DNS 서버가 리스닝할 IP 주소를 지정합니다.
    - `version`: 버전 정보를 숨기거나 사용자 정의 문자열로 바꿀 수 있습니다.
3. **Zone Clause**:
    
    - `type`: zone의 타입을 지정합니다. (예: `master`, `slave`, `hint`, `forward`)
    - `file`: 해당 zone의 데이터 파일 이름을 지정합니다.
    - `allow-transfer`: Zone transfer를 허용할 IP 주소나 네트워크를 지정합니다.
4. **Logging Clause**:
    
    - 로그의 위치, 서버에서 발생하는 이벤트 유형, 로깅 레벨 등을 지정할 수 있습니다.
5. **Security**:
    
    - DNS amplification 공격 등을 방지하기 위한 설정이 필요하며, `rate-limit`, `allow-recursion`, `blackhole` 등의 옵션을 활용할 수 있습니다.
6. **Other Directives**:
    
    - `include`: 다른 설정 파일을 포함시킬 수 있습니다.
    - `controls`: `rndc`와 같은 관리 도구와 통신하기 위한 설정을 포함합니다.

이 외에도 많은 상세 설정이 있지만, 위 항목들은 기본적으로 알아두어야 할 주요 내용들입니다. 실제 시험에서는 각 설정 항목의 정확한 문법, 옵션의 의미 및 사용법에 대한 이해가 요구될 수 있습니다.