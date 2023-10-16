[[15번]]


1. iptables 규칙 확인
```
iptables -L
```

| 옵션 | 설명                                      |
|------|------------------------------------------|
| -A   | 새로운 정책을 가장 마지막에 추가/등록한다 |
| -I   | 지정한 라인 번호에 추가한다 <br>**2번라인에 넣으면 2번~끝 까지 적용됨**              |
| -D   | 정책을 제거한다                          |
| -R   | 정책을 수정한다                          |








---
# GPT해설

`iptables`는 리눅스 기반 시스템에서 네트워크 패킷을 필터링하고 NAT(네트워크 주소 변환)을 수행하는 사용자 공간 유틸리티로, `netfilter` 프레임워크의 일부입니다. 여기서는 `iptables`의 주요 특징, 명령어 및 사용법에 대해 설명합니다.

### 1. iptables의 기본 구조:

1. **테이블(Table)**: 특정 작업을 수행하기 위한 규칙 집합
2. **체인(Chain)**: 규칙을 순서대로 나열한 것
3. **규칙(Rule)**: 패킷에 적용될 작업을 정의하는 것

기본적으로 존재하는 테이블:

- **Filter**: 기본 테이블로 패킷 필터링을 담당
- **NAT**: 네트워크 주소 변환(NAT)을 담당
- **Mangle**: 패킷 헤더 정보를 변경
- **Raw**: 연결 추적을 무시하도록 설정

기본적으로 존재하는 체인:

- **INPUT**: 호스트로 들어오는 패킷을 처리
- **FORWARD**: 호스트를 통과하는 패킷을 처리
- **OUTPUT**: 호스트에서 나가는 패킷을 처리
- **PREROUTING**: 라우팅 전에 패킷을 처리
- **POSTROUTING**: 라우팅 후 패킷을 처리

### 2. iptables 명령어:

1. **규칙 확인**: `iptables -L -n -v`
2. **특정 테이블의 규칙 확인**: `iptables -t [table-name] -L -n -v`
3. **규칙 추가**: `iptables -A [chain-name] [criteria] -j [target]`
4. **규칙 삭제**: `iptables -D [chain-name] [rule-number]`
5. **모든 규칙 삭제**: `iptables -F`
6. **체인의 기본 정책 설정**: `iptables -P [chain-name] [target]`
7. **규칙 저장**: `service iptables save` 또는 `iptables-save > /path/to/file`

### 3. 예제:

1. 모든 입력 패킷 차단: `iptables -P INPUT DROP`
2. SSH 트래픽 허용: `iptables -A INPUT -p tcp --dport 22 -j ACCEPT`
3. 특정 IP에서의 입력 트래픽 차단: `iptables -A INPUT -s [source-ip] -j DROP`

### 4. 주의사항:

1. `iptables` 규칙을 설정할 때 주의가 필요합니다. 잘못 설정하면 원격 접근이 불가능해질 수 있습니다.
2. 규칙 변경 후에는 꼭 `iptables`를 저장하여 리부트 후에도 변경 사항이 유지되도록 해야 합니다.

이는 `iptables`의 기본 개요에 불과합니다. 실제로 많은 옵션과 더 복잡한 설정이 가능하므로, 공부하거나 실무에서 사용할 때는 공식 문서나 자세한 가이드를 참고하는 것이 좋습니다.