


`rpm`은 Red Hat Package Manager의 약자로, RPM 기반의 Linux 배포판(예: Red Hat, Fedora, CentOS 등)에서 패키지를 관리하는 데 사용되는 명령어입니다.
주요 rpm 명령어 옵션들은 다음과 같습니다:
1. **패키지 설치**
    - `-i` 또는 `--install`: 패키지를 설치합니다.
        `rpm -i package_name.rpm`
        
1. **패키지 삭제**
    - `-e` 또는 `--erase`: 패키지를 삭제합니다.
        `rpm -e package_name`
        
3. **패키지 업그레이드**
    - `-U` 또는 `--upgrade`: 패키지를 업그레이드합니다. 해당 패키지가 설치되어 있지 않으면 새로 설치합니다.
        `rpm -U package_name.rpm`
        
4. **패키지 질의**
    - `-q` 또는 `--query`: 설치된 패키지에 대한 정보를 조회합니다.
        - `rpm -q package_name`: 특정 패키지의 설치 상태를 확인합니다.
        - `rpm -qa`: 모든 설치된 패키지를 나열합니다.
        - `rpm -qi package_name`: 특정 패키지의 상세 정보를 확인합니다.
5. **패키지 검증**
    - `-V` 또는 `--verify`: 설치된 패키지를 검증합니다. 파일들이 변경되었는지, 손상되었는지 등을 확인합니다.
        `rpm -V package_name`
        
6. **패키지의 의존성 확인**
    - `--requires`: 패키지가 요구하는 의존성을 표시합니다.
        `rpm -q --requires package_name`
        
7. **rpm 파일의 정보 확인**
    - `-qp`: rpm 파일의 정보를 표시합니다.
        - `rpm -qpi package_name.rpm`: rpm 파일의 설명을 확인합니다.
        - `rpm -qpl package_name.rpm`: rpm 파일에 포함된 파일 목록을 확인합니다.
        - 
8. **기타 옵션**
    - `--import`: RPM 패키지 서명의 공개 키를 가져옵니다.
    - `-K` 또는 `--checksig`: RPM 패키지의 서명을 확인합니다.




### [[제1801회 리눅스마스터 1급 2차 기출문제]]

1. 패키지의 위치 찾기

```
which cat
```

2. 찾은 위치로 패키지 의존성 체크
```
rpm -qf /usr/bin/cat
```

3. i, l 로 자세한 정보 출력 및 설치한 패키지 정보 확인
```
rpm -qif /usr/bin/cat
rpm -qlf /usr/bin/cat
```