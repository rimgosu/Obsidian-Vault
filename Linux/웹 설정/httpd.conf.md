
[[12번]]

### /etc/httpd/conf/httpd.conf

| 설정                                  | 항목           | 설정 값                      |
|---------------------------------------|----------------|------------------------------|
| 포트 번호 지정                        | Listen         | 80                           |
| 도메인명 지정                         | ServerName     | www.ihd.or.kr                |
| 최상위 디렉터리 지정 (웹서버의 설치 디렉터리 지정)  | ServerRoot     | "/usr/local/apache"          |
| 웹문서 위치 지정 (서버의 인덱스 페이지가 위치하는 곳) | DocumentRoot   | "/www"                       |
| 관리자 이메일 주소 지정               | ServerAdmin    | webadmin@example.com         |
| 인덱스 파일 지정                     | DirectoryIndex | index.php index.html index.htm|
| 일반 사용자 홈 디렉터리 지정          | UserDir        | www                          |

- ServerTokens : HTTP 응답헤더에 포함하여 전송할 서버의 정보 수준으로, 보안을 위해 최소 정보밤ㄴ 사용하도록 prod로 설정하는 것을 권장한다
- KeepAlive : On으로 설정하면 아파치의 한 프로세스로 특정 사용자의 지속적인 요청 작업을 계속 처리한다.

## how to solve
- httpd.conf 파일에서 원하는 설정의 이름을 외워두고 풀면 된다.
1. find / -name `*httpd.conf*`
2. cat 경로/httpd.conf | grep -i direcory 와 같이 검색해서 기본값이 어떻게 설정되어있는지 확인하고 그 양식에 맞게 고쳐주면됨.
- 단, UserDir은 없어서 그냥 외워야함 UserDir www 이런식으로 쓰면 된다.
- 관리자 이메일 주소 지정은 ServerAdmin이다. 이메일 어쩌고로 검색하면 안된다.

ServerRoot
- 아파치 웹 서버의 주요 파일들이 저장된 최상위 디렉터리를 절대 경로로 지정한다
- 기본값 : ServerRoot = "/etc/httpd"

Listen
- 아파치 웹 서버가 사용할 TCP 포트 번호를 지정한다
- 기본값 : Listen 80

LoadModule
- DSO 방식으로 로딩할 모듈을 지정한다
- Include conf.d/`*`.conf와 같이 지정하여, 모듈을 로딩하기 위한 별도의 설정 파일을 포함하여 사용할 수 있다
```
LoadModule php5_module modules/libphp5.so
```

User
- 아파치 실행 데몬의 사용자 권한을 지정한다
- 시스템 보안을 위하여 root로 설정하지 않도록 한다
- 기본값 : User appache

Group
- 아파치 실행 데몬의 그릅 권한을 지정한다
- 시스템 보안을 위하여 root로 설정하지 않도록 한다
- 기본값 : Group apache

ServerAdmin
- 아파치 웹 서버 관리자의 이메일 주소를 지정하여, 서버에 문제가 발생할 경우 에러메시지에 함께 표시한다.
- 기본값 : ServerAdmin root@localhost

ServerName
- 서버의 호스트 이름을 지정한다
- 기본값 : ServerName www.example.com:80
- 기본값이 주석으로 처리되어 있으므로, 적절한 값으로 변경하여 적용한다

DocumnetRoot
- 웹 문서가 저장되는 기본 디렉터리 경로를 지정한다
- 기본값 : DocumnetRoot "/var/www/html"

DirectoryIndex
- 웹 디렉터리를 방문할 경우 처음으로 열릴 파일 목록을 정의한다
- 기본값: AccessFileName .htaccess
...

