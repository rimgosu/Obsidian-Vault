
[[1802 09 접근제어]]
### sshd
- 설정 파일 : /etc/sshd/sshd_config



## 설정 값들
- Port : ssh 기본 포트 22 변경
- ListenAddress : ssh 서버에서 listen 할 localhost 주소 
	- 0.0.0.0 : 모든 네트워크를 의미함
- PermitRootLogin : no로 설정하여 root 계정의 접근을 금지한다
- PrintLastLog : 로그인 시 지난번 로그인 기록을 보여준다
- LoginGraceTime : 지정한 시간 안에 로그인하지 않으면 자동으로 접속을 해제한다
- X11Forwarding : 원격에서 X11 포워딩을 허용할지 설정한다
- PrintMotd : yes로 설정하면 /etc/motd 파일의 환영 메시지 등의 정보를 로그인 시 출력한다
- TCPKeepAlive : yes를 지정하면 일정 시간 간격으로 메시지를 전달하여 클라이언트의 접속 유지 여부를 확인한다.

