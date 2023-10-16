[[우선순위변경]]

# [[pgrep]]과의 차이
- pgrep은 딱 pid만 보여주는 반면,
- ps는 자세한 정보를 출력한다.

```
pgrep -u ihduser
```
![[Pasted image 20231016212007.png]]

```
ps -u ihduser
```
![[Pasted image 20231016211954.png]]

### ps aux
- 전체 프로세스 확인

# ps aux | grep 프로세스
- 원하는 프로세스 확인

```
ps aux | grep syslog
```
![[Pasted image 20231013214720.png]]
syslog 로그만 확인할 수 있음.



