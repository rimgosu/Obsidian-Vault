
### find

```
find [경로] [조건과 행동]
```

```
find . -name "*.txt"
```


- `/home/user` 디렉토리에서 2일 이내에 수정된 모든 파일 검색
```
find /home/user -mtime -2
```

- `/var/log` 디렉토리에서 1MB 이상의 파일 검색
```
find /var/log -size +1M
```

### which
- 실행 파일의 경로 보여줌
- `PATH` 환경 변수에 지정된 디렉토리들에서 해당 프로그램을 찾

```
which [프로그램명]
```


- `python`의 위치 찾기
```
which python
```

- `gcc` 컴파일러의 위치 찾기
```
which gcc
```



### type
- which와 거의 비슷한 역할을 함

```
type ls
```

![[Pasted image 20231013201221.png]]


# grep
- cat 파일 | grep 문자열
- -i 속성으로 대소문자 구별을 할 수 있음.
- cat 경로/httpd.conf | grep -i name