- 설정-절전-안함

# Apache Settings [[httpd.conf]]
#### 환경변수 등록하기
- 영구적으로 환경변수에 등록
- Apache는 지금 /usr/local/apache/에 설치되어있고, 실행 파일은 /usr/local/apache/bin에 모여있다. 
```
nano ~/.bashrc
export PATH=$PATH:/usr/local/apache/bin
source ~/.bashrc
```


# Samba Settings [[삼바(SAMBA)]]
- smb 설치
```
yum -y install bind
```

# squid [[squid]]
- squid 설치
```
yum -y install squid
```




