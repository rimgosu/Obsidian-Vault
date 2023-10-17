[[1802 10 cpio]]
## cpio
- 파일 아카이브 만들 때 사용


# 옵션
- -c : SVR4 portable format with no CRC
- -v : 자세한 정보 출력
- -i : 압축 해제모드
- -o : 압축 모드

# examples
### 1.
```
find /home | cpio -ocv > home.cpio
```

### 2.
```
cpio -icv < home.cpio
```


