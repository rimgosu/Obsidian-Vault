
[[1802 05 make]]

# 클린 타겟

```
cd /usr/src/linux
make help | grep clean
```

- make clean : 커널 환경설정을 제외하고 대부분의 파일을 모두 제거한다
- make mrproper : 커널 환경설정을 포함하여 모든 파일을 모두 제거한다
- make distclean : mrproper의 동작을 모두 수행하고, 백업 및 패치 파일도 모두 제거한다.

## 커널 환경설정
### 환경설정 확인하는 Tip

```
cd /usr/src/linux
make help | grep config
```

- make config : 텍스트 기반 환경설정 도구
- make menuconfig : 텍스트 기반 컬러 메뉴, 목록, 다이얼로그를 통한 환경설정 도구이다
- make nconfig : 컬러메뉴 환경설정
- make xconfig : X 윈도우 환경의 Qt 기반의 환경설정 도구
- make gconfig : X 윈도우 환경의 GTK+ 기반의 환경설정 도구이다. GTK+2.0 이상의 설치 필요