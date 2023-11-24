
## 1. docker desktop 실행
## 2. Docker 파일 작성

```
# 베이스 이미지 지정 (Java 17)
FROM openjdk:17-oracle

# JAR 파일 복사
COPY target/demo-0.0.3-SNAPSHOT.war myapp.war

# 애플리케이션 실행
ENTRYPOINT ["java","-jar","/myapp.war"]
```

## 3. docker 이미지 빌드

```
docker build -t myapp:latest .
```

## 4. docker 이미지 실행

```
docker run -p 8081:8081 -v c:/keys:/keys myapp:latest
```