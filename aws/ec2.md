
## eclipse에서 maven build
- Goal
```
clean package
```

## 현재 public 주소

```
3.34.134.84
```

## ec2 접속하기

```
ssh -i C:/keys/ec2/amazon-linux-key.pem ec2-user@ec2-54-180-93-113.ap-northeast-2.compute.amazonaws.com
```

## ec2에 파일 

```
scp -i .\amazon-linux-key.pem "C:\WS\project9\simkoong\target\demo-0.0.20-SNAPSHOT.war" ec2-user@ec2-54-180-93-113.ap-northeast-2.compute.amazonaws.com:/home/ec2-user/
```

```
scp -i .\amazon-linux-key.pem -r "c:/keys" ec2-user@ec2-54-180-93-113.ap-northeast-2.compute.amazonaws.com:/home/ec2-user/
```

## spring 백그라운드에서 실행

```
nohup java -jar practice-0.0.1-SNAPSHOT.war &
```


## 웹에 접속

```
http://3.34.134.84:8081
```









---

## IMDSv2

```
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` \
&& curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/
```


```

Error: com.datastax.oss.driver.api.core.servererrors.InvalidQueryException: The update would cause the row to exceed the maximum allowed size
사진입력으로 들어왔음.
```