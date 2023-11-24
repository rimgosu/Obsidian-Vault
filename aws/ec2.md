
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
ssh -i C:/keys/ec2/amazon-linux-key.pem ec2-user@ec2-54-180-116-156.ap-northeast-2.compute.amazonaws.com
```

## ec2에 파일 업로드

```
scp -i .\amazon-linux-key.pem "C:\WS\project8\demo\target\demo-0.0.3-SNAPSHOT.jar" ec2-user@ec2-54-180-93-113.ap-northeast-2.compute.amazonaws.com:/home/ec2-user/
```

```
scp -i .\amazon-linux-key.pem -r "c:/keys" ec2-user@ec2-43-200-254-135.ap-northeast-2.compute.amazonaws.com:/home/ec2-user/
```

## spring 백그라운드에서 실행

```
nohup java -jar practice-0.0.1-SNAPSHOT.war &
```


## 웹에 접속

```
http://3.34.134.84:8081
```