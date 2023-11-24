## demo

```
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-devtools</artifactId>
	<scope>runtime</scope>
	<optional>true</optional>
</dependency>
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-tomcat</artifactId>
	<scope>provided</scope>
</dependency>
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-test</artifactId>
	<scope>test</scope>
</dependency>

<dependency>
	<groupId>org.glassfish.web</groupId>
	<artifactId>jakarta.servlet.jsp.jstl</artifactId>
</dependency>
```

## finalproject_web

```
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-web</artifactId>
</dependency>

<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-devtools</artifactId>
	<scope>runtime</scope>
	<optional>true</optional>
</dependency>

<dependency>
	<groupId>org.projectlombok</groupId>
	<artifactId>lombok</artifactId>
	<scope>provided</scope>
</dependency>

<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-test</artifactId>
	<scope>test</scope>
</dependency>


<dependency>
	<groupId>org.apache.tomcat.embed</groupId>
	<artifactId>tomcat-embed-jasper</artifactId>
</dependency>

<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-websocket</artifactId>
</dependency>
<dependency>
	<groupId>com.datastax.oss</groupId>
	<artifactId>java-driver-core</artifactId>
</dependency>

<dependency>
	<groupId>com.datastax.oss</groupId>
	<artifactId>java-driver-query-builder</artifactId>
</dependency>

<dependency>
	<groupId>com.datastax.oss</groupId>
	<artifactId>java-driver-mapper-runtime</artifactId>
</dependency>

<!-- 사진 추가 위한 의존성 -->
<!-- https://mvnrepository.com/artifact/servlets.com/cos -->
<dependency>
	<groupId>servlets.com</groupId>
	<artifactId>cos</artifactId>
	<version>05Nov2002</version>
</dependency>
<dependency>
	<groupId>jakarta.servlet.jsp.jstl</groupId>
	<artifactId>jakarta.servlet.jsp.jstl-api</artifactId>
</dependency>
<dependency>
	<groupId>org.glassfish.web</groupId>
	<artifactId>jakarta.servlet.jsp.jstl</artifactId>
</dependency>

<dependency>
	<groupId>javax.servlet</groupId>
	<artifactId>jstl</artifactId>
	<version>1.2</version>
</dependency>
<!--aws관련 의존성-->
<dependency>
	<groupId>software.amazon.awssdk</groupId>
	<artifactId>aws-sdk-java</artifactId>
	<version>2.21.23</version>
</dependency>

<dependency>
	<groupId>software.amazon.awssdk</groupId>
	<artifactId>bom</artifactId>
	<version>2.21.26</version>
	<type>pom</type>
	<scope>import</scope>
</dependency>


<dependency>
	<groupId>com.amazonaws</groupId>
	<artifactId>aws-java-sdk</artifactId>
	<version>1.11.1000</version>
</dependency>

<dependency>
	<groupId>javax.xml.bind</groupId>
	<artifactId>jaxb-api</artifactId>
	<version>2.3.1</version>
</dependency>

<!-- https://mvnrepository.com/artifact/com.amazonaws/aws-java-sdk-s3 -->
<dependency>
	<groupId>com.amazonaws</groupId>
	<artifactId>aws-java-sdk-s3</artifactId>
	<version>1.12.592</version>
</dependency>
<!-- Spring Cloud AWS -->

<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-tomcat</artifactId>
	<scope>provided</scope>
</dependency>
```