
## ec2 log
```
[ec2-user@ip-172-31-32-36 ~]$ java -jar demo-0.0.3-SNAPSHOT.war
Standard Commons Logging discovery in action with spring-jcl: please remove commons-logging.jar from classpath in order to avoid potential conflicts

  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.1.6)

2023-11-24T05:29:28.330Z  INFO 8670 --- [           main] com.example.DemoApplication              : Starting DemoApplication v0.0.3-SNAPSHOT using Java 21.0.1 with PID 8670 (/home/ec2-user/demo-0.0.3-SNAPSHOT.war started by ec2-user in /home/ec2-user)
2023-11-24T05:29:28.335Z  INFO 8670 --- [           main] com.example.DemoApplication              : No active profile set, falling back to 1 default profile: "default"
2023-11-24T05:29:32.986Z  INFO 8670 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat initialized with port(s): 8081 (http)
2023-11-24T05:29:33.032Z  INFO 8670 --- [           main] o.apache.catalina.core.StandardService   : Starting service [Tomcat]
2023-11-24T05:29:33.033Z  INFO 8670 --- [           main] o.apache.catalina.core.StandardEngine    : Starting Servlet engine: [Apache Tomcat/10.1.16]
2023-11-24T05:29:58.360Z  INFO 8670 --- [           main] org.apache.jasper.servlet.TldScanner     : At least one JAR was scanned for TLDs yet contained no TLDs. Enable debug logging for this logger for a complete list of JARs that were scanned but no TLDs were found in them. Skipping unneeded JARs during scanning can improve startup time and JSP compilation time.
2023-11-24T05:29:59.211Z  INFO 8670 --- [           main] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring embedded WebApplicationContext
2023-11-24T05:29:59.213Z  INFO 8670 --- [           main] w.s.c.ServletWebServerApplicationContext : Root WebApplicationContext: initialization completed in 30687 ms
Standard Commons Logging discovery in action with spring-jcl: please remove commons-logging.jar from classpath in order to avoid potential conflicts
2023-11-24T05:30:03.790Z  INFO 8670 --- [           main] o.s.b.a.w.s.WelcomePageHandlerMapping    : Adding welcome page template: index
2023-11-24T05:30:04.873Z  INFO 8670 --- [           main] o.s.b.w.embedded.tomcat.TomcatWebServer  : Tomcat started on port(s): 8081 (http) with context path ''
2023-11-24T05:30:04.876Z  INFO 8670 --- [           main] o.s.m.s.b.SimpleBrokerMessageHandler     : Starting...
2023-11-24T05:30:04.876Z  INFO 8670 --- [           main] o.s.m.s.b.SimpleBrokerMessageHandler     : BrokerAvailabilityEvent[available=true, SimpleBrokerMessageHandler [org.springframework.messaging.simp.broker.DefaultSubscriptionRegistry@75961f16]]
2023-11-24T05:30:04.877Z  INFO 8670 --- [           main] o.s.m.s.b.SimpleBrokerMessageHandler     : Started.
2023-11-24T05:30:04.910Z  INFO 8670 --- [           main] com.example.DemoApplication              : Started DemoApplication in 38.697 seconds (process running for 42.465)
2023-11-24T05:30:15.281Z  INFO 8670 --- [nio-8081-exec-1] o.a.c.c.C.[Tomcat].[localhost].[/]       : Initializing Spring DispatcherServlet 'dispatcherServlet'
2023-11-24T05:30:15.282Z  INFO 8670 --- [nio-8081-exec-1] o.s.web.servlet.DispatcherServlet        : Initializing Servlet 'dispatcherServlet'
2023-11-24T05:30:15.283Z  INFO 8670 --- [nio-8081-exec-1] o.s.web.servlet.DispatcherServlet        : Completed initialization in 1 ms
main으로 들어왔음.
2023-11-24T05:31:03.055Z  INFO 8670 --- [MessageBroker-1] o.s.w.s.c.WebSocketMessageBrokerStats    : WebSocketSession[0 current WS(0)-HttpStream(0)-HttpPoll(0), 0 total, 0 closed abnormally (0 connect failure, 0 send limit, 0 transport error)], stompSubProtocol[processed CONNECT(0)-CONNECTED(0)-DISCONNECT(0)], stompBrokerRelay[null], inboundChannel[pool size = 0, active threads = 0, queued tasks = 0, completed tasks = 0], outboundChannel[pool size = 0, active threads = 0, queued tasks = 0, completed tasks = 0], sockJsScheduler[pool size = 1, active threads = 1, queued tasks = 0, completed tasks = 0]
```

## local log

```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
[32m :: Spring Boot :: [39m              [2m (v3.1.6)[0;39m

[2m2023-11-24T14:24:40.442+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mcom.example.DemoApplication             [0;39m [2m:[0;39m Starting DemoApplication using Java 17.0.8 with PID 8072 (C:\WS\project8\demo\target\classes started by newny in C:\WS\project8\demo)
[2m2023-11-24T14:24:40.442+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mcom.example.DemoApplication             [0;39m [2m:[0;39m No active profile set, falling back to 1 default profile: "default"
[2m2023-11-24T14:24:40.845+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.b.w.embedded.tomcat.TomcatWebServer [0;39m [2m:[0;39m Tomcat initialized with port(s): 8081 (http)
[2m2023-11-24T14:24:40.845+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.apache.catalina.core.StandardService  [0;39m [2m:[0;39m Starting service [Tomcat]
[2m2023-11-24T14:24:40.846+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.apache.catalina.core.StandardEngine   [0;39m [2m:[0;39m Starting Servlet engine: [Apache Tomcat/10.1.16]
[2m2023-11-24T14:24:41.124+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36morg.apache.jasper.servlet.TldScanner    [0;39m [2m:[0;39m At least one JAR was scanned for TLDs yet contained no TLDs. Enable debug logging for this logger for a complete list of JARs that were scanned but no TLDs were found in them. Skipping unneeded JARs during scanning can improve startup time and JSP compilation time.
[2m2023-11-24T14:24:41.126+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.a.c.c.C.[Tomcat].[localhost].[/]      [0;39m [2m:[0;39m Initializing Spring embedded WebApplicationContext
[2m2023-11-24T14:24:41.126+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mw.s.c.ServletWebServerApplicationContext[0;39m [2m:[0;39m Root WebApplicationContext: initialization completed in 680 ms
[2m2023-11-24T14:24:41.204+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.b.a.w.s.WelcomePageHandlerMapping   [0;39m [2m:[0;39m Adding welcome page template: index
[2m2023-11-24T14:24:41.238+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.b.d.a.OptionalLiveReloadServer      [0;39m [2m:[0;39m LiveReload server is running on port 35729
[2m2023-11-24T14:24:41.248+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.b.w.embedded.tomcat.TomcatWebServer [0;39m [2m:[0;39m Tomcat started on port(s): 8081 (http) with context path ''
[2m2023-11-24T14:24:41.249+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.m.s.b.SimpleBrokerMessageHandler    [0;39m [2m:[0;39m Starting...
[2m2023-11-24T14:24:41.249+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.m.s.b.SimpleBrokerMessageHandler    [0;39m [2m:[0;39m BrokerAvailabilityEvent[available=true, SimpleBrokerMessageHandler [org.springframework.messaging.simp.broker.DefaultSubscriptionRegistry@77a387b6]]
[2m2023-11-24T14:24:41.249+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mo.s.m.s.b.SimpleBrokerMessageHandler    [0;39m [2m:[0;39m Started.
[2m2023-11-24T14:24:41.252+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36mcom.example.DemoApplication             [0;39m [2m:[0;39m Started DemoApplication in 0.835 seconds (process running for 408.655)
[2m2023-11-24T14:24:41.253+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[  restartedMain][0;39m [36m.ConditionEvaluationDeltaLoggingListener[0;39m [2m:[0;39m Condition evaluation unchanged
[2m2023-11-24T14:25:41.180+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[MessageBroker-1][0;39m [36mo.s.w.s.c.WebSocketMessageBrokerStats   [0;39m [2m:[0;39m WebSocketSession[0 current WS(0)-HttpStream(0)-HttpPoll(0), 0 total, 0 closed abnormally (0 connect failure, 0 send limit, 0 transport error)], stompSubProtocol[processed CONNECT(0)-CONNECTED(0)-DISCONNECT(0)], stompBrokerRelay[null], inboundChannel[pool size = 0, active threads = 0, queued tasks = 0, completed tasks = 0], outboundChannel[pool size = 0, active threads = 0, queued tasks = 0, completed tasks = 0], sockJsScheduler[pool size = 1, active threads = 1, queued tasks = 0, completed tasks = 0]
[2m2023-11-24T14:32:50.001+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[nio-8081-exec-1][0;39m [36mo.a.c.c.C.[Tomcat].[localhost].[/]      [0;39m [2m:[0;39m Initializing Spring DispatcherServlet 'dispatcherServlet'
[2m2023-11-24T14:32:50.001+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[nio-8081-exec-1][0;39m [36mo.s.web.servlet.DispatcherServlet       [0;39m [2m:[0;39m Initializing Servlet 'dispatcherServlet'
[2m2023-11-24T14:32:50.002+09:00[0;39m [32m INFO[0;39m [35m8072[0;39m [2m---[0;39m [2m[nio-8081-exec-1][0;39m [36mo.s.web.servlet.DispatcherServlet       [0;39m [2m:[0;39m Completed initialization in 1 ms

```