[[메일 관련 서비스 개요]]
## /etc/mail/sendmail.cf
## /etc/mail/sendmail.mc
# /etc/aliases [[aliases]]
- 계정 => 계정
# /etc/mail/access [[access]]
# /etc/mail/virtusertable [[virtusertable]]
- 가상메일 => 계정
## /etc/mail/local-host-names [[local-host-names]]
## ~/.forward





1. **/etc/aliases**:
    
    - 이 파일은 메일 별칭을 정의합니다. 메일 별칭은 한 사용자의 이메일 주소로 들어오는 메일을 다른 주소나 다수의 주소로 전달하도록 설정하는 방법입니다.
    - 예를 들어, `webmaster: john.doe`라는 항목이 있다면 `webmaster`로 보내진 모든 메일은 `john.doe` 사용자에게 전달됩니다.
    - 이 파일을 수정한 후에는 `newaliases` 또는 `sendmail -bi` 명령을 사용하여 변경 사항을 적용해야 합니다.
2. **/etc/mail/access**:
    
    - 이 파일은 메일의 송신과 수신을 제어합니다.
    - IP 주소, 도메인, 이메일 주소 등을 기반으로 메일을 받아들이거나 거부할 수 있습니다.
    - 예: `spam.com REJECT`는 spam.com 도메인에서 보낸 모든 메일을 거부합니다.
3. **/etc/mail/virtusertable**:
    
    - 가상 이메일 주소와 실제 이메일 주소 또는 계정 간의 매핑을 설정하는 파일입니다.
    - 예를 들어, `info@mydomain.com john.doe`라는 항목이 있다면 `info@mydomain.com`로 보내진 모든 메일은 `john.doe` 사용자에게 전달됩니다.
4. **/etc/mail/local-host-names**:
    
    - 이 파일에는 Sendmail이 수신할 수 있는 도메인 이름이 나열되어 있습니다.
    - 서버가 여러 도메인을 처리하는 경우, 해당 도메인의 모든 이름을 이 파일에 추가해야 합니다.