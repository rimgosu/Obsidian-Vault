
[[13번]]

# /etc/samba/smb.conf
##### 외워야만 풀 수 있음 [[암기필요]]
- writable = Yes
- write list = `[사용자명]`
- valid users = `[사용자명]`
- public = no (개인 사용자만 사용할 수 있도록 설정)
- follow symlinks = no (심볼릭 링크를 따르지 않도록 설정하여 잠재적인 보안 위협을 제거한다)

##### 뭘 의미하는지 알아야함 [[암기필요]]
- browseable = No (이용 가능한 공유 리스트에 표시되지 않도록 설정)


