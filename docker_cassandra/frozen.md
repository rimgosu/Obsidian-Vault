

## 테이블 생성


``
```
CREATE TABLE IF NOT EXISTS example ( id UUID PRIMARY KEY, name TEXT, info frozen<MAP<TEXT, frozen<MAP<TEXT, MAP<TEXT, TEXT>>>>>, tags LIST<TEXT> );
```