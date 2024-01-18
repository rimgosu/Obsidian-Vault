#!/bin/bash
git pull origin master
git add .

# ChatGPT API에 요청을 보내는 함수
summarize_changes() {
    local file_content=$1
    local api_key=$2

    # JSON 데이터 형식으로 요청 본문을 준비
    local data="{\"prompt\": \"${file_content:0:4000}\", \"max_tokens\": 100, \"temperature\": 0.7}"

    # ChatGPT API를 사용하여 요청을 보내고 응답을 받음
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $api_key" \
        -d "$data" \
        "https://api.openai.com/v1/engines/davinci-codex/completions")

    # 응답에서 텍스트 내용을 추출
    echo $(echo $response)
}


# Git에서 변경된 파일 목록을 가져옴
changed_files=$(git status -s | awk '{if ($1 == "M" || $1 == "A") print $2}')

# 변경된 파일들의 내용을 저장
content=""
for file in $changed_files; do
    content+="$(cat $file)"
done

# ChatGPT API 키를 환경 변수에서 가져옴
api_key=$GPT_API_KEY

# 변경 내용을 요약
commit_message=$(summarize_changes "$content" "$api_key")

# 커밋
git commit -m "$commit_message"

# 푸시
git push origin master




