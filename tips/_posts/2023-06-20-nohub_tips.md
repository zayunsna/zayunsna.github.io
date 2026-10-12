---
lang: ko
layout: post
title: Nohup 사용법
description: >
  잊기전에 적어두자 nohup 사용법!
image: /assets/img/tips/nohub_tips/cover.png
lastmod: 2026-10-12
last_modified_at: 2026-10-12
sitemap:
  changefreq: daily
  priority: 1.0
---

# nohup 명령어 정리

**nohup은 리눅스나 유닉스 시스템에서 터미널 종료 시 전달되는 SIGHUP 신호를 무시하고 process를 실행하는 명령어.** 백그라운드로 실행하려면 명령어 끝에 `&`를 따로 붙여야 한다.

### 1. 기본 구조

```bash
# Run a shell script in the background
nohup bash run.sh &

# Run an application in the background
nohup uvicorn main:app --host=0.0.0.0 --port=8000 &
```

### 2. nohup에서 제공하는 log(info, error 등)을 저장하는 법

```bash
nohup bash run.sh > out_log.txt 2>&1 &
nohup uvicorn main:app --host=0.0.0.0 --port=8000 > system_log.log 2>&1 &
```

표준출력(stdout)은 `>`로 지정한 파일에 저장하고, 표준에러(stderr)는 `2>&1`로 같은 파일에 합쳐 저장하는 방식이다. `>`는 기존 내용을 덮어쓰고, 이어 쓰고 싶으면 `>>`를 사용하면 된다.

### 3. [번외] nohup을 이용해 background로 실행중인 process 찾아 종료하기

```bash
nohup bash run.sh > out_log.txt 2>&1 &
JOB_ID=$!
printf '%s\n' "$JOB_ID" > run.pid
```

실행 직후 `$!`를 저장하면 방금 백그라운드로 실행한 process의 PID를 알 수 있다. 이름에 `run`이 들어간 process를 전부 검색하는 것보다 종료할 대상을 정확하게 잡을 수 있다.

아래 코드를 `kill_proc.sh`로 저장하고, 기록한 PID를 argument로 전달하면 된다.

```bash
#!/bin/bash

if [ "$#" -ne 1 ]; then
    printf 'Usage: bash kill_proc.sh PID\n' >&2
    exit 1
fi

JOB_ID=$1
case "$JOB_ID" in
    ''|*[!0-9]*|0*|1)
        printf 'Expected a single PID greater than 1, without leading zeros.\n' >&2
        exit 1
        ;;
esac

if [ "${#JOB_ID}" -gt 10 ] || [ "$JOB_ID" -gt 2147483647 ]; then
    printf 'PID is outside the supported range.\n' >&2
    exit 1
fi

ps -p "$JOB_ID" -o pid=,args= || exit 1
read -r -p 'Stop this process? [y/N] ' ANSWER || exit 1
case "$ANSWER" in
    y|Y)
        kill "$JOB_ID" && printf 'Termination signal sent to PID %s.\n' "$JOB_ID"
        ;;
    *) printf 'Cancelled.\n' ;;
esac
```

```bash
bash kill_proc.sh "$(cat run.pid)"
```

원리는 간단하다. PID가 비어 있거나 숫자가 아니면 중단하고, `ps`로 해당 process의 명령어를 보여준 뒤 `y`를 입력한 경우에만 종료 신호를 보낸다. `kill`은 기본적으로 SIGTERM을 보내기 때문에 process가 정리 작업을 할 수 있고, 즉시 종료됐다는 뜻은 아니다.

PID는 process가 끝난 뒤 다른 process에 재사용될 수 있다. 오래된 `run.pid`를 그대로 믿지 말고, 화면에 나온 명령어가 내가 실행한 작업인지 꼭 확인하자. 여러 작업을 돌리면 PID 파일 이름도 각각 다르게 저장하는 편이 좋다.

또한 이 방법은 기록한 PID 하나에만 신호를 보낸다. `run.sh`가 별도의 자식 process를 만들었다면 그 자식까지 모두 종료되는 것은 아니다. 직접 Python 작업을 돌릴 거라면 `nohup python -u train.py > out_log.txt 2>&1 &`처럼 시작해 해당 PID를 기록하는 게 더 단순하다.

참고: [nohup](https://www.gnu.org/s/coreutils/manual/html_node/nohup-invocation.html), [Bash의 `$!`](https://www.gnu.org/software/bash/manual/html_node/Special-Parameters.html) — 2026-10-12 확인.
