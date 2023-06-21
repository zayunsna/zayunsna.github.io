---
layout: post
title: Nohub 사용법
description: >
  잊기전에 적어두자 nohub 사용법!
image: /assets/img/tips/nohub_tips/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# nohup 명령어 정리

**nohub은 리눅스나 유닉스 시스템에서 백그라운드로 프로세스를 실행시키고, 해당 터미널이 종료된 후에도 그 프로세스가 계속 실행되게 하는 스크립트.**

### 1. 기본 구조

```bash
$ nohup <process_name>
   ex1) nohup run.sh
   ex2) nohup uvicorn main:app --reload --host=0.0.0.0 --port=8000
```

### 2. nohup에서 제공하는 log(info, error 등)을 저장하는 법

```bash
$ nohup <process_name> > <log_file_name> 2>&1 &
   ex1) nohup run.sh > out_log.txt 2>&1 &
   ex2) nohup uvicorn main:app --reload --host=0.0.0.0 --port=8000 > system_log.log 2>&1 &
```

화면에 print되는 명령어를 전부 ‘>’ 명령어로 지정한 파일에 단순히 뽑아 저장하는 방식이다.

### 3. [번외] nohup을 이용해 background로 실행중인 process 찾아 종료하기

```bash
$ nuhup run.sh > out_log.txt 2>&1 &

$ cat kill_proc.sh
#!/bin/bash

TARGET_NAME=$1
TARGET_LINE=$(ps -ef | grep $TARGET_NAME)
JOB_ID=$(echo #TARGET_LINE | awk '{print $2}')
echo 'Background process [' $TARGET_NAME ', Process ID : ' $JOB_ID '] has been killed. '

kill &JOB_ID

$ . kill_proc.sh run
Background process [run , Process ID : 230053] has been killed.
```

내가 사용하고자 만든 아주 간단한 shell script로, shell 전문가가 아니라 좀 무식한 방법일 순 있지만 아직까지 잘 써먹고있다.

원리는 위 예제 코드의 아랫부분에 적혀있는것 처럼 ‘run’이란 argument를 입력하고 내가 nohup으로 실행시킨 shell script의 이름을 grep해와서 해당하는 process ID 기준으로 kill명령어를 실행하는 간단한 script다.
