---
layout: post
title: Linux System info, 환경정보 확인
description: >
  가끔이지만 꼭 한번은 필요한 Linux 환경정보 확인!
image: /assets/img/tips/linux_tips/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Linux 환경 정보 확인

## System 정보

```bash
## Kernal 정보
>> uname -a
## Bios 정보
>> dmidecode -t bios
## System 정보
>> dmidecode -t system
## System hardware 정보
>> sudo lshw
## 또는 요약 버전
>> sudo lshw -short
## 번외 - HTML버전으로도 볼 수 있다. 생성된 lshw.html을 열어보자.
>> sudo lshw -html > lshw.html
```

## CPU 정보

```bash
## CPU 기본 정보
>> cat /proc/cpuinfo
## NUMA 정보도 함께 출력
>> lscpu
## process 정보
>> dmidecode -t processor
```

## GPU 정보

```bash
## GPU 기본 정보
>> sudo lshw -C display
## [번외] nvidia driver설치 되어있는 경우 nvidia gpu driver 확인
>> nvidia-smi
```

## Memory 정보

```bash
## Memory 기본 정보
>> cat /proc/meminfo
## 또는 dmidecode 이용. Memory Device아래 정보가 장착된 memory정보.
>> dmidecode -t memory
```

## Disk 정보

```bash
## disk 용량 확인
>> df -ah # detail 버전
>> df -h # short 버전
## disk 파티션별 용량 확인
>> lsblk
## 또는 모든 정보는
>> lsblk -a
## disk 파티션별 용량 확인 및 정보 [추천하는 방법]
>> sudo fdisk -l
```
