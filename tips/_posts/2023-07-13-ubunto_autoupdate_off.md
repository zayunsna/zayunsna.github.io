---
lang: ko
layout: post
title: "Ubuntu 자동 업데이트 끄기 (드라이버만 고정하는 방법 포함)"
description: >
  Ubuntu 자동 업데이트를 끄는 설정 파일과, 보안 업데이트는 살리고 그래픽 드라이버 같은 특정 패키지만 고정하는 방법(apt-mark hold, Package-Blacklist)을 정리했다.
lastmod: 2026-09-29
last_modified_at: 2026-09-29
image: /assets/img/tips/ubuntu_autoupdate_off/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Ubuntu 자동 업데이트 끄기 (드라이버만 고정하는 방법 포함)

> **Updated September 2026:** Added a safer option that keeps security updates on and only holds specific packages, based on the current Ubuntu Server documentation (24.04 LTS). The commands in the new sections were checked against the official docs but not re-run on an Ubuntu machine for this update.
{:.note}

## 요약: 전부 끌까, 일부만 고정할까

| 방법 | 효과 | 추천 상황 |
|---|---|---|
| **특정 패키지만 고정** (`apt-mark hold`, `Package-Blacklist`) | 보안 업데이트는 계속 받고, 드라이버·커널처럼 깨지면 곤란한 패키지만 멈춘다 | 대부분의 경우 (**권장**) |
| **자동 업데이트 전부 끄기** (아래 원문 방법) | 모든 자동 업데이트가 멈춘다 | 인터넷과 분리된 실험 장비처럼 업데이트를 직접 관리할 때 |

Ubuntu 공식 문서는 자동 업데이트를 끌 수 있는 방법을 안내하면서도, 문제가 생길 위험이 "보안 업데이트를 적용하지 않는 위험보다 작다"고 본다. 드라이버 때문이라면 전부 끄기보다 해당 패키지만 고정하는 편이 낫다.

## 방법 1 (권장): 특정 패키지만 고정하기

먼저 고정할 패키지 이름을 확인한다. 예를 들어 NVIDIA 드라이버라면:

```bash
apt list --installed 2>/dev/null | grep -i nvidia
```

**apt-mark hold** — 해당 패키지의 자동 설치·업그레이드·삭제를 막는다.

```bash
sudo apt-mark hold <패키지이름>      # 고정
apt-mark showhold                    # 고정된 패키지 목록
sudo apt-mark unhold <패키지이름>    # 고정 해제
```

**Package-Blacklist** — 자동 업데이트(unattended-upgrades)에서만 제외한다. `/etc/apt/apt.conf.d/50unattended-upgrades`에 파이썬 정규식으로 적는다.

```
Unattended-Upgrade::Package-Blacklist {
    "nvidia-";
};
```

공식 문서의 주의: 어떤 패키지를 막으면, 그 패키지에 의존하는 다른 업데이트도 설치되지 않을 수 있다.

## 방법 2: 자동 업데이트 전부 끄기 (2023년 원문)

오픈소스로 굴러가는 Ubuntu는 OS version update할 때 마다 기존 설치되어있는 driver들이 망가지는 아주 귀찮은 단점이 있다.

보통 초반에 자동 업데이트 설정을 꺼두고 잊고 살고 있지만 이번에 새로 환경을 설정하면서 기록해 놓기 위해 적어 놓는다.

### 설정 파일 위치

```bash
## Directory
>> cd /etc/apt/apt.conf.d/

## Target file
>> 10periodic
>> 20auto-upgrades
```

### 10periodic (데스크톱에 있는 경우)

```bash
## Before
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";

## After
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";
```

### 20auto-upgrades

```bash
## Before
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";

## After
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
```

`20auto-upgrades`의 두 값을 `"0"`으로 바꾸는 방법은 현재 Ubuntu Server 문서(24.04 LTS)에 나온 방법과 같다.

## 참고 문서

- [Automatic updates — Ubuntu Server documentation](https://ubuntu.com/server/docs/how-to/software/automatic-updates/) — 2026-09-29 확인
- [apt-mark(8) — Ubuntu Manpage](https://manpages.ubuntu.com/manpages/noble/en/man8/apt-mark.8.html)
