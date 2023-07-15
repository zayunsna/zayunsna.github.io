---
layout: post
title: Ubuntu 자동 업데이트 끄기
description: >
  How to turn-off Ubuntu driver auto-update
image: /assets/img/tips/ubuntu_autoupdate_off/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Ubuntu 자동 업데이트 끄기

오픈소스로 굴러가는 Ubuntu는 OS version update할 때 마다 기존 설치되어있는 driver들이 망가지는 아주 귀찮은 단점이 있다.

보통 초반에 자동 업데이트 설정을 꺼두고 잊고 살고 있지만 이번에 새로 환경을 설정하면서 기록해 놓기 위해 적어 놓는다.

## 설정 파일 위치

```bash
## Directory
>> cd /etc/apt/apt.conf.d/

## Target file
>> 10periodic
>> 20auto-upgrades
```

## 10preiodic

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

## 20auto-upgrade

```bash
## Before
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";

## After
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
```
