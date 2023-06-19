---
layout: post
title: Conda 사용법
description: >
  잊기전에 적어두자 Conda 사용법!
image: /assets/img/tips/conda_tips/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Conda 사용법

## 환경 생성

```bash
option1 : conda create --name <Env_Name>
  -> ex) $ conda create --name ml_work

option2 : conda create --name <Env_Name> python==<version>
  -> ex) $ conda create --name ml_work python==3.9.7

```

## 환경 활성화 / 비활성화

```bash
conda activate <Env_Name> # 활성화

conda deactivate # 비활성화
```

## 환경 삭제

```bash
conda remove --name <Env_Name> --all
  -> ex) $ conda remove --name ml_work --all
```

## 생성된 환경 목록 검색

```bash
conda-env --list
또는
conda info --envs
```

## 환경에 설치된 Package확인

```bash
# activate 이후
conda list
```
