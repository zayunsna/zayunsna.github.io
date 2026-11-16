---
lang: ko
layout: post
title: Conda 사용법
description: >
  잊기전에 적어두자 Conda 사용법!
image: /assets/img/tips/conda_tips/cover.png
lastmod: 2026-11-16
last_modified_at: 2026-11-16
sitemap:
  changefreq: daily
  priority: 1.0
---

# Conda 사용법

> 2026-11-16 갱신: macOS 27.0.1 arm64의 Miniconda(conda 25.11.1)에서 임시 환경 생성·활성화·조회·삭제를 확인했다. 원문의 환경 목록 명령을 고치고 실행 결과를 붙였다.
{:.note}

잊기전에 적어두자 Conda 사용법! 프로젝트마다 Python과 패키지 버전을 분리할 때 쓴다.
`conda create`로 환경을 만들고 `conda activate`로 들어간 뒤 패키지를 설치하면 된다. 환경을 바꾼 뒤에는 Python 경로도 한 번 확인해두자.

## 새 환경 만들기

이름만 주면 Python을 지정하지 않은 환경을 만든다. Python까지 필요하면 버전 조건을 같이 적는다.
`<Env_Name>`은 쓸 이름으로 바꿔야 하는 자리다. 아래 `ml_work`처럼 꺾쇠 없이 입력한다.

```bash
conda create --name <Env_Name>
conda create --name <Env_Name> python=<version>
```

원문의 `python==3.9.7`은 정확한 버전 지정 예시였다. 여기서는 `python=3.13.11`로 실행했다.
선택한 채널과 운영체제에서 받을 수 있는 버전을 써야 한다.
[Conda 환경 관리 문서](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)의 생성·활성화·삭제 절을 확인했다(2026-11-16).

이번 확인은 기존 환경과 이름이 겹치지 않게 별도 임시 환경 폴더에서 했다.
다음은 실행한 생성 명령이다. `--offline`은 로컬 패키지 캐시를 쓰는 옵션이라 캐시에 필요한 패키지가 있어야 한다.
처음 설치하는 경우에는 이 옵션을 빼고 다운로드한다. `-y`는 설치 계획 확인에 자동으로 동의한다.

```bash
conda create -y --offline --name hk_adsense_empty
conda create -y --offline --name ml_work python=3.13.11
```

두 명령 모두 완료됐다. 첫 환경에는 패키지를 지정하지 않았고, 두 번째에는 Python과 필요한 의존 패키지가 설치됐다.

## 활성화한 Python 확인

`conda activate`는 현재 셸이 사용할 환경을 바꾼다. 원문 명령은 다음과 같다.

```bash
conda activate <Env_Name>
conda deactivate
```

실제 확인에는 만든 이름 `ml_work`를 넣었다. 패키지 설치 전에 실행 파일이 의도한 환경을 가리키는지 확인하면 엉뚱한 환경에 설치하는 일을 줄일 수 있다.

```bash
conda activate ml_work
python --version
python -c 'import sys; print(sys.prefix.endswith("/envs/ml_work"))'
```

실행 결과:

```text
Python 3.13.11
True
```

`True`는 이번 임시 폴더의 `ml_work`에서 실행 중이라는 확인값이다. 일반 사용에서는 `python -c 'import sys; print(sys.executable)'`로 전체 실행 경로를 확인할 수 있다.
`conda activate`가 셸 초기화 오류를 내면 설치한 Conda의 셸 설정을 먼저 확인한다.

## 등록된 환경 목록

어떤 이름으로 만들었는지 잊었을 때는 아래 둘 중 하나를 쓴다. 원문의 `conda-env --list`는 `conda env list`로 바로잡았다.

```bash
conda env list
conda info --envs
```

두 명령에서 같은 임시 환경을 확인했다. 결과 중 해당 행만 발췌했고 로컬 경로 앞부분은 `<temp>`로 가렸다.

```text
ml_work              *   <temp>/envs/ml_work
```

별표 `*`가 현재 활성 환경이다. 이름이 비슷한 환경이 여러 개라면 오른쪽 경로까지 확인하자.

## 설치된 패키지 조회

활성화한 환경에 무엇이 들어있는지 볼 때 쓴다. 오류를 공유할 때 Python뿐 아니라 문제 패키지의 버전·빌드도 같이 확인할 수 있다.

```bash
conda list
```

실행 결과 중 Python과 pip 행이다. 전체 출력에는 버전, 빌드, 채널 열이 있다.

```text
pip                        25.3             pyhc872135_0
python                     3.13.11          hc7d8306_100_cp313
```

## 작업을 끝낸 환경 삭제

먼저 `conda deactivate`로 환경에서 나온다. 이번 실행에서는 앞서 활성화되어 있던 `base`로 돌아왔다.
비활성화는 환경을 디스크에 남겨두므로 다시 쓸 수 있다. 삭제는 해당 환경의 패키지까지 없앤다.

```bash
conda deactivate
conda remove --name <Env_Name> --all
```

실제로는 이번에 만든 두 환경만 지정해 지웠다. 평소에는 `-y`를 빼고 삭제 대상 경로와 계획을 읽은 뒤 확인하는 편이 좋다.

```bash
conda remove -y --name ml_work --all
conda remove -y --name hk_adsense_empty --all
```

명령이 끝난 뒤 두 환경 디렉터리가 사라진 것까지 확인했다. 패키지를 지정하지 않은 빈 환경은 삭제 때 아래 메시지도 나왔다.

```text
No packages found in <temp>/envs/hk_adsense_empty. Continuing environment removal
```

위 결과는 이번 로컬 캐시에 있던 패키지로 확인한 것이다. 기존 `base`와 작업용 환경은 변경하지 않았다.
