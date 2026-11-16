---
lang: ko
layout: post
title: Linux System info, 환경정보 확인
description: >
  가끔이지만 꼭 한번은 필요한 Linux 환경정보 확인!
image: /assets/img/tips/linux_tips/cover.png
lastmod: 2026-11-16
last_modified_at: 2026-11-16
sitemap:
  changefreq: daily
  priority: 1.0
---

# Linux 환경 정보 확인

> 2026-11-16 갱신: 명령별 용도와 확인할 항목을 추가했다. macOS 27.0.1 arm64에서 `uname`·`df`를 실행했고, Linux 전용 명령은 실행하지 않은 예시로 구분했다.
{:.note}

가끔이지만 꼭 한번은 필요한 Linux 환경정보 확인! 드라이버를 설치하거나 장비 사양을 전달할 때 찾아보게 된다.
어떤 정보를 찾는지에 따라 명령을 골라 쓰자. 디스크 사용량은 `df`, 장치와 파티션 연결은 `lsblk`로 확인한다.

이번 작업 환경에는 Docker·Podman과 Linux 실행 환경이 없었다. 아래 Linux 전용 명령의 출력 항목은 링크한 공식 문서에서 확인했다(2026-11-16).
예시의 `sudo`는 해당 Linux 장비의 관리자 권한으로 실행한다는 뜻이다. 펌웨어·장치 접근 권한과 가상화 여부에 따라 조회 가능한 내용이 달라진다.

## System 정보

커널과 아키텍처를 확인할 때 `uname -a`를 쓴다. 원문의 `Kernal` 표기는 `Kernel`로 고쳤고, 복사할 때 불필요한 `>>`도 제거했다.
[uname 문서](https://man7.org/linux/man-pages/man1/uname.1.html)에 옵션별 출력 항목이 정리되어 있다.

```bash
uname -a
```

이 Mac에서 실행한 결과 중 커널 정보 앞부분을 발췌했다. 호스트 이름은 `<hostname>`으로 가렸다.

```text
Darwin <hostname> 27.0.0 Darwin Kernel Version 27.0.0:
```

실제 Linux 장비에서는 커널 이름과 버전, 머신 아키텍처를 확인한다. 배포판 버전과 커널 버전은 따로 관리될 수 있다.

BIOS 버전이나 장비 모델을 찾을 때는 DMI 테이블을 조회한다. 아래는 **실행하지 않은 예시**다.
원문에 권한 표시가 없던 `dmidecode`에는 `sudo`를 붙였다.

```bash
sudo dmidecode -t bios
sudo dmidecode -t system
```

`bios`에서는 Vendor·Version·Release Date, `system`에서는 Manufacturer·Product Name을 확인한다.
[dmidecode 문서](https://manpages.ubuntu.com/manpages/noble/man8/dmidecode.8.html)는 이 정보가 펌웨어에서 온 값이며 부정확하거나 빠질 수도 있다고 설명한다.
권한이 있어도 DMI 테이블을 제공하지 않는 장비에서는 조회가 안 될 수 있다.

하드웨어 구성을 넓게 훑어볼 때는 `lshw`를 쓴다. 세 명령 모두 실행하지 않은 예시다.

```bash
sudo lshw
sudo lshw -short
sudo lshw -html > lshw.html
```

기본 명령은 장치 트리의 상세 정보를, `-short`는 하드웨어 경로와 장치 클래스 중심의 요약을 보여준다.
`-html` 결과는 현재 폴더의 `lshw.html`을 브라우저로 열어볼 수 있다.
[lshw 문서](https://manpages.ubuntu.com/manpages/noble/man1/lshw.1.html)에 따르면 일반 사용자로 실행하면 일부 정보만 나온다.
HTML 저장은 현재 사용자가 해당 폴더에 쓸 수 있어야 하며, 같은 이름의 기존 파일은 덮어쓴다.
결과를 공유하기 전에는 일련번호 같은 식별 정보도 확인하자.

## CPU 정보

논리 CPU 구성이나 NUMA 배치를 확인할 때 쓴다. 아래 세 명령은 실행하지 않은 예시다.

```bash
cat /proc/cpuinfo
lscpu
sudo dmidecode -t processor
```

`/proc/cpuinfo`는 커널이 제공하는 CPU 정보다. 출력 항목은 아키텍처마다 달라서 다른 장비에서 봤던 모델명·플래그가 그대로 있다고 가정하면 안 된다.
[proc_cpuinfo 문서](https://man7.org/linux/man-pages/man5/proc_cpuinfo.5.html)를 확인했다.

`lscpu`는 CPU·스레드·코어·소켓·NUMA 정보를 모아 보여준다. 병렬 작업 설정 전에 CPU 수와 소켓 구성을 볼 때 편하다.
[lscpu 문서](https://man7.org/linux/man-pages/man1/lscpu.1.html)는 가상 환경의 출력이 게스트 구성을 반영하며 일부 정보가 잘못될 수도 있다고 설명한다.
컨테이너나 VM의 출력만 보고 물리 호스트의 사양을 확정하기는 어렵다.

`dmidecode -t processor`는 펌웨어의 Processor Information 항목을 조회한다.
모델·소켓·속도를 대조할 때 쓰되 커널의 현재 CPU 사용률을 보여주는 명령으로 쓰지는 않는다.
원문의 `process` 설명은 `processor` 하드웨어 정보로 고쳤다.

## GPU 정보

그래픽 장치의 모델과 드라이버 상태를 찾을 때 쓴다. 아래는 실행하지 않은 예시다.

```bash
sudo lshw -C display
nvidia-smi
```

`lshw -C display`는 display 클래스만 걸러낸다. product·vendor와 드라이버 관련 항목을 확인할 수 있으며 출력 범위는 장치 탐지 결과에 달려 있다.
`nvidia-smi`는 NVIDIA GPU와 지원 드라이버가 설치된 환경에서 Driver Version·GPU 이름·메모리 사용량 등을 확인할 때 쓴다.
[NVIDIA 공식 문서](https://docs.nvidia.com/deploy/nvidia-smi/index.html)를 확인했다.
명령을 찾지 못하거나 드라이버 통신 오류가 나면 설치·장치 접근 상태를 먼저 확인한다.
이 Mac에서는 NVIDIA 드라이버 경로를 검증하지 않았다.

## Memory 정보

메모리 부족을 볼 때와 장착된 메모리 구성을 볼 때 필요한 명령이 다르다. 둘 다 실행하지 않은 예시다.

```bash
cat /proc/meminfo
sudo dmidecode -t memory
```

`/proc/meminfo`에서는 MemTotal·MemFree·MemAvailable·SwapTotal 등을 확인한다.
캐시와 회수 가능한 메모리까지 고려해 여유를 볼 때는 MemAvailable을 같이 본다.
[proc_meminfo 문서](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html)의 필드 정의를 참고했다.

`dmidecode -t memory`에서는 Memory Device의 Size·Locator·Type 같은 항목으로 슬롯과 모듈 구성을 확인한다.
이 값 역시 펌웨어가 제공하므로 비어 있는 슬롯 정보나 누락된 값이 있을 수 있다.

## Disk 정보

파일을 더 저장할 공간이 있는지 볼 때는 `df`를 쓴다.
`-h`는 읽기 쉬운 단위, `-a`는 Linux GNU df에서 의사 파일시스템 등을 포함해 출력하는 옵션이다.
[df 문서](https://man7.org/linux/man-pages/man1/df.1.html)를 확인했다.

```bash
df -ah
df -h
```

두 명령은 이 Mac에서도 실행했다. 다음은 BSD 계열 `df -h` 결과의 첫 파일시스템 행을 발췌한 것이다. Linux GNU df와 열·옵션 동작이 같다고 가정하지 않는다.

```text
Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s1s1   460Gi    13Gi   217Gi     6%    484k  2.3G    0%   /
```

디스크가 가득 찼는지는 Used·Avail·사용률과 마운트 지점을 같이 본다. 원문에서 `-ah`를 상세 버전이라고 적었지만,
Linux에서 `-a`는 추가 파일시스템을 포함하는 옵션이다. 열별 상세 정보를 전부 늘려주는 옵션이라는 뜻은 아니다.

파티션 구조가 필요하면 다음으로 넘어간다. 아래 세 명령은 실행하지 않은 Linux 예시다.

```bash
lsblk
lsblk -a
sudo fdisk -l
```

`lsblk`는 블록 장치를 트리로 보여주므로 디스크와 그 아래 파티션의 관계를 확인하기 편하다.
`-a`는 기본 출력에서 제외될 수 있는 장치까지 포함한다.
[lsblk 문서](https://man7.org/linux/man-pages/man8/lsblk.8.html)에 따라 NAME·SIZE·TYPE·마운트 지점을 대조한다.
장치 크기와 파일시스템의 사용 가능 용량은 서로 다른 값이다.

`fdisk -l`은 파티션 테이블을 나열한다. 디스크별 용량과 파티션의 시작·끝·타입을 확인할 때 쓴다.
여기서는 목록 조회 옵션 `-l`까지 넣어 사용한다.
[fdisk 문서](https://man7.org/linux/man-pages/man8/fdisk.8.html)의 기본 대화형 모드는 파티션을 변경할 수도 있으니 명령을 입력할 때 옵션을 확인하자.
