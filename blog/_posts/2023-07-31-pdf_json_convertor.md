---
lang: ko
layout: post
title: Python으로 PDF를 페이지별 JSON으로 변환하기 (pdfminer.six + KoNLPy)
description: >
  LLM 파인튜닝 데이터를 만들려고 짠 PDF → JSON 변환 코드. 2026년에 다시 돌려보고 버그 3개(NameError, 이미 지운 PDF를 다시 여는 문제, PDF가 아닌 파일)를 고친 버전 포함.
image: /assets/img/post/pdf_json_convertor/cover.png
lastmod: 2026-10-06
last_modified_at: 2026-10-06
sitemap:
  changefreq: daily
  priority: 1.0
---

# Python으로 PDF를 페이지별 JSON으로 변환하기 (pdfminer.six + KoNLPy)

> **Corrected October 2026:** The code was re-run on pdfminer.six 20260107 and KoNLPy 0.6.0. Three bugs were fixed: an undefined `pdf_dir` in the first snippet, `symmetric_difference` trying to open PDFs that only exist as old JSON files, and `glob('*')` picking up non-PDF files. Two claims were also wrong: LangChain's PDF loader works offline, and KoNLPy is not based on soynlp. A fixed version is at the end.
{:.note}

## Convertor를 만들게 된 계기

개인 Local LLM모델을 만들어보고자 model을 불러오고 fine tuning하는 과정에서 입력될 데이터의 format변환이 필요했다.

내 목표는 외부 API를 쓰지 않는 것이기 때문에 langchain의 강력한 기능중 pdf read를 사용할 수 없다. 이어지는 tokenizing을 사용하지 못하기 때문.

> (2026 수정) 이건 내가 잘못 알고 있었다. LangChain의 PDF 로더(`PyPDFLoader`)는 pypdf로 파일을 직접 읽어서 외부 API가 필요 없다. 이번에 API 키 없이 돌려보니 샘플 PDF 2쪽을 그대로 읽었다(langchain-community 0.4.2, pypdf 6.19.0). 그래도 페이지별로 내가 원하는 형식을 만들고 싶으면 아래처럼 직접 짜는 것도 나쁘지 않다.

그래서 자체적으로 PDF를 읽어와 각 페이지의 있는 content를 학습 dataset에 맞춰 Json으로 변환해주는 코드를 작성해봤다.

먼저 읽어올 pdf의 file 이름을 list에 넣는다.

```python
My_DIR = os.getcwd()
PDF_DIR = My_DIR+'/pdf/'
os.chdir(PDF_DIR)
pdf_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))
```

(2026 수정: 원래 `os.chdir(pdf_dir)`로 적어서 `NameError`가 났다. 위에서 만든 변수는 대문자 `PDF_DIR`이다.)

위 코드가 동작하면, pdf파일 안에 있는 파일들이 확장자 없이 list에 저장된다.

왜 확장자를 지우고 파일 이름만 저장해 두었냐면 나중에 변환된 json의 파일과 비교해서 이미 변환된 경우는 제외하고 진행하고자 했기 때문이다.

/pdf/ 폴더안에 있는 모든 pdf 파일을 불러와 변환 시키는 구조로 코딩을 하려고 하는데, 새로운 pdf을 넣을 때마다 기존 pdf까지 다시 변환하고 싶지는 않기 때문이다.

그래서 변환할 파일 이름 list를 가져오는 함수는 아래와 같다.

```python
def get_filelists(pdf_dir, json_dir):
    # pdf가 있는 폴더 이동
    os.chdir(pdf_dir)
    # pdf 모든 파일 이름 list화
    pdf_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))
    # 파일 존재 여부 확인을 위한 json(output) 폴더 이동
    os.chdir(json_dir)
    # 변환되어있는 json 파일 이름 list화
    json_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))

    # unique비교를 통해 변환되지 않은 pdf 파일 이름만 저장
    unique_files = pdf_filelist.symmetric_difference(json_filelist)

    return unique_files
```

> (2026 수정) 여기 버그가 두 개 있었다. 다시 돌려보고 알았다.
> 1. `symmetric_difference`는 양쪽 중 한쪽에만 있는 이름을 다 돌려준다. 그래서 PDF는 지웠는데 예전 JSON이 남아 있으면 그 이름도 "변환할 목록"에 들어가고, 없는 PDF를 열다가 `FileNotFoundError`가 난다. 필요한 건 "PDF에는 있고 JSON에는 없는 것", 즉 차집합 `pdf_filelist - json_filelist`이다.
> 2. `glob('*')`는 확장자를 안 가린다. pdf 폴더에 `README.txt` 하나만 있어도 `README.pdf`를 열려다 같은 에러가 난다. `glob('*.pdf')`로 걸러야 한다.

그 다음 중요한 것은, pdf파일을 불러와 각 page에 있는 content를 읽고 저장하는 기능이다.

먼저 난 각 page별로 dataset을 만들어 학습시킬 생각이므로 page별로 불러와 안에 있는 content를 빼내야 한다.

아래 코드는 pdf의 특정 page의 content를 추출하는 코드이다.

```python
def extract_text_from_pdf(file_path, page_number):
    page_number = page_number - 1

    extracted_text = ""

    # pdf의 모든 페이지에 대해서만 반복
    for i, page_layout in enumerate(extract_pages(file_path)):
        # 실제 동작은 내가 원하는 page에서만 작동
        if i == page_number:
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    extracted_text += element.get_text()
            break

    # '.'를 기준으로 문장을 구별
    sentences = extracted_text.split('.')

    # Json형식으로 변환
    json_objects = [{"text": sentence.strip()} for sentence in sentences]

    return json_objects
```

위 의 코드대로 실행하면, 내부에 있는 content는 전부 잘 불어와 지는데, ‘.’를 기반으로 마지막에 나눠지기 때문에 page안의 content들이 문장 단위로 저장되게 된다.

거기다가 한글은 잘 구별해주지 못한다.

여기서 한국어 형태소 분석 패키지인 konlpy의 Kkma로 문장을 나눴다.

(2026 수정: 원래 "soynlp기반 konlpy"라고 적었는데 둘은 다른 패키지다. KoNLPy는 Kkma 같은 Java 기반 분석기를 파이썬에서 쓰게 해주는 패키지라서 Java가 깔려 있어야 한다. 그리고 아래 코드가 하는 일은 단어 parsing이 아니라 `Kkma().sentences()`로 **문장 나누기**다.)

konlpy와 soynlp의 자세한 사용방법은 이곳에서. [https://wikidocs.net/92961](https://wikidocs.net/92961)

추가로 특정 page가 아닌 입력된 pdf전체에 대해 저장하는 구조까지 추가하면 아래와 같은 코드가 완성된다.

```python
def pdfReader_byPage(file_path):
    extract_texts = []
    count = 1

    # PDF 안의 모든 page에 대해 반복
    for page_layout in extract_pages(file_path):
        page_text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()

        # 각 page별 konlpy를 이용해 문장 추출
        sentences = Kkma().sentences(page_text)
        # 각 문장사이 빈칸을 추가하여 모두 합침
        out_sentences = ' '.join(sentences)
        # 각 page별로 json형태 변환 후 최종 json format에 append
        extract_texts.append({"text": out_sentences,'page': count})
        count += 1

    return extract_texts
```

위 정의 된 두 함수를 이용하여, 선정된 pdf파일에 대해 반복하는 함수를 작성하면 끝.

git repo : [https://github.com/zayunsna/PDF_to_Json_convertor](https://github.com/zayunsna/PDF_to_Json_convertor)

## 최종 코드

(2023년에 쓴 원래 코드다. 위에서 말한 버그 두 개가 그대로 있어서, 2026년에 고친 버전을 맨 아래에 따로 붙였다.)

```python
import os
import glob
import json
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer
from konlpy.tag import Kkma

def get_filelists(pdf_dir, json_dir):
    os.chdir(pdf_dir)
    pdf_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))
    os.chdir(json_dir)
    json_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))

    unique_files = pdf_filelist.symmetric_difference(json_filelist)

    return unique_files

def pdfReader_byPage(file_path):
    extract_texts = []
    count = 1

    for page_layout in extract_pages(file_path):
        page_text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()

        sentences = Kkma().sentences(page_text)
        out_sentences = ' '.join(sentences)
        extract_texts.append({"text": out_sentences,'page': count})
        count += 1

    return extract_texts

My_DIR = os.getcwd()
PDF_DIR = My_DIR+'/pdf/'
Json_DIR = My_DIR+'/json/'

filelist = get_filelists(PDF_DIR, Json_DIR)


print("#"*100)
print("PDF file path : ", PDF_DIR)
print("Result will be saved here : ", Json_DIR)
print("#"*100)
print(" The file that already converted and existed in /json/ dir will be skipped. ")
print("#"*100)
print(" Now On-progress, ")
for filename in filelist:
    print(" => ", filename)
    input_filename = PDF_DIR+filename+'.pdf'
    result = pdfReader_byPage(input_filename)
    result_filename = Json_DIR+filename+'.json'
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print("#"*100)
print(" Converting has been done !! ")
```

## 2026년에 다시 돌려보기

3년 만에 다시 돌려봤다. 한국어 문장이 들어간 2쪽짜리 샘플 PDF를 만들고, 위 원래 코드를 세 가지 상황에서 실행했다(Python 3.12, pdfminer.six 20260107, KoNLPy 0.6.0, JPype1 1.7.1, OpenJDK 27, 2026-10-06).

| 상황 | 원래 코드 | 고친 코드 |
|---|---|---|
| pdf/에 sample.pdf만 있음 | 정상 변환 | 정상 변환 (결과 동일) |
| json/에 PDF 없는 `old_notes.json`이 남아 있음 | `FileNotFoundError: ... pdf/old_notes.pdf` | sample만 변환, old_notes는 그대로 둠 |
| pdf/에 `README.txt`가 섞여 있음 | `FileNotFoundError: ... pdf/README.pdf` | txt는 무시하고 sample만 변환 |
| 한 번 더 실행 | (같음) | 이미 변환된 파일은 건너뜀 |

고친 코드다. 구조는 그대로 두고 파일 목록 부분만 `pathlib`으로 바꿨고, Kkma는 JVM을 띄우느라 느려서 한 번만 만들게 했다.

```python
import json
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from konlpy.tag import Kkma

kkma = Kkma()  # JVM을 띄우는 데 시간이 걸리니 한 번만 만든다


def get_filelists(pdf_dir, json_dir):
    # 확장자로 거른다: pdf/ 안의 .pdf, json/ 안의 .json만
    pdf_names = {p.stem for p in Path(pdf_dir).glob("*.pdf")}
    json_names = {p.stem for p in Path(json_dir).glob("*.json")}
    # PDF는 있는데 JSON이 아직 없는 것만 (차집합)
    return sorted(pdf_names - json_names)


def pdfReader_byPage(file_path):
    extract_texts = []
    for count, page_layout in enumerate(extract_pages(file_path), start=1):
        page_text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()

        sentences = kkma.sentences(page_text)
        extract_texts.append({"text": " ".join(sentences), "page": count})

    return extract_texts


PDF_DIR = Path.cwd() / "pdf"
JSON_DIR = Path.cwd() / "json"

for filename in get_filelists(PDF_DIR, JSON_DIR):
    print(" => ", filename)
    result = pdfReader_byPage(PDF_DIR / f"{filename}.pdf")
    with open(JSON_DIR / f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print(" Converting has been done !! ")
```

샘플 PDF를 넣고 돌린 결과(`json/sample.json`):

```json
[
  {
    "text": "로컬 LLM을 파인 튜닝하려면 학습 데이터가 필요하다. PDF 문서를 페이지 별로 읽어 JSON으로 저장한다. 버전은 3.2.1 이다.",
    "page": 1
  },
  {
    "text": "두 번째 페이지입니다. 문장이 잘 나뉘는지 확인한다. 외부 API는 쓰지 않는다.",
    "page": 2
  }
]
```

하나 새로 알게 된 것. PDF 원문은 "파인튜닝", "페이지별로", "3.2.1이다"였는데 결과에는 "파인 튜닝", "페이지 별로", "3.2.1 이다"로 들어갔다. Kkma가 문장을 나누면서 **띄어쓰기도 자기 기준으로 바꾼다.** 학습 데이터를 원문 그대로 남기고 싶으면 문장 나누기를 빼고 `page_text`를 그대로 저장하는 게 낫다.

설치할 때 하나 더. KoNLPy는 Java가 있어야 돌아간다. 맥에서는 `brew install openjdk`로 깔고 `JAVA_HOME=/opt/homebrew/opt/openjdk`를 지정해서 실행했다.
