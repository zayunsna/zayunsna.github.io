---
layout: post
title: PDF to Json convertor
description: >
  KoAlpaca 데이터셋을 위해 만들어본 심플 PDF to Json 컨버터
image: /assets/img/post/pdf_json_convertor/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# PDF to Json Convertor

## Convertor를 만들게 된 계기

개인 Local LLM모델을 만들어보고자 model을 불러오고 fine tuning하는 과정에서 입력될 데이터의 format변환이 필요했다.

내 목표는 외부 API를 쓰지 않는 것이기 때문에 langchain의 강력한 기능중 pdf read를 사용할 수 없다. 이어지는 tokenizing을 사용하지 못하기 때문.

그래서 자체적으로 PDF를 읽어와 각 페이지의 있는 content를 학습 dataset에 맞춰 Json으로 변환해주는 코드를 작성해봤다.

먼저 읽어올 pdf의 file 이름을 list에 넣는다.

```python
My_DIR = os.getcwd()
PDF_DIR = My_DIR+'/pdf/'
os.chdir(pdf_dir)
pdf_filelist = set(os.path.splitext(file)[0] for file in glob.glob('*'))
```

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

여기서 soynlp기반 한국어 tokenizer인 konlpy를 사용해서 단어를 parsing하였다.

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
