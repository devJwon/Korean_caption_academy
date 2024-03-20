# [한국어 제목학원 데이터셋]



## 데이터셋 설명

- 한국어 제목학원 데이터셋은 한국어 유머 이미지와 그에 대한 유머 캡션 및 설명을 포함하는 데이터셋입니다. 이 데이터셋은 이미지 캡셔닝, 자연어 처리, 유머 인식 등의 연구 분야에서 활용될 수 있습니다. 데이터셋의 이미지들은 googleImageDownloader를 활용하여, 구글에 ‘제목학원’이라는 키워드를 검색하였을 때 등장하는 이미지를 무작위로 크롤링하였습니다. 
- 수집된 이미지는 그림과 캡션이 붙어있는 형태였기에 이를 수동으로 분리하는 작업을 수행하였습니다. 분리 작업의 경우 이미지에 대해서는 Microsoft의 캡처 도구를 활용하여 크롭하는 방식으로, 캡션에 대해서는 그림 내의 텍스트를 타이핑하여 정리하는 방식으로 수행하였습니다.



## 데이터 구조

- 데이터셋은 JSON 파일과 CSV 파일, 두 가지 형식으로 제공됩니다.

  ### JSON 파일 구조

  - `image_url`: 이미지 URL
  - `humor_caption`: 이미지에 대한 한국어 유머 캡션
  - `description`: 이미지에 대한 설명

  ### CSV 파일 구조

  - `image_url`: 이미지 URL
  - `humor_caption`: 이미지에 대한 한국어 유머 캡션
  - `description`: 이미지에 대한 설명



## 데이터 예시

```javascript

[{"image_url":"https:\/\/github.com\/devJwon\/humor-image-captioning\/blob\/c4f76c427eab74c86a8be40bdb0aa80ea02d351b\/images\/crop\/crop_00000002.jpg?raw=true","humor_caption":"아빠 저녁거리 데려왔어요; 순서대로 털 벗고 불판에 올라가거라","description":"호랑이 캐릭터가 녹색 스웨터를 입고 바비큐를 하고 있는 동안 다른 호랑이 캐릭터들이 기대에 찬 눈빛으로 보고 있어요."}]

```



## 데이터 통계

- 전체 이미지 개수: 100개
- 전체 유머 캡션 개수: 100개
- 전체 설명 개수: 100개



## 인용 방법

현재 투고 후 심사 중



## 기여 방법

데이터셋 확장, 오류 수정, 추가 애노테이션 등 기여를 원하시는 경우 jwon.lee70@gmail.com 로 연락 주시기 바랍니다.

## 연락처

데이터셋 관련 문의사항은 jwon.lee70@gmail.com 로 연락 주시기 바랍니다.



