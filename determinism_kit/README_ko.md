# TLS-SMF Stage A 크로스 머신 결정론 검증 킷

리뷰어 지적("결정론적 기하 백본이 다른 OS·라이브러리 버전·하드웨어에서도
결정론적인지 검증되지 않았다") 대응용. 다른 PC(예: RTX 4070 노트북 — GPU는
무관, 백본은 CPU 전용)에서 동일한 E57 스캔으로 Stage A를 돌리고, 산출된
**클러스터별 .ply 전체의 SHA256**을 레퍼런스 머신 결과와 비교한다.

- 파이프라인: `extract_objects.py` (voxel 0.03 m → seeded RANSAC 구조 제거 →
  DBSCAN eps 0.30 + giant-split → ceiling-soffit remnant 필터)
- 기대값 (vis_n2 스캔): **클러스터 88개 → soffit 필터 후 81개 유지**
- 레퍼런스 (`reference_checksums.json`): 데스크탑(Ryzen 9 9900X, Ubuntu
  24.04 호스트)에서 이 컨테이너로 생성. 플랫폼 정보(CPU/glibc/패키지 버전)가
  json 안에 기록돼 있음. 참고: 컨테이너(glibc 2.41 / Python 3.12.14)와
  호스트 네이티브 .venv(glibc 2.39 / Python 3.12.3) 결과가 이미 바이트
  동일 — OS·라이브러리 변인 1차 통과. 이미지 크기: 전송(압축) 약 770 MB,
  디스크 약 3 GB.

## 준비물

1. Docker 설치된 노트북
2. 스캔 파일: `vis_n2 1.e57` (약 1.3 GB) — USB 등으로 복사
3. 이 킷 zip (`determinism_kit_20260901.zip`)

## 1단계 — 이미지 준비 (둘 중 하나)

**방법 A: 노트북에서 직접 빌드** (인터넷 필요, 수 분):

```bash
unzip determinism_kit_20260901.zip && cd determinism_kit
docker build -t tlssmf-determinism .
```

**방법 B: 데스크탑에서 이미지 파일로 전달** (노트북 인터넷 불필요):

```bash
# 데스크탑에서 (이미지는 이미 빌드돼 있음):
docker save tlssmf-determinism | gzip > tlssmf-determinism.tar.gz
# 파일을 노트북으로 복사한 뒤, 노트북에서:
docker load < tlssmf-determinism.tar.gz
```

## 2단계 — 실행 (한 줄)

E57이 들어 있는 폴더를 `/data`로 마운트하고 파일명을 인자로 준다:

```bash
docker run --rm -v "/home/USER/scans:/data" tlssmf-determinism "/data/vis_n2 1.e57"
```

- `/home/USER/scans` 부분만 실제 E57이 있는 폴더로 바꾸면 됨 (공백 있는
  파일명이라 따옴표 유지).
- 끝나면 같은 폴더에 **`determinism_report.json`** 이 생긴다.
- 소요 시간: 레퍼런스 데스크탑(Ryzen 9 9900X) 기준 컨테이너 실행 약 25초.
  노트북에서는 1–3분 예상. 메모리 여유 8 GB 이상 권장.
- 로그 마지막에 `clusters retained: 81 (pre-filter 88, dropped 7)`,
  `points 244,975` 와 `aggregate_sha256` 이 출력된다. 레퍼런스 aggregate:
  `9b36600da0780db349d31607b41a092aedca04312e5d5d90c23de335c2e5d0a1`
  — 이 한 값만 같아도 사실상 전체 일치.

## 3단계 — 레퍼런스와 비교

```bash
python3 compare_checksums.py reference_checksums.json "/home/USER/scans/determinism_report.json"
```

- **`RESULT: MATCH`** → 81개 클러스터 + objects.csv 전부 바이트 동일.
  크로스 머신 결정론 입증 완료.
- **`RESULT: MISMATCH`** → 어떤 파일이 다른지 목록이 출력된다. 이 경우
  `determinism_report.json` 파일만 회수해서(USB/메신저) 데스크탑으로
  가져오면 됨 — 그 자체로 논문에 쓸 분석 대상 (플랫폼별 부동소수점
  차이가 어느 클러스터 경계를 흔들었는지 json 해시 대조로 특정 가능).

비교 스크립트는 표준 라이브러리만 사용 — 노트북에 python3만 있으면 되고,
없으면 json 파일만 회수해 와도 된다.

## 파일 목록

| 파일 | 용도 |
|---|---|
| `Dockerfile` + `code/` + `requirements-pinned.txt` | 검증 이미지 빌드 (백본 코드 + 고정 버전) |
| `verify_determinism.sh` | 컨테이너 엔트리포인트 (Stage A 실행 + 해시) |
| `make_report.py` | 클러스터 SHA256 + 플랫폼 정보 → report json |
| `compare_checksums.py` | 레퍼런스 vs 노트북 report 비교 |
| `reference_checksums.json` | 레퍼런스 머신(데스크탑, 컨테이너) 결과 |
