# Deepfake Classification

## 개요
- 모델: **(여기에 모델명/백본 작성)**
- 입력: `test_data/` 폴더의 MP4/JPG 파일
- 비디오 처리: 균등 샘플링으로 프레임 `num_frames`개 추출 → 프레임별 fake 확률 산출 → 평균
- 출력: `submission.csv` (또는 `output/baseline_submission.csv`)

---

## 파일 구조
```text
DACON/
├─ config/
│  └─ config.yaml
├─ env/
│  ├─ Dockerfile
│  ├─ environment.yml
│  └─ requirements.txt
├─ model/
│  ├─ model.pt
│  └─ README.txt
├─ output/
│  └─ README.txt
├─ src/
│  ├─ __pycache__/
│  ├─ dataset.py
│  ├─ models.py
│  └─ utils.py
├─ test_data/
├─ train_data/
├─ venv/
├─ .gitignore
├─ baseline.ipynb
├─ eval.py
├─ inference.py
├─ README.md
├─ submission.csv
├─ sudo_binary.csv
└─ train.py
