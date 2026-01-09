# Deepfake Classification


- 모델: 
- 입력: `test_data/` 폴더의 MP4/JPG 파일들
- 비디오: 균등 샘플링 프레임 `num_frames`개 → 프레임별 fake 확률 → 평균
- 출력: `submission.csv`

---

## 실행 방법

1) `your_submission/` 루트에 `sample_submission.csv`를 둠  
2) `test_data/`에 평가 파일들을 넣음  
3) 추론 실행:

```bash
pip install -r env/requirements.txt
python inference.py --config config/config.yaml
```

결과:
- `output/baseline_submission.csv`

---