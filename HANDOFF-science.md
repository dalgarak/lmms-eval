# `science` 브랜치 인수인계 문서

다른 서버에서 이 작업을 이어받기 위한 문서입니다.
저장소: `https://github.com/dalgarak/lmms-eval` (upstream `EvolvingLMMs-Lab/lmms-eval`의 fork)
작업 브랜치: `science`

---

## 1. 이 브랜치가 하는 일

Qwen3.5 계열 모델을 **과학 도메인 멀티모달 벤치마크**로 평가하기 위해,
lmms-eval에 없는 태스크들을 직접 추가한 브랜치입니다.

`science` 고유 커밋은 upstream `main` 위에 얹혀 있고, 주기적으로 `origin/main`을 머지해 최신화합니다.

```
* Merge remote-tracking branch 'origin/main' into science
* Merge remote-tracking branch 'origin/main' into science
* feat: add science-domain benchmark tasks
* MolParse, OpenRxn 기본 테스트 완료
* (upstream main)
```

---

## 2. 새 서버에서 시작하기

### 2.1 클론 & 브랜치

```bash
git clone https://github.com/dalgarak/lmms-eval.git
cd lmms-eval
git checkout science
```

### 2.2 git 신원 설정 (필요)

이 저장소에는 git user 설정이 없습니다. 커밋 전에 한 번 설정하세요.

```bash
git config --local user.name  "dalgarak"
git config --local user.email "20063100+dalgarak@users.noreply.github.com"
```

### 2.3 push 인증

HTTPS remote라 자격증명이 필요합니다. 둘 중 하나:

```bash
# (a) SSH 키를 GitHub 계정에 등록한 경우
git remote set-url origin git@github.com:dalgarak/lmms-eval.git

# (b) PAT 사용
git config --local credential.helper store   # 최초 1회 입력 후 저장
```

### 2.4 환경

기존 작업 서버 기준(참고값):

| 항목 | 값 |
|---|---|
| Python | 3.12.3 (`requires-python >=3.10`) |
| 패키지 매니저 | `uv` 0.9.18 |
| GPU | NVIDIA B200 183GB × 4 |
| 가상환경 | 리포 루트의 `.venv` (uv 관리) |

```bash
uv sync
```

주의할 의존성 (`pyproject.toml`의 `dependencies`에 이 브랜치가 추가한 것):

| 패키지 | 용도 |
|---|---|
| `rouge_score`, `bert_score`, `pymeteor` | sfe 계열 생성 태스크 채점 |
| `causal-conv1d`, `flash-linear-attention` | Qwen3.5 추론 가속 |

- `pymeteor`는 **test.pypi.org**에서 받습니다. `pyproject.toml`의 `[tool.uv.sources]` / `[[tool.uv.index]]` 설정이 이를 담당하니 지우지 마세요.
- Qwen3.5 모델 로딩에는 `transformers>=5.3.0`이 필요합니다 (스크립트 상단 주석 참고).
- upstream이 최근 린터를 black/isort → **ruff**로 교체했습니다. 포맷 관련 커밋 시 참고.

---

## 3. 실행 방법

실행 스크립트는 `examples/models/` 아래 두 개입니다. 같은 내용이고 **GPU만 나눠 쓰는 구조**입니다.

| 스크립트 | GPU | 포트 |
|---|---|---|
| `examples/models/qwen35_science.sh` | 기본(0,1) | 12346 |
| `examples/models/qwen35_science_2.sh` | `CUDA_VISIBLE_DEVICES=2,3` | 12348 |

```bash
bash examples/models/qwen35_science.sh
```

스크립트 안에서 바꿔가며 쓰는 값:

- `TASKS` — 돌릴 태스크. 여러 개는 콤마로 구분 (히스토리가 주석으로 남아 있음)
- `MODEL` — 현재 `Qwen/Qwen3.5-9B`
- `--limit 5` — 스모크 테스트용. 전체 실행 시 주석 처리 (현재 `_2.sh`에만 켜져 있음)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — OOM 회피용으로 붙여둔 것

결과는 `./logs/<모델명>/<타임스탬프>_results.json`에 쌓입니다.
**`logs/`는 `.gitignore` 대상이라 커밋되지 않습니다.** 서버를 옮길 때 결과가 필요하면 따로 복사하세요.

---

## 4. 태스크 인벤토리

이 브랜치가 추가한 태스크 목록과 현재 상태입니다.

| 태스크 ID | 데이터 출처 | 채점 방식 | 상태 |
|---|---|---|---|
| `MolParse` | HF `InnovatorLab/MolParse` | LLM judge (API) | ✅ 전체 실행 완료 |
| `OpenRxn` | HF `InnovatorLab/OpenRxn` | LLM judge (API) | ✅ 전체 실행 완료 |
| `EMVista` | HF `InnovatorLab/EMVista` | exact match | ✅ 전체 실행 완료 |
| `rxnbench_vqa_en` / `_zh` | HF `UniParser/RxnBench` (`RxnBench-VQA`) | exact match | ✅ 전체 실행 완료 |
| `rxnbench_vqa` (그룹) | 위 둘 묶음 | — | ⚠️ 그룹 집계 결과가 비어서(`{}`) 나옴. 개별 태스크로 돌리는 중 |
| `msearth_mcq` | HF `MSEarth/MSEarth_MCQ` | accuracy | ✅ 전체 실행 완료 |
| `microvqa` | HF `jmhb/microvqa` | accuracy | ✅ 전체 실행 완료 — **단 구현이 교체됨, 5절 참고** |
| `ProteinLMBench` | HF `tsynbio/ProteinLMBench` (`evaluation`, gated) | accuracy | ✅ 전체 실행 완료(SciEagle-Granite v261021, 9절) |
| `livexiv_vqa_v0` ~ `v7` / `livexiv_tqa_v0` ~ `v7` | HF `LiveXiv/LiveXiv` (MIT, 비게이트) | 정답 글자 비교(4지선다), 외부 API 불필요 | ⚠️ 스모크만(8~48문항). **전체 실행 미완, 10절 참고** |
| `sfe-en` / `sfe-zh` | HF `etri-lirs/SFE` | LLM judge (API) + ROUGE/BERTScore/METEOR | ❌ 미실행 |
| `superchem_en` / `superchem_cn` | 로컬 parquet | — | ❌ 미실행, **데이터 경로 미해결 (6절)** |
| `SmolInstruct` (그룹, 14개 하위 태스크) | 로컬 jsonl | 태스크별 상이 | ❌ 미실행, **데이터 전처리 필요 (6절)** |

`SmolInstruct` 하위 태스크 14종: `smol_forward_synthesis`, `smol_retrosynthesis`, `smol_molecule_captioning`, `smol_molecule_generation`, `smol_name_conversion-{i2f,i2s,s2f,s2i}`, `smol_property_prediction-{esol,lipo,bbbp,clintox,hiv,sider}`

### 참고 baseline (`Qwen/Qwen3.5-9B`, 전체 실행)

이어받는 쪽에서 회귀 여부를 판단할 기준값입니다.

| 태스크 | 지표 | 값 |
|---|---|---|
| `scienceqa` | exact_match | 0.9283 |
| `rxnbench_vqa_en` | exact_match | 0.8807 |
| `rxnbench_vqa_zh` | exact_match | 0.8728 |
| `msearth_mcq` | accuracy | 0.5675 |
| `microvqa` | accuracy | 0.5547 |
| `xlrs-lite` | micro / macro | 0.4455 / 0.4729 |
| `EMVista` | exact_match_accuracy | 0.4389 |
| `OpenRxn` | api_judge_accuracy | 0.2350 |
| `MolParse` | api_judge_accuracy | 0.1295 |

---

## 5. 환경변수

LLM judge를 쓰는 태스크는 환경변수 설정 없이는 점수가 무의미합니다. `.env`는 gitignore 대상이라 저장소에 없으니 **새 서버에서 직접 만들어야 합니다.**

### `MolParse`, `OpenRxn`

| 변수 | 기본값 | 비고 |
|---|---|---|
| `OPENAI_API_KEY` | `"EMPTY"` | |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | 기본값이 로컬이라, **judge용 OpenAI 호환 서버(vLLM 등)를 별도로 띄워 쓰던 구조로 보입니다** |
| `OPENAI_MODEL_NAME` | `"default-model"` | |

> ⚠️ 기존 서버에서 어떤 judge 모델/서버를 띄웠는지는 저장소에 기록이 없습니다. 위 세 값을 이전 담당자에게 확인하지 않으면 MolParse/OpenRxn 점수를 그대로 재현할 수 없습니다.

### `sfe-en`, `sfe-zh`

| 변수 | 기본값 |
|---|---|
| `API_TYPE` | `openai` (또는 `azure`) |
| `MODEL_VERSION` | `gpt-4o-2024-08-06` |
| `FILE_NAME` | `sfe_test.json` |
| `OPENAI_API_BASE` / `OPENAI_API_KEY` | `https://api.openai.com/v1` / — |
| `AZURE_ENDPOINT` / `AZURE_API_KEY` | (`API_TYPE=azure`일 때) |

### 공통

| 변수 | 용도 |
|---|---|
| `HF_HOME` | `rxnbench_vqa`가 이미지 캐시 경로 확인에 사용 (기본 `~/.cache/huggingface/`) |
| `HF_TOKEN` | `ProteinLMBench`가 gated 데이터셋이라 `dataset_kwargs.token: True`로 설정됨. HF 로그인 필요 |

---

## 6. 미해결 이슈 / 다음 할 일

우선순위 순입니다.

1. **`SmolInstruct` 데이터 준비** — 각 yaml이 `dataset_path: json` + 상대경로 jsonl(`molecule_captioning.jsonl` 등)을 참조합니다. `lmms_eval/tasks/SmolInstruct/process_data.py`로 생성하는 구조인데, 그 스크립트의 `snapshot_path`가 **빈 문자열**이라 그대로는 동작하지 않습니다. HF 스냅샷 경로를 채우고 실행 → 생성된 `./processed_smol_test_data` 경로를 14개 yaml의 `data_files`에 반영해야 합니다.
2. **`superchem` 데이터 경로 수정** — `dataset_path: "parquet"`인데 `data_files.test`가 `"ZehuaZhao/SUPERChem"`(HF repo id)로 들어가 있습니다. parquet 로더에 repo id를 넘기는 형태라 로컬 parquet 파일 경로로 바꾸거나 `dataset_path`를 HF 방식으로 고쳐야 합니다.
3. **`sfe-en` / `sfe-zh` 첫 실행** — judge API 키 설정 후 `--limit 5`로 스모크 테스트부터.
4. ~~**`ProteinLMBench` 전체 실행**~~ — 2026-10-21 SciEagle-Granite v261021 로 전체 실행 완료(accuracy 0.5890, n=944, 9절).
5. **`rxnbench_vqa` 그룹 집계** — 그룹으로 돌리면 결과가 `{}`로 비어 나옵니다. 그룹 aggregation 설정 확인 필요. 당장은 `rxnbench_vqa_en`, `rxnbench_vqa_zh`를 개별로 돌리면 됩니다.

6. **LiveXiv 전체 평가** — `livexiv_vqa_v0~v7`, `livexiv_tqa_v0~v7` 를 모델당 한 번 돌린다(문항 약 12만 5천 개, 10절). 스모크만 확인됐다.
7. **`rxnbench_vqa` 그룹 집계 재확인** — 위 5번 증상은 여러 태스크를 한 프로세스에 넣을 때 생기던 래퍼 버그(9.3절, 수정됨)와 관련이 있을 수 있다. 수정 후 그룹 실행이 정상인지 다시 확인할 것(아직 시험하지 않았다).

---

## 7. 인수인계 시 반드시 알아야 할 함정

### 7.1 `microvqa` 구현이 upstream 것으로 교체됨

upstream이 `microvqa`를 독립적으로 추가하면서 add/add 충돌이 났고, **upstream 버전을 채택**했습니다. 두 구현은 평가 프로토콜이 다릅니다.

| | 현재(upstream) | 이전(science) |
|---|---|---|
| 선택지 표기 | 숫자 `(1) (2)` | 알파벳 `(A) (B)` |
| 정답 파싱 | 숫자만 | 숫자·알파벳 + 단일문자 fallback |
| 모드 | 단일 | `stage1` / `noimage` / `noimage_v2` / `noimage_v3` |
| 샘플링 | temp 1.0, do_sample, seed 0 | temp 0.0 greedy |

→ **머지 이전에 뽑은 microvqa 결과와 이후 결과를 직접 비교하면 안 됩니다.** 위 baseline 표의 0.5547은 교체 이전 값입니다. 이전 구현이 필요하면 `git show cb518730:lmms_eval/tasks/microvqa/utils.py`로 꺼낼 수 있습니다.

### 7.2 Qwen3-VL pixel budget 계산 변경

upstream PR #1490, #1502가 Qwen3-VL의 visual token budget 계산을 수정했습니다. 실행 스크립트가 `MAX_PIXELS=$((2048*32*32))`, `MIN_PIXELS=$((256*32*32))`를 넘기고 있어, **적용되는 실제 visual token 수가 이전 실행과 달라질 수 있습니다.**

사용 중인 모델 로더는 `--model qwen3_5`이고 수정된 파일은 `lmms_eval/models/simple/qwen3_vl.py`라 서로 다른 파일이지만, 공유 로직 여부는 실제 실행으로 확인하는 것이 확실합니다.

### 7.3 결과 로그는 저장소에 없음

`logs/`가 gitignore 대상입니다. 위 baseline 표가 남아 있는 유일한 기록이므로, 서버 이전 시 원본이 필요하면 `logs/` 디렉터리를 별도로 옮기세요.

---

## 8. upstream 최신화 절차

정기적으로 이렇게 따라잡습니다.

```bash
git fetch origin
git branch -f main origin/main        # 로컬 main은 항상 origin/main과 동일하게 유지
git checkout science
git merge origin/main                 # rebase 아님 — 이미 push된 브랜치이므로
# 충돌 해결 후
git commit
git push origin science
```

지금까지 충돌이 난 곳은 `lmms_eval/tasks/microvqa/`뿐입니다. `pyproject.toml`은 매번 자동 병합됐지만, 머지 후 이 브랜치가 추가한 의존성이 살아있는지 확인하는 습관을 권합니다.

```bash
grep -E 'pymeteor|rouge_score|bert_score|causal-conv1d|flash-linear-attention' pyproject.toml
```


---

## 9. SciEagle-Granite-VL 평가 (2026-10)

대상은 HF `etri-lirs/scieagle-granite-9b-preview` 태그 `v261021`(비공개, Stage 3 최종본)이다. 이 모델은 `trust_remote_code=True` 로 단독 구동된다.

### 9.1 래퍼와 실행

`--model scieagle_granite` (`lmms_eval/models/chat/scieagle_granite.py`). 범용 `huggingface` 챗 래퍼를 상속하고 네 가지를 보정한다.

1. `config.model_type` 이 transformers 에 등록돼 있지 않아 `AutoModel` 로 불러 실패 → `AutoModelForImageTextToText` 로 강제.
2. 래퍼가 처리기에 넘기는 빈 `videos`/`audios` 를 처리기가 받지 못함 → 빈 인자는 빼고 호출.
3. 처리기가 일반 `dict` 를 돌려주는데 래퍼는 `.to(device)` 를 부름 → `BatchFeature` 로 감쌈.
4. 체크포인트의 `padding_side` 가 `right` 인데 디코더의 배치 생성은 왼쪽 패딩이어야 함 → 평가 래퍼에서만 `left` 로 강제. 안 하면 `batch_size > 1` 에서 짧은 프롬프트의 응답이 깨진다(chartqa 8문항 배치 4: 정답률 0.125, 배치 1: 0.75).

```bash
# 태스크마다 별도 프로세스로 실행한다(9.3 참고). `uv run` 은 쓰지 않는다(9.2 참고).
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m lmms_eval \
  --model scieagle_granite \
  --model_args pretrained=<모델 경로>,trust_remote_code=True,max_num_frames=1 \
  --tasks chartqa --batch_size 4 \
  --gen_kwargs max_new_tokens=512,temperature=0 \
  --log_samples --output_path ./logs/chartqa
```

`--gen_kwargs` 는 태스크 yaml 의 `generation_kwargs` 를 덮어쓴다(chartqa 는 yaml 기본 16 → 실효 512, 결과 json 의 `configs` 에서 확인).

### 9.2 환경 (RTX PRO 6000 Blackwell, CUDA 12.9 서버)

- `uv sync` 는 `causal-conv1d` 를 소스 빌드하다 실패한다(서버 CUDA 12.9 ↔ PyTorch cu130). 이 모델에는 필요 없으므로 `uv sync --no-install-package causal-conv1d --no-install-package flash-linear-attention`.
- 이 모델은 `transformers==5.8.1` 로 학습·내보내졌다. `uv sync` 가 더 최신(5.20.1)을 깔면 모델 코드의 `create_causal_mask(cache_position=...)` 에서 실패한다. 환경에서 `uv pip install --python .venv/bin/python transformers==5.8.1` 로 맞추거나, 업로드된 `v261021` 의 모델 코드(양쪽 버전에서 동작하도록 수정됨)를 쓴다.
- `uv run` 은 실행 때마다 환경을 다시 동기화해서 위에서 뺀 패키지를 또 빌드하려 한다. `.venv/bin/python -m lmms_eval` 로 직접 실행한다.

### 9.3 함정

1. **태스크 하나당 프로세스 하나.** 범용 래퍼가 한 배치의 모든 문서에 **첫 문서의 `doc_to_messages`** 를 쓰던 버그가 있었다(`doc_to_messages[0]`). 여러 태스크를 한 프로세스에 넣으면 태스크가 섞인 배치에서 엉뚱한 변환 함수가 적용된다(`rxnbench_vqa_en` + `ProteinLMBench` 가 `KeyError: 'image'` 로 실패). 이 브랜치에서 문서마다 자기 함수를 쓰도록 고쳤다(수정 전 재현·수정 후 통과 확인). 그래도 태스크별 프로세스가 안전하다. 수정 전에 묶어 돌린 `chartqa`/`microvqa`/`msearth_mcq`/`EMVista` 는 개별 재실행과 소수점 4자리까지 같았다.
2. **lmms-eval 은 오류를 내부에서 잡고 종료 코드 0 으로 끝난다.** 성공 여부는 종료 코드가 아니라 로그의 `Error during evaluation` 유무와 `results.json` 존재로 확인한다. 로그를 `grep` 할 때 바이너리로 취급되면 `-a` 를 쓴다.
3. **응답 형식.** Stage 3 모델은 서술형 추론 뒤에 `Answer: X` 를 붙이는 경우가 많다. `max_new_tokens` 가 모자라면 `Answer` 앞에서 잘린다. OCRBench 는 사용자 결정으로 태스크 기본값 128 을 쓴다.

### 9.4 결과 (v261021, 단일 태스크 프로세스, 2026-10-21)

설정: bf16, transformers 5.8.1, 배치 4(왼쪽 패딩), `max_new_tokens=512`(OCRBench 만 128), `temperature=0`, GPU 1장/프로세스.

| 태스크 | 표본 | 지표 | 점수 | 표준오차 |
|---|---:|---|---:|---:|
| `docvqa_val` | 5,349 | ANLS | 0.9005 | 0.0037 |
| `ocrbench` | 1,000 | accuracy | 0.6660 | 이항 근사 약 0.015 |
| `chartqa` | 2,500 | relaxed overall | 0.8240 | 0.0076 |
| `scienceqa` (전체) | 4,241 | exact match | 0.8809 | 0.0050 |
| `scienceqa_img` | 2,017 | exact match | 0.8523 | 0.0079 |
| `rxnbench_vqa_en` | 1,525 | exact match | 0.7659 | 0.0108 |
| `rxnbench_vqa_zh` | 1,525 | exact match | 0.7744 | 0.0107 |
| `ProteinLMBench` | 944 | accuracy | 0.5890 | 0.0160 |
| `msearth_mcq` | 2,784 | accuracy | 0.5420 | 0.0094 |
| `EMVista` | 1,850 | exact match | 0.3822 | 0.0113 |
| `microvqa` | 1,042 | accuracy | 0.4213 | 0.0153 |

`mathverse_testmini` 는 외부 API(LLM 심판)가 필요해 제외했다. OCRBench 를 `max_new_tokens=2048` 로 돌리면 0.6740(128 토큰 상한 도달 96건 → 0건)이고 차이 0.008 은 노이즈 범위다.

`rxnbench_vqa_en` 점수는 이미지 기여분만의 값이 아니다. 이미지를 흰 그림으로 바꿔 같은 1,525문항을 풀리면 **0.6184**(이미지 있음 0.7659, 우연 수준 0.25)가 나온다. 문항의 약 62% 는 질문과 선택지만으로 풀리고 이미지의 기여분은 약 15%p 다. 채점 문제는 아니다: 응답 1,525개가 전부 글자 하나이고 이 태스크의 채점(첫 a-d 글자 허용)·정확 일치·명시적 글자 세 방식이 모두 0.7659 로 같다. 다른 모델과 비교할 때는 텍스트만 풀었을 때의 값도 같이 볼 것.

---

## 10. LiveXiv (`livexiv_vqa_v0~v7`, `livexiv_tqa_v0~v7`)

### 10.1 벤치마크

arXiv 논문의 그림(VQA)·표(TQA)로 만든 4지선다이고, 새 논문이 나올 때마다 버전이 추가된다(논문: arXiv:2410.10783, ICLR 2025). 코드와 "효율적 평가" 노트북은 `NimrodShabtay/LiveXiv`, 데이터는 HF `LiveXiv/LiveXiv`(MIT, 비게이트, 약 35GB)다. 채점은 정답 글자 비교이므로 **외부 API 가 필요 없다**(API 는 문제를 새로 만들 때만 쓰인다).

| 버전 | HF 설정(VQA / TQA) | 문항 수(VQA / TQA) |
|---|---|---|
| v0 | `v0-VQA` / `v0-TQA` | 2,067 / 2,375 |
| v1 | `VQA-2024-09-21` / `TQA-2024-09-21` | 7,328 / 9,000 |
| v2 | `VQA-2024-10-26` / `TQA-2024-10-26` | 10,375 / 7,712 |
| v3 | `v3-VQA` / `v3-TQA` | 9,942 / 6,653 |
| v4 | `v4-VQA` / `v4-TQA` | 10,804 / 7,913 |
| v5 | `v5-VQA` / `v5-TQA` | 11,114 / 6,157 |
| v6 | `v6-VQA` / `v6-TQA` | 9,783 / 8,402 |
| v7 | `v7-VQA` / `v7-TQA` | 5,120 / 10,131 |

### 10.2 이 브랜치에서 한 일

`v1~v6` 은 이미 있었다(프롬프트가 원 저장소의 `Answer with the option's letter from the given choices directly.` 와 같고 버전 매핑도 맞다). 이번에 다음을 바꿨다.

1. `v0`, `v7` 태스크와 그룹 항목 추가.
2. `v0` 는 `dataset_path: parquet` + `data_files` 로 직접 읽는다. 데이터셋 카드가 선언한 열 목록에 `image_path` 가 없는데 `v0` 파일에는 있어서 `datasets` 가 `CastError` 로 실패하기 때문이다(`v1` 이후 파일에는 이 열이 없다).
3. `doc_to_target: "answer"` 는 필드 참조가 아니라 상수여서 샘플 로그의 `target` 이 문자열 `answer` 로 찍혔다 → `"{{gt}}"`. 점수에는 영향이 없었다.
4. **`livexiv_{vqa,tqa}_lenient` 지표 추가.** 기존 지표(`livexiv_vqa`, `livexiv_tqa`)는 응답의 첫 글자를 답으로 보는데, 서술형으로 답하는 모델(9.3절 3번)은 `The …` 의 `T` 가 예측이 되어 오답 처리된다. 새 지표는 글자만 있는 응답, 마지막 `Answer: X`, `answer is X`/`option X`, 맨 앞의 `X.` 순으로 글자를 뽑는다. 기존 지표는 그대로 두었으므로 공개 리더보드와 비교할 때는 기존 지표를 쓴다. 시험(v7 VQA 48문항, 512토큰): 기존 0.6667, 새 지표 0.8125.

### 10.3 알아 둘 점

- **비용.** 전체는 약 12만 5천 문항이다. 버전마다 3~5GB 를 받으니, 디스크가 좁으면 버전별로 돌리고 `~/.cache/huggingface/datasets` 를 정리한다. 처리량은 아직 재지 않았다.
- **표준오차가 `N/A`** 로 나온다(사용자 정의 집계). 문항별 0/1 을 돌려주도록 바꾸면 계산할 수 있다.
- **분야별 분석 불가.** 논문 ID(`image_path`)는 `v0` 에만 있다.
- **학습 데이터 겹침.** SciEagle-Granite 로 확인한 값: `v0`(2020년 1~2월 논문 96편)의 **50%(48편)가 ArxivCap 에 있다.** ArxivCap 의 범위는 2007-07 ~ 2023-06 이다. `v1~v7` 은 논문 ID 가 없어 직접 측정하지 못했으며, 2024년 이후 논문이라 ArxivCap 과는 겹치지 않을 가능성이 높다(OmniScience 등 다른 학습 데이터는 미확인). 사용자는 질문(QA)이 학습 데이터와 다르므로 허용 가능하다고 판단했다(2026-10-21).
- **상태.** `v0`(각 8문항), `v1`(각 8문항), `v7` VQA(8·48문항) 스모크만 했다. 전체 실행은 하지 않았다.
