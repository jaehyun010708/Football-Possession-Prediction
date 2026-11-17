# Football Possession Prediction

**IDSSE tracking 데이터 기반 축구 점유권(H/A/D) 예측 파이프라인**

선수 위치만으로 공의 생존 여부(Alive/Dead)를 추정하고, 관측 가능한 공 위치와 선수 움직임을 결합해 점유 팀(Home/Away)을 예측합니다. 프레임 단위 예측에 시간적 후처리를 적용해 연속적인 점유 구간(*possession spells*)을 생성합니다.

이 저장소는 **IDSSE 데이터 전처리 → 모델 학습 → 경기 단위 검증**을 재현하기 위한 독립 프로젝트입니다. Barcelona ATD 원본 데이터나 대회용 예측 파일은 필요하지 않습니다.

## 1. Problem & approach

공이 작거나 선수에 가려지면 트래킹 데이터에서 공 위치가 누락될 수 있습니다. 동시에 점유권 전환 시점을 프레임별로만 판단하면 매우 짧은 구간이 과도하게 생성되기 쉽습니다. 이를 해결하기 위해 Alive/Dead 판별과 Alive 구간 내 H/A 분류를 나누었습니다.

```text
IDSSE Tracking / Metadata / Events (25 FPS)
             │
             ▼
     5 FPS alignment + GT relabeling
             │
       ┌─────┴───────────────┐
       ▼                     ▼
 Player-only AD-V1       AD-Tree (347 features)
 (Attention + GRU)       (HistGradientBoosting)
       └─────────┬───────────┘
                 ▼
          P(Dead) ensemble
                 │
                 ▼
       Missing-ball simulation
                 │
                 ▼
       E4HA (Attention + BiGRU)
       P(Home | Alive)
                 │
                 ▼
    P(H), P(A), P(D) fusion
                 │
                 ▼
          3-state Viterbi
                 │
                 ▼
     Frame predictions + SpellF1 proxy
```

최종 확률은 다음과 같이 구성됩니다.

- `P(D)`: V1과 HGB 모델이 예측한 Dead 확률의 평균
- `P(H) = (1 - P(D)) × P(H | Alive)`
- `P(A) = (1 - P(D)) × (1 - P(H | Alive))`

기본 Viterbi 전환 비용은 H↔A `2.0`, Alive↔Dead `2.5`입니다. 이 값들은 실험 설정이며 데이터에 따라 재검증이 필요합니다.

## 2. Dataset & labels

- **Dataset:** [PySport IDSSE / Sportec tracking](https://huggingface.co/datasets/pysport/idsse-data)
- **입력:** 경기별 Tracking XML, Match Information XML, Event XML (세 종류)
- **경기:** `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WQQ`, `J03WR9`, `J03WPY`
- **Sampling:** 원본 25 FPS → 5 FPS
- **타깃:** `H=0` (Home), `A=1` (Away), `D=2` (Dead ball)
- **기본 분할:** 6경기 학습, `J03WPY` 1경기 검증 (match-level holdout)

전처리는 `period`와 `frame`을 기준으로 선수·공 정보를 정확하게 맞추며, 원본 GT(`gt_orig`) 외에 이벤트 규칙으로 조정한 `gt_mid` 및 `gt_loose`를 저장합니다.

**중요:** V1과 E4HA는 `gt_mid`의 라벨로 학습하고, AD-Tree의 Dead 타깃은 `gt_orig`를 사용합니다. 따라서 이 저장소의 기본 검증 점수는 *원본 GT가 아닌 `gt_mid` 기준*입니다.

## 3. Models

| 단계 | 구현 | 역할 |
| --- | --- | --- |
| AD-V1 | Player Attention + GRU | 선수 좌표·이동 정보로 H/A/D 확률 예측 |
| AD-Tree | HistGradientBoosting, 347 features | 선수 배치·상대 관계 등의 특징으로 Dead 확률 예측 |
| AD Ensemble | 두 모델의 Dead 확률 평균 | Alive/Dead 확률 결합 |
| Missing-ball augmentation | 연속 결측 구간 시뮬레이션 | 공 관측이 없는 상황을 학습에 반영 |
| E4HA | Player Set Attention + BiGRU | Alive 조건부 Home/Away 확률 예측 |
| Temporal decoder | 3-state Viterbi | 짧고 불안정한 예측 전환 완화 |

공 결측 증강은 연속 결측 길이 분포를 사용해 **Alive 구간에서 최종 결측률 약 10.6%를 목표**로 생성합니다. 원래 관측되지 않은 공 좌표는 복원하거나 임의로 채우지 않습니다. 기본 `--mask-mode fixed`를 사용하며, `dynamic` 옵션도 지원합니다.

검증 경기에서 결측 마스크를 만드는 기준은 **학습된 AD 모델의 예측**이고 검증 경기의 GT Alive를 참조하지 않습니다. 모델 학습을 위한 라벨과 검증 지표 계산에는 해당 GT가 사용됩니다.

## 4. Validation results

아래 수치는 `J03WPY`를 제외한 6경기로 학습한 후 **사용자 로컬 실행에서 보고된 한 차례의 검증 결과**입니다. 저장소에 학습 가중치나 원본 데이터는 포함하지 않으므로, 아래 숫자는 환경·설정·학습 실행에 따라 달라질 수 있습니다.

| Metric | Result |
| --- | ---: |
| Frame accuracy (`gt_mid`) | **0.94159** |
| Diagnostic SpellF1 proxy | **0.79723** |
| GT H/A spells | 214 |
| Predicted H/A spells | 219 |

| IoU threshold | H/A spell F1 |
| --- | ---: |
| 0.3 | 0.8730 |
| 0.4 | 0.8499 |
| 0.5 | 0.8083 |
| 0.6 | 0.7667 |
| 0.7 | 0.6882 |

> **Metric caveat:** `spell_f1_proxy`는 `gt_mid`를 기준으로 전·후반별 H/A 구간을 만들고, 각 팀에 대해 IoU 임계값 `0.3~0.7`로 1:1 매칭한 F1을 평균한 **자체 진단 지표**입니다. 대회 공식 SpellF1과 동일한 지표라고 해석하면 안 됩니다. 또한 이벤트 재라벨링은 평가 구간 수와 경계 난이도를 바꿀 수 있으므로 원본 GT 기반 결과와 직접 비교할 수 없습니다.

## 5. Setup & reproduction

Python **3.13.14** 기준으로 구성되었습니다. Ubuntu/WSL 예시:

```bash
uv python install 3.13.14
uv venv --python 3.13.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step A — 데이터 다운로드 및 전처리

```bash
python run_pipeline.py --mode prepare --download-missing
```

`--download-missing` 옵션은 [Hugging Face의 IDSSE 데이터](https://huggingface.co/datasets/pysport/idsse-data)에서 없는 XML 파일을 내려받으며, 다운로드 크기는 약 **2.6GB**일 수 있습니다. 이미 원본 XML을 갖고 있다면 `data/idsse/`에 넣고 다음처럼 실행할 수 있습니다.

```bash
python run_pipeline.py --mode prepare
```

전처리 결과: `outputs/base/<match_id>.npz`. 다른 위치의 원본을 사용하려면 `--idsse-dir /absolute/path`를 지정합니다.

### Step B — 6경기 학습 / 1경기 검증

```bash
python run_pipeline.py --mode validate
```

1. 학습 6경기에 대해 AD-V1과 AD-Tree 학습
2. 홀드아웃 경기의 AD 확률 예측
3. 학습·검증 경기별 결측 마스크 생성
4. E4HA 학습 및 홀드아웃 검증
5. AD/E4HA 결합 → Viterbi → 프레임 정확도 및 SpellF1 proxy 산출

주요 결과 파일:

```text
outputs/
├── base/<match_id>.npz
├── models/validate/
│   ├── v1_mid.pt
│   ├── tree.joblib
│   └── e4ha_mid.pt
├── predictions/J03WPY_validate.npz
└── predictions/J03WPY_metrics.json
```

빠른 기능 점검용으로만 사용할 명령어 (실제 성능 측정용이 아님):

```bash
python run_pipeline.py --mode validate --epochs-v1 1 --epochs-e4 1 --windows-per-epoch 64
```

다른 검증 경기를 지정하려면 `--val-match J03WR9`처럼 변경할 수 있습니다. 한 번에 검증과 7경기 전체 재학습까지 실행하려면 `--mode all`을 사용합니다.

### Step C — 7경기 전체 학습 (선택)

```bash
python run_pipeline.py --mode fit-all
```

가능하면 앞선 검증에서 선택한 epoch 수를 사용하고, 그렇지 않으면 명령행 기본 epoch 수를 사용합니다. 결과는 `outputs/models/fit-all/`에 저장됩니다. `fit-all`은 **홀드아웃 평가가 아니라 최종 학습**이므로 이 단계에서 얻은 값은 독립 검증 점수가 아닙니다.

### Tests

```bash
python -m unittest discover -s tests -v
```

포함된 테스트는 형태·프레임 매칭·결측 생성·디코더의 기본 동작을 점검하는 **단위/스모크 테스트**입니다. 모든 실제 IDSSE 경기의 학습 성능을 보장하지는 않습니다.

## 6. Directory structure

```text
Barcelona_AutoPipeline/
├── data/idsse/                # original IDSSE XMLs (not tracked by Git)
├── preprocessing/            # tracking/event parsing & GT relabeling
├── ad_v1/                    # player-only neural model
├── ad_tree/                  # 347-feature dead-ball tree
├── ad_ensemble/              # P(Dead) fusion
├── ball_missing/             # contiguous ball-missing augmentation
├── e4ha/                     # ball + players sequence model
├── inference/                # decoding & IDSSE holdout validation
├── third_party/              # reused model implementations
├── tests/                    # unit/smoke tests
├── settings.py
├── run_pipeline.py           # prepare / validate / fit-all / all
├── requirements.txt
└── NOTICE.md                 # code provenance
```

`outputs/`, 학습 모델, 데이터 파일, `.venv/`는 `.gitignore`로 Git 추적에서 제외합니다. 기존 결과를 재현하려면 실제 데이터와 모델 학습을 별도로 수행해야 합니다. 기존 캐시는 입력이나 전처리 로직이 변경되어도 자동으로 무효화되지 않으므로, 변경 시 관련 `outputs/` 캐시를 삭제한 뒤 다시 실행해야 합니다.

## 7. Limitations & provenance

- **Label dependence:** `gt_mid`는 이벤트 기반 휴리스틱으로 원본 점유 라벨을 재구성하므로, 성능 평가는 그 라벨 정의에 종속됩니다.
- **Generalization:** 1경기 홀드아웃 결과만으로 다른 리그·구단·트래킹 공급업체의 성능을 보장할 수 없습니다.
- **Reproducibility:** PyTorch 학습 환경, 하이퍼파라미터, 캐시 상태와 난수성에 따라 결과가 달라질 수 있습니다.
- **Model differences:** 이 리팩터링은 원본 팀 제출 코드의 `sub_13`과 완전히 동일한 구성은 아닙니다. AD-Tree는 5 FPS 347-feature HGB 변형을 사용합니다.
- **Scope:** 이 저장소는 IDSSE 학습·검증 프로젝트이며 대회 데이터 추론·영상 평가 도구는 포함하지 않습니다.

이 프로젝트는 기존 협업 프로젝트의 모델 구현과 전처리 로직을 일부 **재사용·리팩터링**했습니다. 특히 `third_party/pa_sm/`, `third_party/models_new.py`, Tree 특징 생성기, 이벤트 기반 재라벨링의 출처는 [NOTICE.md](NOTICE.md)에 정리되어 있습니다. 공 결측 증강 및 IDSSE 자동 파이프라인 통합도 그 위에 구성되어 있습니다. **모든 모델 구성 요소를 개인이 독자적으로 처음 개발한 것으로 주장하지 않습니다.**

원본 코드의 재배포 권한과 적용 라이선스를 확인한 후 공개 저장소에 게시해야 합니다. 참고 데이터셋: [PySport IDSSE](https://huggingface.co/datasets/pysport/idsse-data).
