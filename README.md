# test-mlx-llm

iPhone에서 **완전 온디바이스**로 LLM을 실행하는 iOS 앱.  
Apple의 [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 프레임워크를 사용해 Metal GPU 가속 추론을 제공합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **31종 모델 지원** | 초고성능(14B~30B) / 고성능(7B~13B) / 중형(0.3B~4B) 3단계 티어 |
| **개별 다운로드·삭제** | 모델별 HuggingFace 캐시 다운로드 / 삭제 (용량 표시) |
| **백그라운드 다운로드** | 모델 A를 사용하면서 모델 B를 백그라운드에서 다운로드 |
| **Think 모드** | Qwen3·DeepSeek R1·AceReason·Jamba 계열 추론 토글 |
| **Markdown 렌더링** | 볼드·이탤릭·헤더·코드블록 등 실시간 렌더링 |
| **URL 뉴스 요약** | 주소 입력 시 본문 자동 추출 → LLM 요약 |
| **한국어 기본 프롬프트** | 모든 답변이 한국어로 출력 |
| **Liquid Glass 버튼** | iOS 26+ 전송·피커 확인 버튼에 유리 재질 효과 |

---

## 지원 모델 (31종)

### 🏆 초고성능

| 모델 | 크기 | Think |
|------|------|-------|
| Qwen3 MoE 30B-A3B | ~15GB | ✅ |
| GPT-OSS 20B | ~12GB | - |
| Baichuan-M1 14B | ~8GB | - |
| Mistral NeMo 12B | ~7GB | - |

### ⚡ 고성능 (7B~13B)

| 모델 | 크기 | Think |
|------|------|-------|
| Qwen3 8B | ~4.5GB | ✅ |
| Qwen3 1.7B | ~1GB | ✅ |
| Qwen2.5 7B | ~4GB | - |
| DeepSeek R1 7B | ~4GB | ✅ |
| DeepSeek R1 (Full) | ~4GB | ✅ |
| Gemma 2 9B | ~5.5GB | - |
| GLM4 9B | ~5.5GB | - |
| Llama 3.1 8B | ~4.5GB | - |
| Llama 3 8B | ~4.5GB | - |
| AceReason 7B | ~4GB | ✅ |
| OLMo-2 7B | ~4GB | - |
| MiMo 7B | ~4GB | - |
| LFM2 8B-A1B (MoE) | ~3GB | - |
| CodeLlama 13B | ~7.5GB | - |
| Mistral 7B | ~4GB | - |

### ✨ 중형 (0.3B~4B)

| 모델 | 크기 | Think |
|------|------|-------|
| EXAONE 4.0 1.2B ★ | ~700MB | - |
| Gemma4 E4B IT | ~2.5GB | - |
| Gemma3n E4B (bf16) | ~8GB | - |
| Gemma3n E4B (4bit) | ~2.5GB | - |
| Qwen3 4B | ~2.3GB | ✅ |
| Phi-2 (2.7B) | ~1.5GB | - |
| Phi-3.5 MoE | ~21GB | - |
| Phi-3.5 Mini (3.8B) | ~2.2GB | - |
| Jamba 3B | ~6GB | ✅ |
| SmolLM3 3B | ~1.8GB | - |
| Llama 3.2 3B | ~1.8GB | - |
| ERNIE 4.5 0.3B | ~0.7GB | - |

> ★ 추천: EXAONE 4.0 1.2B — LG AI Research 한국어 특화 모델, 700MB 크기로 7B급 한국어 성능

---

## 모델 상태 아이콘

| 아이콘 | 상태 |
|--------|------|
| ☁️ | 미다운로드 |
| ⬇️ | 다운로드 중 |
| 📥 | 다운로드됨 (RAM 미로드) |
| 🔄 | 로드 중 |
| ⚡ | 로드됨 (사용 가능) |

---

## 아키텍처

### 소스 파일

```
test-mlx-llm/
├── ViewController.swift     # 메인 UI + LLM 추론 로직
└── ArticleExtractor.swift   # URL 뉴스 본문 추출기
```

### 모델 상태 관리

동시에 독립적인 3가지 상태를 관리합니다:

```
loadedModelIndex   — RAM에서 실행 중인 모델 인덱스
loadingIndex       — 캐시 → RAM 로드 중인 모델 (빠름, ~수 초)
downloadingIndex   — HuggingFace에서 다운로드 중인 모델 (느림, 수 분)
```

**핵심 원칙:**
- 한 번에 하나의 모델만 RAM에 올림
- 다운로드는 `HubClient().downloadSnapshot()`을 사용해 **디스크에만** 저장 (MLX RAM 비점유)
- 이미 캐시된 모델은 다른 모델이 다운로드 중이어도 즉시 로드 가능
- 동시 다운로드 불가 (메모리 보호)

### Load 버튼 동작 (선택 모델 기준)

```
⚡ 로드됨      → [언로드] (빨간색)
🔄 로드 중     → [로드 중...] 비활성화
⬇️ 다운로드 중 → [다운로드 취소] (주황색)
📥 캐시 있음   → [로드] (다른 모델 다운로드 중이어도 활성화)
☁️ 미다운로드  → [다운로드] / [다운로드 중...] (다른 다운로드 진행 시 비활성화)
```

### URL 뉴스 요약 흐름

```
URL 입력
  → HTTP fetch (User-Agent: iPhone Safari)
  → 인코딩 감지 (UTF-8 / EUC-KR 자동)
  → Readability-like 본문 추출
      우선순위: <article> → <main> → content div → <p> 점수 기반
  → 보일러플레이트 제거
      (저작권 고지 / 댓글 정책 / SNS 공유 / 관련기사 등)
  → LLM 요약 프롬프트 생성
  → 스트리밍 응답
```

---

## 빌드 요구사항

- **Xcode** 16.0+
- **iOS** 26.0+ (Liquid Glass UI) / iOS 17.0+ (기본 기능)
- **실기기 필수** — MLX Metal 가속은 시뮬레이터 미지원
- Swift Package 자동 해결 (`mlx-swift-lm`, `swift-transformers`, `swift-huggingface`)

### 실기기 설정

1. Xcode → Signing & Capabilities → Team 설정
2. Build Settings → `IPHONEOS_DEPLOYMENT_TARGET` = `17.0` (디바이스 미인식 시)
3. 빌드 후 설정 → VPN 및 기기 관리 → 개발자 앱 신뢰

---

## 의존 패키지

| 패키지 | 용도 |
|--------|------|
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | MLX 추론 엔진, 모델 팩토리, LLMRegistry |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 토크나이저, HubApi |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | HubClient, 모델 파일 다운로드 |

---

## 기술 노트

### Think 모드
Qwen3 계열은 `additionalContext: ["enable_thinking": false]`로 비활성화.  
Think 미지원 모델에는 이 파라미터가 무시되므로 항상 전달해도 안전합니다.

### 메모리 관리
`loadContainer()`는 다운로드 + MLX 로드를 동시에 수행합니다.  
백그라운드 다운로드 시 `HubClient().downloadSnapshot()`만 호출해 RAM 점유를 방지합니다.  
모델 전환 시 기존 `ModelContainer`를 먼저 nil 처리해 메모리를 확보한 뒤 새 모델을 로드합니다.

### 모델 캐시 경로
```
Library/Caches/huggingface/hub/models--{org}--{repo}/
```
삭제 시 이 디렉터리를 제거합니다.

### 한국어 인코딩
뉴스 사이트 EUC-KR 감지는 `CFStringConvertIANACharSetNameToEncoding("EUC-KR")`을 사용합니다.  
Content-Type 헤더 → HTML charset 메타태그 → UTF-8 시도 후 EUC-KR 폴백 순서로 처리합니다.

### Markdown 렌더링
`AttributedString(markdown:, options: .init(interpretedSyntax: .full))`로 파싱.  
`<think>...</think>` 블록은 회색 소형 폰트로 별도 렌더링합니다.
