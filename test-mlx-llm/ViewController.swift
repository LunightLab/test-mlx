import UIKit
import Darwin
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers


class ViewController: UIViewController {

    // MARK: - Model Info

    private struct ModelInfo {
        let label: String
        let description: String
        let size: String
        let maxTokens: Int
        let supportsThinking: Bool
        let config: ModelConfiguration
    }

    private let models: [ModelInfo] = [
        // ── 초고성능 ─────────────────────────────────────────────────────
        ModelInfo(label: "Qwen3 MoE 30B-A3B", description: "🏆 초고성능 | MoE 아키텍처로 한국어·금융·요약 압도적 탑. 다운로드 ~15GB, 추론 RAM ~2GB", size: "~15GB", maxTokens: 32_768, supportsThinking: true, config: LLMRegistry.qwen3MoE_30b_a3b_4bit),
        ModelInfo(label: "GPT-OSS 20B", description: "🏆 초고성능 | 20B 체급·높은 비트수로 고난도 문맥 파악 유리. 다운로드 ~12GB", size: "~12GB", maxTokens: 32_768, supportsThinking: false, config: LLMRegistry.gpt_oss_20b_MXFP4_Q8),
        ModelInfo(label: "Baichuan-M1 14B", description: "🏆 초고성능 | 아시아권 언어(한국어·중국어) 및 요약 우수. 다운로드 ~8GB", size: "~8GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.baichuan_m1_14b_instruct_4bit),
        ModelInfo(label: "Mistral NeMo 12B", description: "🏆 초고성능 | 12B 체급, 긴 뉴스 문맥 파악·요약 탁월. 다운로드 ~7GB", size: "~7GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.mistralNeMo4bit),
        // ── 고성능 (7B~13B) ────────────────────────────────────────────
        ModelInfo(label: "Qwen3 8B", description: "⚡ 고성능 | 최신 Qwen3 8B, 한국어·금융 수식/도표 이해 최상위. 다운로드 ~4.5GB", size: "~4.5GB", maxTokens: 16_384, supportsThinking: true, config: LLMRegistry.qwen3_8b_4bit),
        ModelInfo(label: "Qwen3 1.7B", description: "⚡ 고성능 | Qwen3 1.7B 소형. Think 모드 지원. 다운로드 ~1GB", size: "~1GB", maxTokens: 8_192, supportsThinking: true, config: LLMRegistry.qwen3_1_7b_4bit),
        ModelInfo(label: "Qwen2.5 7B", description: "⚡ 고성능 | 전 세대 대장급, 한국어 금융 데이터 정제·요약 매우 안정적. 다운로드 ~4GB", size: "~4GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.qwen2_5_7b),
        ModelInfo(label: "DeepSeek R1 7B", description: "⚡ 고성능 | 추론 특화, 복잡한 증권 시황 분석·인과관계 파악 강력. 다운로드 ~4GB", size: "~4GB", maxTokens: 16_384, supportsThinking: true, config: LLMRegistry.deepSeekR1_7B_4bit),
        ModelInfo(label: "DeepSeek R1 (Full)", description: "⚡ 고성능 | DeepSeek-R1 4bit 전체 모델. 추론 특화. 다운로드 ~4GB", size: "~4GB", maxTokens: 16_384, supportsThinking: true, config: LLMRegistry.deepseek_r1_4bit),
        ModelInfo(label: "Gemma 2 9B", description: "⚡ 고성능 | 구글 9B 모델, 논리적 요약 능력 우수. 다운로드 ~5.5GB", size: "~5.5GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.gemma_2_9b_it_4bit),
        ModelInfo(label: "GLM4 9B", description: "⚡ 고성능 | 다국어·정보 추출 뛰어남, 뉴스 요약 강점. 다운로드 ~5.5GB", size: "~5.5GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.glm4_9b_4bit),
        ModelInfo(label: "Llama 3.1 8B", description: "⚡ 고성능 | 범용성 뛰어나고 한국어 이해도 전작 대비 대폭 향상. 다운로드 ~4.5GB", size: "~4.5GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.llama3_1_8B_4bit),
        ModelInfo(label: "Llama 3 8B", description: "⚡ 고성능 | 검증된 범용 LLM. 다운로드 ~4.5GB", size: "~4.5GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.llama3_8B_4bit),
        ModelInfo(label: "AceReason 7B", description: "⚡ 고성능 | 추론 모델 계열, 증권 분석 유리. 다운로드 ~4GB", size: "~4GB", maxTokens: 16_384, supportsThinking: true, config: LLMRegistry.acereason_7b_4bit),
        ModelInfo(label: "OLMo-2 7B", description: "⚡ 고성능 | Allen AI 오픈 모델, 안정적 추론. 다운로드 ~4GB", size: "~4GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.olmo_2_1124_7B_Instruct_4bit),
        ModelInfo(label: "MiMo 7B", description: "⚡ 고성능 | SFT 튜닝 범용 모델. 다운로드 ~4GB", size: "~4GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.mimo_7b_sft_4bit),
        ModelInfo(label: "LFM2 8B-A1B (MoE)", description: "⚡ 고성능 | 8B MoE 구조로 1B 활성 추론, 속도·성능 밸런스. 다운로드 ~3GB", size: "~3GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.lfm2_8b_a1b_3bit_mlx),
        ModelInfo(label: "CodeLlama 13B", description: "⚡ 고성능 | 코딩 특화이나 13B 체급으로 기본 요약 가능. 다운로드 ~7.5GB", size: "~7.5GB", maxTokens: 16_384, supportsThinking: false, config: LLMRegistry.codeLlama13b4bit),
        ModelInfo(label: "Mistral 7B", description: "⚡ 고성능 | 검증된 유럽산 7B 모델, 범용 추론 안정적. 다운로드 ~4GB", size: "~4GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.mistral7B4bit),
        // ── 중형 (1B~4B) ──────────────────────────────────────────────
        ModelInfo(label: "EXAONE 4.0 1.2B ★", description: "✨ 중형 | ★추천: LG 한국어 특화, 1.2B임에도 한국어·금융 성능 7B급. 다운로드 ~700MB", size: "~700MB", maxTokens: 4_096, supportsThinking: false, config: LLMRegistry.exaone_4_0_1_2b_4bit),
        ModelInfo(label: "Gemma4 E4B IT", description: "✨ 중형 | 최신 4B 모델, 모바일 온디바이스 빠른 요약 가능. 다운로드 ~2.5GB", size: "~2.5GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.gemma4_e4b_it_4bit),
        ModelInfo(label: "Gemma3n E4B (bf16)", description: "✨ 중형 | bf16 고정밀도, 품질 우선. 다운로드 ~8GB", size: "~8GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.gemma3n_E4B_it_lm_bf16),
        ModelInfo(label: "Gemma3n E4B (4bit)", description: "✨ 중형 | 4bit 경량화, bf16 대비 용량 절반. 다운로드 ~2.5GB", size: "~2.5GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.gemma3n_E4B_it_lm_4bit),
        ModelInfo(label: "Qwen3 4B", description: "✨ 중형 | 가볍고 정교한 가성비 금융 요약 모델. Think 지원. 다운로드 ~2.3GB", size: "~2.3GB", maxTokens: 8_192, supportsThinking: true, config: LLMRegistry.qwen3_4b_4bit),
        ModelInfo(label: "Phi-2 (2.7B)", description: "✨ 중형 | Microsoft phi-2, 소형임에도 이성적 추론 뛰어남. 다운로드 ~1.5GB", size: "~1.5GB", maxTokens: 4_096, supportsThinking: false, config: LLMRegistry.phi4bit),
        ModelInfo(label: "Phi-3.5 MoE", description: "✨ 중형 | 42B MoE, 6.6B 활성 파라미터. 대용량 다운로드 주의. 다운로드 ~21GB", size: "~21GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.phi3_5MoE),
        ModelInfo(label: "Phi-3.5 Mini (3.8B)", description: "✨ 중형 | phi-3.5 경량 버전, 온디바이스 최적화. 다운로드 ~2.2GB", size: "~2.2GB", maxTokens: 8_192, supportsThinking: false, config: LLMRegistry.phi3_5_4bit),
        ModelInfo(label: "Jamba 3B", description: "✨ 중형 | AI21 추론 특화 3B (bf16). 다운로드 ~6GB", size: "~6GB", maxTokens: 8_192, supportsThinking: true, config: LLMRegistry.jamba_3b),
        ModelInfo(label: "SmolLM3 3B", description: "✨ 중형 | Hugging Face SmolLM3, 경량 범용. 다운로드 ~1.8GB", size: "~1.8GB", maxTokens: 4_096, supportsThinking: false, config: LLMRegistry.smollm3_3b_4bit),
        ModelInfo(label: "Llama 3.2 3B", description: "✨ 중형 | Meta 최신 소형 모델, 빠르고 안정적. 다운로드 ~1.8GB", size: "~1.8GB", maxTokens: 4_096, supportsThinking: false, config: LLMRegistry.llama3_2_3B_4bit),
        ModelInfo(label: "ERNIE 4.5 0.3B", description: "✨ 중형 | Baidu ERNIE 초소형(0.3B) bf16, 빠른 응답. 다운로드 ~0.7GB", size: "~0.7GB", maxTokens: 2_048, supportsThinking: false, config: LLMRegistry.ernie_45_0_3BPT_bf16_ft),
    ]

    // MARK: - State

    private var selectedIndex = 0
    private var isThinkingEnabled = true

    private var modelContainer: ModelContainer?
    private var loadedModelIndex: Int?     // currently active model (in RAM)
    private var loadingIndex: Int?         // loading from cache → RAM
    private var downloadingIndex: Int?     // downloading from HuggingFace

    private var loadTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?
    private var generateTask: Task<Void, Never>?
    private var compareTask: Task<Void, Never>?
    private var isComparing = false
    private var isGenerating = false
    private var isBusy: Bool { isGenerating || isComparing }
    private var accumulatedText = ""
    private var lastProgressUpdate = Date.distantPast
    // 마지막으로 LLM에 전송한 실제 입력값 (입력 확인 기능용)
    private var lastSentSystem = ""
    private var lastSentUser   = ""

    private let systemPrompt = "당신은 친절한 AI 어시스턴트입니다. 모든 답변은 반드시 한국어로 작성해 주세요."

    // MARK: - UI

    private lazy var modelPickerField: UITextField = {
        let tf = UITextField()
        tf.borderStyle = .roundedRect
        tf.tintColor = .clear
        tf.inputView = modelPickerView
        tf.inputAccessoryView = pickerToolbar
        let chevron = UIImageView(image: UIImage(systemName: "chevron.up.chevron.down"))
        chevron.tintColor = .secondaryLabel
        tf.rightView = chevron
        tf.rightViewMode = .always
        return tf
    }()

    private lazy var modelPickerView: UIPickerView = {
        let pv = UIPickerView()
        pv.delegate = self
        pv.dataSource = self
        return pv
    }()

    private lazy var pickerDoneButton: UIButton = {
        var cfg = UIButton.Configuration.plain()
        cfg.title = "선택 완료"
        let btn = UIButton(configuration: cfg)
        btn.addTarget(self, action: #selector(pickerDone), for: .touchUpInside)
        return btn
    }()

    private lazy var pickerToolbar: UIToolbar = {
        let tb = UIToolbar()
        tb.sizeToFit()
        tb.items = [
            UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil),
            UIBarButtonItem(customView: pickerDoneButton),
        ]
        return tb
    }()

    private lazy var loadButton: UIButton = {
        var cfg = UIButton.Configuration.filled()
        cfg.title = "다운로드"
        let btn = UIButton(configuration: cfg)
        btn.addTarget(self, action: #selector(loadTapped), for: .touchUpInside)
        return btn
    }()

    private lazy var deleteButton: UIButton = {
        var cfg = UIButton.Configuration.filled()
        cfg.image = UIImage(systemName: "trash")
        cfg.baseBackgroundColor = .systemRed
        cfg.contentInsets = NSDirectionalEdgeInsets(top: 0, leading: 14, bottom: 0, trailing: 14)
        let btn = UIButton(configuration: cfg)
        btn.addTarget(self, action: #selector(deleteTapped), for: .touchUpInside)
        btn.isEnabled = false
        btn.setContentHuggingPriority(.required, for: .horizontal)
        btn.setContentCompressionResistancePriority(.required, for: .horizontal)
        btn.configurationUpdateHandler = { b in b.alpha = b.isEnabled ? 1 : 0.35 }
        return btn
    }()

    private lazy var buttonRow: UIStackView = {
        let stack = UIStackView(arrangedSubviews: [loadButton, deleteButton])
        stack.axis = .horizontal
        stack.spacing = 8
        return stack
    }()

    private lazy var modelDescriptionLabel: UILabel = {
        let lbl = UILabel()
        lbl.font = .systemFont(ofSize: 12)
        lbl.textColor = .secondaryLabel
        lbl.numberOfLines = 0
        lbl.translatesAutoresizingMaskIntoConstraints = false
        return lbl
    }()

    private lazy var descriptionCard: UIView = {
        let card = UIView()
        card.backgroundColor = .secondarySystemBackground
        card.layer.cornerRadius = 8
        card.addSubview(modelDescriptionLabel)
        NSLayoutConstraint.activate([
            modelDescriptionLabel.topAnchor.constraint(equalTo: card.topAnchor, constant: 8),
            modelDescriptionLabel.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -8),
            modelDescriptionLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 10),
            modelDescriptionLabel.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -10),
        ])
        return card
    }()

    private lazy var thinkLabel: UILabel = {
        let lbl = UILabel()
        lbl.font = .systemFont(ofSize: 14)
        lbl.text = "Think 모드"
        return lbl
    }()

    private lazy var thinkSwitch: UISwitch = {
        let sw = UISwitch()
        sw.isOn = true
        sw.addTarget(self, action: #selector(thinkingToggled(_:)), for: .valueChanged)
        return sw
    }()

    private lazy var thinkingRow: UIStackView = {
        let stack = UIStackView(arrangedSubviews: [thinkLabel, UIView(), thinkSwitch])
        stack.axis = .horizontal
        stack.alignment = .center
        return stack
    }()

    private lazy var statsLabel: UILabel = {
        let lbl = UILabel()
        lbl.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        lbl.textColor = .secondaryLabel
        lbl.textAlignment = .center
        lbl.adjustsFontSizeToFitWidth = true
        lbl.minimumScaleFactor = 0.7
        return lbl
    }()

    private var statsTimer: Timer?

    private lazy var progressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .default)
        pv.isHidden = true
        return pv
    }()

    private lazy var statusLabel: UILabel = {
        let lbl = UILabel()
        lbl.text = "모델을 선택하고 다운로드·로드를 눌러주세요"
        lbl.font = .systemFont(ofSize: 12)
        lbl.textColor = .secondaryLabel
        lbl.textAlignment = .center
        lbl.numberOfLines = 2
        return lbl
    }()

    private lazy var inputField: UITextField = {
        let tf = UITextField()
        tf.placeholder = "질문을 입력하세요..."
        tf.borderStyle = .roundedRect
        tf.returnKeyType = .send
        tf.delegate = self
        // 텍스트가 길어져도 버튼을 밀어내지 않도록 압축 허용
        tf.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return tf
    }()

    private lazy var sendButton: UIButton = {
        var cfg = UIButton.Configuration.filled()
        cfg.title = "전송"
        let btn = UIButton(configuration: cfg)
        btn.addTarget(self, action: #selector(sendTapped), for: .touchUpInside)
        btn.isEnabled = false
        btn.setContentHuggingPriority(.required, for: .horizontal)
        // 텍스트가 긴 inputField에 밀려 축소되지 않도록 강제 고정
        btn.setContentCompressionResistancePriority(.required, for: .horizontal)
        btn.configurationUpdateHandler = { b in b.alpha = b.isEnabled ? 1 : 0.4 }
        return btn
    }()

    private lazy var outputTextView: UITextView = {
        let tv = UITextView()
        tv.isEditable = false
        tv.font = .systemFont(ofSize: 14)
        tv.text = "응답이 여기에 표시됩니다."
        tv.textColor = .placeholderText
        tv.layer.borderColor = UIColor.separator.cgColor
        tv.layer.borderWidth = 0.5
        tv.layer.cornerRadius = 8
        return tv
    }()

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "MLX LLM"
        view.backgroundColor = .systemBackground
        setupLayout()
        updateUI()
        startStatsTimer()
        if #available(iOS 26, *) {
            applyGlassEffect(to: sendButton)
            applyGlassEffect(to: pickerDoneButton)
        }
        let longPress = UILongPressGestureRecognizer(target: self, action: #selector(sendLongPressed(_:)))
        longPress.minimumPressDuration = 0.8
        sendButton.addGestureRecognizer(longPress)

        // statusLabel 탭 → 마지막으로 LLM에 보낸 실제 입력값 확인
        statusLabel.isUserInteractionEnabled = true
        statusLabel.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(showLastInput)))
    }

    @objc private func showLastInput() {
        guard !lastSentUser.isEmpty else { return }
        showInputSheet(system: lastSentSystem, user: lastSentUser)
    }

    private func showInputSheet(system: String, user: String) {
        let vc = LLMInputInspectorVC(system: system, user: user)
        let nav = UINavigationController(rootViewController: vc)
        nav.modalPresentationStyle = .pageSheet
        if let sheet = nav.sheetPresentationController {
            sheet.detents = [.medium(), .large()]
            sheet.prefersGrabberVisible = true
        }
        present(nav, animated: true)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        statsTimer?.invalidate()
        statsTimer = nil
    }

    // MARK: - Layout

    private func setupLayout() {
        let inputRow = UIStackView(arrangedSubviews: [inputField, sendButton])
        inputRow.axis = .horizontal
        inputRow.spacing = 8

        let root = UIStackView(arrangedSubviews: [
            modelPickerField,
            buttonRow,
            descriptionCard,
            thinkingRow,
            progressView,
            statusLabel,
            statsLabel,
            inputRow,
            outputTextView,
        ])
        root.axis = .vertical
        root.spacing = 10
        root.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(root)
        NSLayoutConstraint.activate([
            root.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            root.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            root.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            root.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            outputTextView.heightAnchor.constraint(greaterThanOrEqualToConstant: 250),
        ])
    }

    // MARK: - Glass Effect

    @available(iOS 26, *)
    private func applyGlassEffect(to button: UIButton) {
        var cfg = button.configuration ?? .plain()
        cfg.background.backgroundColor = .clear
        button.configuration = cfg
        button.clipsToBounds = true
        button.layer.cornerRadius = 10

        let glass = UIVisualEffectView(effect: UIGlassEffect())
        glass.frame = button.bounds
        glass.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        glass.isUserInteractionEnabled = false
        button.insertSubview(glass, at: 0)
    }

    // MARK: - UI State

    private func updateUI() {
        updatePickerDisplay()
        updateLoadButton()
        updateDeleteButton()
        updateThinkToggle()
        modelPickerView.reloadAllComponents()
    }

    private func updatePickerDisplay() {
        let info = models[selectedIndex]
        modelPickerField.text = "\(modelIcon(for: selectedIndex))  \(info.label)  ·  \(info.size)"
        modelDescriptionLabel.text = info.description
    }

    private func modelIcon(for index: Int) -> String {
        if loadedModelIndex == index   { return "⚡" }
        if loadingIndex == index       { return "🔄" }
        if downloadingIndex == index   { return "⬇️" }
        if isDownloaded(models[index].config) {
            switch memoryCompatibility(for: index).badge {
            case "❌": return "❌"
            case "⚠️": return "⚠️"
            default:   return "📥"
            }
        }
        return "☁️"
    }

    private func updateLoadButton() {
        let idx = selectedIndex
        var title: String
        var color: UIColor = .tintColor
        var enabled = true

        if isBusy {
            var cfg = loadButton.configuration ?? .filled()
            cfg.title = "로드"
            cfg.baseBackgroundColor = .systemGray
            loadButton.configuration = cfg
            loadButton.isEnabled = false
            return
        }

        if loadedModelIndex == idx {
            title = "언로드"
            color = .systemRed
        } else if loadingIndex == idx {
            title = "로드 중..."
            color = .systemGray
            enabled = false
        } else if downloadingIndex == idx {
            title = "다운로드 취소"
            color = .systemOrange
        } else if isDownloaded(models[idx].config) {
            // Cached model: can always load, even while another is downloading.
            let compat = memoryCompatibility(for: idx)
            switch compat.badge {
            case "❌": title = "로드 ❌"; color = .systemRed
            case "⚠️": title = "로드 ⚠️"; color = .systemOrange
            default:   title = "로드"
            }
            enabled = loadingIndex == nil
        } else if downloadingIndex != nil {
            // Another download is running — block new downloads.
            title = "다운로드 중..."
            color = .systemGray
            enabled = false
        } else {
            title = "다운로드"
            enabled = loadingIndex == nil
        }

        var cfg = loadButton.configuration ?? .filled()
        cfg.title = title
        cfg.baseBackgroundColor = color
        loadButton.configuration = cfg
        loadButton.isEnabled = enabled
    }

    private func updateDeleteButton() {
        let idx = selectedIndex
        deleteButton.isEnabled = !isBusy && (isDownloaded(models[idx].config) || loadedModelIndex == idx)
    }

    private func updateThinkToggle() {
        let info = models[selectedIndex]
        if info.supportsThinking {
            thinkLabel.text = "Think 모드"
            thinkLabel.textColor = .label
            thinkSwitch.isEnabled = !isBusy
        } else {
            thinkLabel.text = "Think 모드  (미지원)"
            thinkLabel.textColor = .tertiaryLabel
            thinkSwitch.isEnabled = false
            thinkSwitch.isOn = false
            isThinkingEnabled = false
        }
    }

    // MARK: - Generation Error Handling

    private func showTimeoutAlert(retryPrompt: String, stage: String = "생성") {
        let modelName = loadedModelIndex.map { models[$0].label } ?? "모델"
        let alert = UIAlertController(
            title: "⏱ 응답 타임아웃 (\(stage) 단계)",
            message: "\(modelName)이 3초 내 응답하지 못했습니다.\n대형 모델은 첫 토큰까지 시간이 걸릴 수 있습니다.\n언로드 후 재로드하거나 더 작은 모델을 시도해 보세요.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "재시도", style: .default) { [weak self] _ in
            self?.generate(prompt: retryPrompt)
        })
        alert.addAction(UIAlertAction(title: "언로드", style: .destructive) { [weak self] _ in
            self?.unloadModel()
        })
        alert.addAction(UIAlertAction(title: "닫기", style: .cancel))
        present(alert, animated: true)
        statusLabel.text = "⏱ 타임아웃 — 재시도하거나 모델을 언로드해 주세요"
    }

    // MARK: - Toast

    private func showToast(_ message: String) {
        let container = UIView()
        container.backgroundColor = UIColor.black.withAlphaComponent(0.75)
        container.layer.cornerRadius = 10
        container.clipsToBounds = true
        container.translatesAutoresizingMaskIntoConstraints = false

        let label = UILabel()
        label.text = message
        label.font = .systemFont(ofSize: 13)
        label.textColor = .white
        label.textAlignment = .center
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false

        container.addSubview(label)
        view.addSubview(container)

        NSLayoutConstraint.activate([
            label.topAnchor.constraint(equalTo: container.topAnchor, constant: 8),
            label.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -8),
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 14),
            label.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -14),
            container.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            container.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            container.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 32),
            container.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -32),
        ])

        UIView.animate(withDuration: 0.3, delay: 2.0, options: []) {
            container.alpha = 0
        } completion: { _ in
            container.removeFromSuperview()
        }
    }

    // MARK: - Picker Actions

    @objc private func thinkingToggled(_ sender: UISwitch) {
        isThinkingEnabled = sender.isOn
    }

    @objc private func pickerDone() {
        modelPickerField.resignFirstResponder()
        updateUI()
        if let loadedIdx = loadedModelIndex, loadedIdx != selectedIndex {
            showToast("⚡ \(models[loadedIdx].label) 로드됨 · 전송 시 이 모델로 실행됩니다")
        }
    }

    private func isDownloaded(_ config: ModelConfiguration) -> Bool {
        guard let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return false
        }
        switch config.id {
        case .directory:
            return true
        case .id(let idString, _):
            let dirName = "models--" + idString.replacingOccurrences(of: "/", with: "--")
            let modelDir = cachesDir
                .appendingPathComponent("huggingface/hub")
                .appendingPathComponent(dirName)
            return FileManager.default.fileExists(atPath: modelDir.path)
        }
    }

    // MARK: - Load Button Action

    @objc private func loadTapped() {
        let idx = selectedIndex

        // Unload: selected model is active
        if loadedModelIndex == idx {
            unloadModel()
            return
        }

        // Cancel: selected model is currently downloading
        if downloadingIndex == idx {
            downloadTask?.cancel()
            downloadTask = nil
            downloadingIndex = nil
            progressView.isHidden = true
            statusLabel.text = "다운로드 취소됨"
            updateUI()
            return
        }

        // Load from cache (fast) — allowed even while another model downloads in background
        if isDownloaded(models[idx].config) {
            loadFromCache(at: idx)
            return
        }

        // Block new download if one is already in progress
        if downloadingIndex != nil {
            let alert = UIAlertController(
                title: "다운로드 중",
                message: "\(models[downloadingIndex!].label) 다운로드가 진행 중입니다.\n완료 후 다운로드하거나, 해당 모델 선택 후 '다운로드 취소'를 눌러주세요.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "확인", style: .default))
            present(alert, animated: true)
            return
        }

        // Start background download
        startDownload(at: idx)
    }

    // MARK: - Load from Cache

    private func loadFromCache(at index: Int) {
        let neededGB   = estimatedRAMGB(for: index)
        let willFreeGB = loadedModelIndex.map { estimatedRAMGB(for: $0) } ?? 0
        let physGB     = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        // 실측 Jetsam 한도: iPhone Air 8GB → 3376MB(41.2%). 0.40을 보수적 상한으로 사용
        let limitGB    = physGB * 0.40

        // 정적 한도 초과 → 기기에서 이 모델은 절대 로드 불가
        if neededGB > limitGB {
            let alert = UIAlertController(
                title: "❌ 이 기기에서 로드 불가",
                message: String(
                    format: "필요 RAM: 약 %.1fGB\n기기 RAM %.1fGB · iOS 앱 한도 약 %.1fGB\n\n이 모델은 이 기기의 iOS 메모리 한도를 초과합니다.",
                    neededGB, physGB, limitGB
                ),
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "확인", style: .cancel))
            present(alert, animated: true)
            return
        }

        // 동적 체크: OS가 직접 알려주는 실제 남은 헤드룸 (Jetsam 까지 쓸 수 있는 바이트)
        let headroomGB  = availableAppMemoryGB()
        let effectiveGB = headroomGB + willFreeGB

        if neededGB <= effectiveGB {
            performLoadFromCache(at: index)
            return
        }

        // 한도 이내지만 현재 헤드룸 부족 → 경고 후 선택
        let alert = UIAlertController(
            title: "⚠️ 메모리 부족 경고",
            message: String(
                format: "필요 RAM: 약 %.1fGB\n현재 가용 (기존 모델 해제 후): %.1fGB\n\n기기 RAM %.1fGB · iOS 앱 한도 약 %.1fGB",
                neededGB, effectiveGB, physGB, limitGB
            ),
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "위험 감수하고 로드", style: .destructive) { [weak self] _ in
            self?.performLoadFromCache(at: index)
        })
        alert.addAction(UIAlertAction(title: "취소", style: .cancel))
        present(alert, animated: true)
    }

    private func performLoadFromCache(at index: Int) {
        // Release previous model to free RAM before loading the new one
        if modelContainer != nil {
            modelContainer = nil
            Memory.clearCache()  // Metal 버퍼 즉시 반환
            loadedModelIndex = nil
            sendButton.isEnabled = false
        }

        loadingIndex = index
        statusLabel.text = "🔄 \(models[index].label) 로드 중..."
        updateUI()

        loadTask = Task {
            do {
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: models[index].config,
                    progressHandler: { _ in }
                )
                if Task.isCancelled { return }

                self.modelContainer = container
                self.loadedModelIndex = index
                self.loadingIndex = nil
                self.sendButton.isEnabled = true
                self.statusLabel.text = "✅ \(self.models[index].label) 로드됨"
                self.updateUI()
            } catch {
                self.loadingIndex = nil
                self.statusLabel.text = "❌ \(error.localizedDescription)"
                self.updateUI()
            }
        }
    }

    // MARK: - System Stats

    // MARK: - Memory Estimation

    private func estimatedRAMGB(for index: Int) -> Double {
        var s = models[index].size.trimmingCharacters(in: .whitespaces)
        if s.hasPrefix("~") { s = String(s.dropFirst()) }
        let upper = s.uppercased()
        let numericStr = s.prefix(while: { $0.isNumber || $0 == "." })
        guard let value = Double(numericStr) else { return 4.0 }
        let rawGB = upper.contains("MB") ? value / 1024.0 : value
        return rawGB * 1.5   // ~50% overhead: KV cache + Metal buffers + tokenizer parsing (Gemma 256K vocab, etc.)
    }

    private func memoryCompatibility(for index: Int) -> (badge: String, label: String, color: UIColor) {
        let physGB   = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        // 실측: iPhone Air 8GB → Jetsam limit 3376MB = 41.2%. 0.40을 보수적 상한으로 사용
        let limitGB  = physGB * 0.40
        let neededGB = estimatedRAMGB(for: index)
        if neededGB > limitGB        { return ("❌", "불가", .systemRed) }
        if neededGB > limitGB * 0.85 { return ("⚠️", "주의", .systemOrange) }
        return ("✅", "적합", .systemGreen)
    }

    private func availableAppMemoryGB() -> Double {
        // os_proc_available_memory()는 현재 프로세스가 Jetsam 전까지 사용 가능한 실제 바이트를 반환
        let bytes = os_proc_available_memory()
        return bytes > 0 ? Double(bytes) / 1_073_741_824 : 0
    }

    // Metal 버퍼가 OS에 실제로 반환될 때까지 폴링 대기
    // 반환값: true = 충분한 메모리 확보, false = 15초 내 확보 실패 (중단 필요)
    @discardableResult
    private func waitForMemoryHeadroom(needed neededGB: Double = 2.0) async -> Bool {
        let deadline = Date().addingTimeInterval(15)
        while Date() < deadline {
            let availGB = Double(os_proc_available_memory()) / 1_073_741_824
            if availGB >= neededGB { return true }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false  // 시간 내 메모리 확보 실패
    }

    private func startStatsTimer() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        updateStats()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateStats()
        }
    }

    private func updateStats() {
        let ramStr = appRAMString()
        let diskStr = systemFreeDiskString()
        let thermalIcon = thermalStateIcon()
        let batteryStr = batteryString()
        statsLabel.text = "RAM \(ramStr)  ·  여유 \(diskStr)  ·  발열\(thermalIcon)  ·  \(batteryStr)"
    }

    private func physFootprintGB() -> Double {
        // task_vm_info_data_t is the correct struct for phys_footprint
        // (mach_task_basic_info does NOT have this field)
        var vmInfo = task_vm_info_data_t()
        var count  = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(vmInfo.phys_footprint) / 1_073_741_824
    }

    private func appRAMString() -> String {
        let gb = physFootprintGB()
        guard gb > 0 else { return "—" }
        let mb = gb * 1024
        return mb >= 1024 ? String(format: "%.1fGB", gb) : String(format: "%.0fMB", mb)
    }

    private func systemFreeDiskBytes() -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()),
              let free = attrs[.systemFreeSize] as? Int64 else { return 0 }
        return free
    }

    private func systemFreeDiskString() -> String {
        let bytes = systemFreeDiskBytes()
        return bytes > 0 ? ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) : "—"
    }

    private func thermalStateIcon() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:    return "🟢"
        case .fair:       return "🟡"
        case .serious:    return "🟠"
        case .critical:   return "🔴"
        @unknown default: return "❓"
        }
    }

    private func batteryString() -> String {
        let level = UIDevice.current.batteryLevel
        guard level >= 0 else { return "🔋—" }
        let pct = Int(level * 100)
        let icon = UIDevice.current.batteryState == .charging ? "⚡" : "🔋"
        return "\(icon)\(pct)%"
    }

    // MARK: - Download Verification

    private func verifyDownloadedFiles(at index: Int) -> (ok: Bool, detail: String) {
        let config = models[index].config
        guard case .id(let idString, _) = config.id,
              let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return (false, "캐시 디렉터리 접근 실패")
        }
        let dirName = "models--" + idString.replacingOccurrences(of: "/", with: "--")
        let modelDir = cachesDir.appendingPathComponent("huggingface/hub").appendingPathComponent(dirName)

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            return (false, "모델 디렉터리 없음")
        }

        let snapshotsDir = modelDir.appendingPathComponent("snapshots")
        guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path),
              let snapshot = snapshots.first else {
            return (false, "스냅샷 없음 — 다운로드가 중단됐을 수 있습니다")
        }

        let snapshotDir = snapshotsDir.appendingPathComponent(snapshot)
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: snapshotDir.path) else {
            return (false, "파일 목록 읽기 실패")
        }

        let hasConfig  = files.contains("config.json")
        let hasWeights = files.contains { $0.hasSuffix(".safetensors") || $0.hasSuffix(".npz") || $0.hasSuffix(".gguf") }

        if !hasConfig  { return (false, "config.json 없음") }
        if !hasWeights { return (false, "가중치 파일(.safetensors) 없음 — 재다운로드 필요") }

        return (true, "\(files.count)개 파일")
    }

    // MARK: - Background Download

    private func startDownload(at index: Int) {
        guard case .id(let idString, let revision) = models[index].config.id,
              let repoID = Repo.ID(rawValue: idString) else { return }

        downloadingIndex = index
        lastProgressUpdate = .distantPast
        progressView.isHidden = false
        progressView.progress = 0
        statusLabel.text = "⬇️ \(models[index].label) 다운로드 중..."
        updateUI()

        downloadTask = Task {
            do {
                // downloadSnapshot downloads model files to disk (Caches/huggingface/hub/)
                // WITHOUT loading weights into MLX RAM, preventing out-of-memory crashes
                // when another model is already loaded.
                _ = try await HubClient().downloadSnapshot(
                    of: repoID,
                    revision: revision,
                    progressHandler: { @MainActor [weak self] progress in
                        guard let self, self.downloadingIndex == index else { return }
                        let now = Date()
                        guard now.timeIntervalSince(self.lastProgressUpdate) >= 1.0 else { return }
                        self.lastProgressUpdate = now
                        let pct = Float(progress.fractionCompleted)
                        self.progressView.progress = pct
                        let done = ByteCountFormatter.string(fromByteCount: progress.completedUnitCount, countStyle: .file)
                        let total = ByteCountFormatter.string(fromByteCount: progress.totalUnitCount, countStyle: .file)
                        self.statusLabel.text = String(format: "⬇️ %@  %@ / %@  (%.0f%%)", self.models[index].label, done, total, pct * 100)
                    }
                )

                if Task.isCancelled {
                    self.downloadingIndex = nil
                    self.progressView.isHidden = true
                    self.updateUI()
                    return
                }

                self.downloadingIndex = nil
                self.downloadTask = nil
                self.progressView.isHidden = true

                // Verify downloaded files before proceeding
                let (ok, detail) = self.verifyDownloadedFiles(at: index)
                guard ok else {
                    self.statusLabel.text = "❌ 검증 실패: \(detail)\n다시 다운로드해 주세요."
                    self.updateUI()
                    return
                }

                // Auto-load if no model is active; otherwise just notify
                if self.loadedModelIndex == nil {
                    self.loadFromCache(at: index)
                } else {
                    self.updateUI()
                    self.statusLabel.text = "✅ \(self.models[index].label) 다운로드 완료 (\(detail)). '로드'를 눌러 전환하세요."
                }
            } catch {
                if !Task.isCancelled {
                    self.downloadingIndex = nil
                    self.progressView.isHidden = true
                    self.statusLabel.text = "❌ \(error.localizedDescription)"
                    self.updateUI()
                }
            }
        }
    }

    // MARK: - Unload

    private func unloadModel() {
        generateTask?.cancel()
        generateTask = nil
        modelContainer = nil
        Memory.clearCache()  // Metal 버퍼 즉시 반환
        loadedModelIndex = nil
        sendButton.isEnabled = false
        accumulatedText = ""
        statusLabel.text = "모델 언로드됨"
        updateUI()
    }

    // MARK: - Comparison

    @objc private func sendLongPressed(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began, !isComparing else { return }
        guard let text = inputField.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            showToast("먼저 질문을 입력하세요")
            return
        }
        let downloadedIndices = models.indices.filter { isDownloaded(models[$0].config) }
        guard downloadedIndices.count >= 2 else {
            showToast("비교하려면 다운로드된 모델이 2개 이상 필요합니다")
            return
        }
        inputField.resignFirstResponder()

        let selectionVC = ComparisonModelSelectionVC(
            modelLabels: models.map { $0.label },
            downloadedIndices: downloadedIndices
        ) { [weak self] selectedIndices in
            self?.startComparison(prompt: text, modelIndices: selectedIndices)
        }
        let nav = UINavigationController(rootViewController: selectionVC)
        nav.modalPresentationStyle = .pageSheet
        if let sheet = nav.sheetPresentationController {
            sheet.detents = [.medium(), .large()]
            sheet.prefersGrabberVisible = true
        }
        present(nav, animated: true)
    }

    private func startComparison(prompt: String, modelIndices: [Int]) {
        let physGB  = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        // 비교 모드는 모델을 순차 로드하므로 단일 로드(0.40)보다 보수적인 한도 적용
        // 토크나이저 파싱 피크 메모리까지 고려해 0.30으로 제한
        let limitGB = physGB * 0.30
        let loadableIndices = modelIndices.filter { estimatedRAMGB(for: $0) <= limitGB }

        guard !loadableIndices.isEmpty else {
            showToast("선택한 모델이 모두 이 기기 메모리 한도를 초과합니다")
            return
        }
        if loadableIndices.count < modelIndices.count {
            let skipped = modelIndices.count - loadableIndices.count
            showToast("메모리 한도 초과 \(skipped)개 모델 제외됩니다")
        }

        // 기존 모델 언로드 (RAM 확보)
        generateTask?.cancel()
        generateTask = nil
        modelContainer = nil
        loadedModelIndex = nil
        Memory.clearCache()  // 이전 모델의 Metal 버퍼를 즉시 캐시에서 제거

        isComparing = true
        sendButton.isEnabled = false
        inputField.isEnabled = false
        updateUI()

        let total = loadableIndices.count
        var results: [(label: String, text: String, tps: Double)] = []

        compareTask = Task {
            // 첫 모델 로드 전: 이전 모델 Metal 버퍼가 OS에 반환될 때까지 대기
            // 실패(15초 내 2GB 미확보)시 비교 자체를 중단
            guard await waitForMemoryHeadroom() else {
                self.isComparing = false
                self.inputField.isEnabled = true
                self.updateUI()
                self.statusLabel.text = "❌ 메모리 부족으로 비교를 시작할 수 없습니다. 기존 모델을 언로드 후 시도하세요."
                return
            }

            for (i, idx) in loadableIndices.enumerated() {
                if Task.isCancelled { break }
                let label = models[idx].label
                let step = "\(i + 1)/\(total)"

                self.statusLabel.text = "🔄 비교 \(step) · \(label) 로드 중..."

                do {
                    var container: ModelContainer? = try await LLMModelFactory.shared.loadContainer(
                        from: #hubDownloader(),
                        using: #huggingFaceTokenizerLoader(),
                        configuration: models[idx].config,
                        progressHandler: { _ in }
                    )
                    guard !Task.isCancelled, let c = container else { container = nil; break }

                    self.statusLabel.text = "⏳ 비교 \(step) · \(label) 생성 중..."

                    let additionalContext: [String: any Sendable]? = self.isThinkingEnabled ? nil : ["enable_thinking": false]
                    let userInput = UserInput(
                        chat: [.system(self.systemPrompt), .user(prompt)],
                        additionalContext: additionalContext
                    )
                    let lmInput = try await c.prepare(input: userInput)
                    let stream = try await c.generate(
                        input: lmInput,
                        parameters: GenerateParameters(maxTokens: min(self.models[idx].maxTokens, 1024), temperature: 0.7)
                    )

                    var text = ""
                    var tps = 0.0
                    for await event in stream {
                        if Task.isCancelled { break }
                        switch event {
                        case .chunk(let chunk): text += chunk
                        case .info(let info):   tps = info.tokensPerSecond
                        default: break
                        }
                    }
                    results.append((label: label, text: text, tps: tps))
                    // container, c, stream 모두 do 블록 종료 시 해제됨
                } catch {
                    results.append((label: label, text: "오류: \(error.localizedDescription)", tps: 0))
                }
                // do-catch 블록이 끝난 뒤 — c, stream 참조 소멸 후 캐시 클리어
                Memory.clearCache()
                // 다음 모델 전 메모리 확보 대기, 실패 시 남은 결과로 비교 완료
                if i < loadableIndices.count - 1 {
                    guard await waitForMemoryHeadroom() else {
                        self.statusLabel.text = "⚠️ 메모리 부족으로 \(i + 1)/\(total)개 모델까지만 비교됩니다."
                        break
                    }
                }
            }

            self.isComparing = false
            self.inputField.isEnabled = true
            self.sendButton.isEnabled = self.modelContainer != nil
            self.updateUI()

            guard !results.isEmpty, !Task.isCancelled else {
                self.statusLabel.text = "비교 취소됨"
                return
            }

            self.statusLabel.text = "✅ 비교 완료 — \(results.count)/\(total)개 모델"
            let renderer: (String) -> NSAttributedString = { [weak self] text in
                self?.renderOutput(text) ?? NSAttributedString(string: text)
            }
            let vc = ComparisonResultsVC(
                prompt: prompt,
                systemPrompt: self.systemPrompt,
                results: results,
                renderer: renderer
            )
            let nav = UINavigationController(rootViewController: vc)
            nav.modalPresentationStyle = .fullScreen
            self.present(nav, animated: true)
        }
    }

    // MARK: - Delete

    @objc private func deleteTapped() {
        let idx = selectedIndex
        let info = models[idx]

        // Block deletion while the model is being loaded into RAM — the loadContainer()
        // call is reading local files and cannot be safely interrupted mid-parse.
        if loadingIndex == idx {
            let alert = UIAlertController(
                title: "로드 중",
                message: "\(info.label) 로드가 진행 중입니다.\n잠시 후 다시 시도해 주세요.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "확인", style: .default))
            present(alert, animated: true)
            return
        }

        let alert = UIAlertController(
            title: "모델 삭제",
            message: "\(info.label)를 삭제하시겠습니까?\n(\(info.size) 공간이 확보됩니다)",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "삭제", style: .destructive) { [weak self] _ in
            self?.deleteModel(at: idx)
        })
        alert.addAction(UIAlertAction(title: "취소", style: .cancel))
        present(alert, animated: true)
    }

    private func deleteModel(at index: Int) {
        if loadedModelIndex == index {
            generateTask?.cancel()
            generateTask = nil
            modelContainer = nil
            loadedModelIndex = nil
            sendButton.isEnabled = false
        }
        if loadingIndex == index {
            loadTask?.cancel()
            loadTask = nil
            loadingIndex = nil
        }
        if downloadingIndex == index {
            downloadTask?.cancel()
            downloadTask = nil
            downloadingIndex = nil
            progressView.isHidden = true
        }

        let freeBefore = systemFreeDiskBytes()
        let config = models[index].config
        switch config.id {
        case .directory:
            break
        case .id(let idString, _):
            guard let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
                statusLabel.text = "❌ 캐시 디렉터리 접근 실패"
                updateUI()
                return
            }
            let dirName = "models--" + idString.replacingOccurrences(of: "/", with: "--")
            let modelDir = cachesDir
                .appendingPathComponent("huggingface/hub")
                .appendingPathComponent(dirName)
            guard FileManager.default.fileExists(atPath: modelDir.path) else {
                statusLabel.text = "⚠️ 캐시 없음 (이미 삭제됨?)\n경로: \(modelDir.lastPathComponent)"
                updateUI()
                return
            }
            do {
                try FileManager.default.removeItem(at: modelDir)
            } catch {
                statusLabel.text = "❌ 삭제 실패: \(error.localizedDescription)"
                updateUI()
                return
            }
        }

        updateUI()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            let freed = self.systemFreeDiskBytes() - freeBefore
            if freed > 0 {
                let freedStr = ByteCountFormatter.string(fromByteCount: freed, countStyle: .file)
                self.statusLabel.text = "🗑 \(self.models[index].label) 삭제됨 · \(freedStr) 확보"
            } else {
                self.statusLabel.text = "🗑 \(self.models[index].label) 삭제됨 (공간 변화 없음 — OS 캐시 지연일 수 있음)"
            }
        }
    }

    // MARK: - Markdown Rendering

    private func renderOutput(_ text: String) -> NSAttributedString {
        let result = NSMutableAttributedString()
        var remaining = text

        while let thinkStart = remaining.range(of: "<think>") {
            let before = String(remaining[remaining.startIndex..<thinkStart.lowerBound])
            if !before.isEmpty { result.append(renderMarkdown(before)) }
            remaining = String(remaining[thinkStart.upperBound...])

            if let thinkEnd = remaining.range(of: "</think>") {
                let thinkContent = String(remaining[remaining.startIndex..<thinkEnd.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !thinkContent.isEmpty {
                    result.append(NSAttributedString(
                        string: thinkContent + "\n\n",
                        attributes: [.foregroundColor: UIColor.tertiaryLabel, .font: UIFont.systemFont(ofSize: 12)]
                    ))
                }
                remaining = String(remaining[thinkEnd.upperBound...].drop(while: { $0.isNewline }))
            } else {
                result.append(NSAttributedString(
                    string: remaining,
                    attributes: [.foregroundColor: UIColor.tertiaryLabel, .font: UIFont.systemFont(ofSize: 12)]
                ))
                remaining = ""
            }
        }

        if !remaining.isEmpty { result.append(renderMarkdown(remaining)) }
        return result
    }

    private func renderMarkdown(_ text: String) -> NSAttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .full)
        let baseFont = UIFont.systemFont(ofSize: 14)
        guard let attrStr = try? AttributedString(markdown: text, options: options) else {
            return NSAttributedString(string: text, attributes: [.font: baseFont, .foregroundColor: UIColor.label])
        }
        let ns = NSMutableAttributedString(attrStr)
        let full = NSRange(location: 0, length: ns.length)
        ns.addAttribute(.foregroundColor, value: UIColor.label, range: full)
        ns.enumerateAttribute(.font, in: full) { value, range, _ in
            if value == nil { ns.addAttribute(.font, value: baseFont, range: range) }
        }
        return ns
    }

    // MARK: - Generation

    @objc private func sendTapped() {
        guard let text = inputField.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return }
        inputField.resignFirstResponder()
        generate(prompt: text)
    }

    private func generate(prompt: String) {
        guard let container = modelContainer, let loadedIdx = loadedModelIndex else { return }
        if selectedIndex != loadedIdx {
            selectedIndex = loadedIdx
            modelPickerView.selectRow(loadedIdx, inComponent: 0, animated: false)
            updateUI()
        }
        let maxTokens = models[loadedIdx].maxTokens

        generateTask?.cancel()
        accumulatedText = ""
        isGenerating = true
        sendButton.isEnabled = false
        outputTextView.attributedText = nil
        outputTextView.text = ""
        outputTextView.textColor = .label
        updateUI()

        generateTask = Task {
            do {
                self.lastSentSystem = systemPrompt
                self.lastSentUser   = prompt
                let additionalContext: [String: any Sendable]? = isThinkingEnabled ? nil : ["enable_thinking": false]
                let userInput = UserInput(
                    chat: [.system(systemPrompt), .user(prompt)],
                    additionalContext: additionalContext
                )

                let modelName = self.models[loadedIdx].label
                self.statusLabel.text = "⏳ \(modelName) · 입력 토큰화 중..."

                let lmInput = try await container.prepare(input: userInput)
                if Task.isCancelled { return }

                self.statusLabel.text = "🤔 \(modelName) · 첫 토큰 생성 중... (대형 모델은 수 초 소요)"

                let stream = try await container.generate(
                    input: lmInput,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.7)
                )

                self.statusLabel.text = "✍️ \(modelName) · 생성 중..."

                for await event in stream {
                    if Task.isCancelled { break }
                    switch event {
                    case .chunk(let text):
                        self.accumulatedText += text
                        self.outputTextView.attributedText = self.renderOutput(self.accumulatedText)
                        let len = self.outputTextView.text.count
                        if len > 0 {
                            self.outputTextView.scrollRangeToVisible(NSRange(location: len - 1, length: 1))
                        }
                    case .info(let info):
                        self.outputTextView.attributedText = self.renderOutput(self.accumulatedText)
                        let modelName = self.loadedModelIndex.map { self.models[$0].label } ?? ""
                        self.statusLabel.text = String(format: "✅ %@ · %.1f tok/s", modelName, info.tokensPerSecond)
                    default:
                        break
                    }
                }
            } catch {
                let desc = error.localizedDescription
                if desc.lowercased().contains("timeout") {
                    let stage = self.statusLabel.text?.contains("토큰화") == true ? "입력 처리" : "첫 토큰 생성"
                    self.showTimeoutAlert(retryPrompt: prompt, stage: stage)
                } else {
                    self.outputTextView.text = "오류: \(desc)"
                    self.outputTextView.textColor = .systemRed
                }
            }

            self.isGenerating = false
            self.sendButton.isEnabled = self.modelContainer != nil
            self.updateUI()
            self.inputField.text = ""
        }
    }
}

// MARK: - UIPickerViewDataSource, UIPickerViewDelegate

extension ViewController: UIPickerViewDataSource, UIPickerViewDelegate {

    func numberOfComponents(in pickerView: UIPickerView) -> Int { 1 }

    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        models.count
    }

    func pickerView(_ pickerView: UIPickerView, rowHeightForComponent component: Int) -> CGFloat { 56 }

    func pickerView(_ pickerView: UIPickerView, viewForRow row: Int, forComponent component: Int, reusing view: UIView?) -> UIView {
        let label = (view as? UILabel) ?? UILabel()
        label.textAlignment = .center
        label.numberOfLines = 2

        let info   = models[row]
        let icon   = modelIcon(for: row)
        let compat = memoryCompatibility(for: row)

        var statusText: String
        if loadedModelIndex == row        { statusText = "로드됨" }
        else if loadingIndex == row       { statusText = "로드 중..." }
        else if downloadingIndex == row   { statusText = "다운로드 중..." }
        else if isDownloaded(info.config) { statusText = "다운로드됨" }
        else                              { statusText = "미다운로드" }

        // Line 1: icon + model name
        let text = NSMutableAttributedString(
            string: "\(icon)  \(info.label)\n",
            attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .medium)]
        )
        // Line 2: size · status in secondary color, then compatibility badge in tier color
        let line2 = NSMutableAttributedString(
            string: "\(info.size)  ·  \(statusText)  · ",
            attributes: [.font: UIFont.systemFont(ofSize: 12), .foregroundColor: UIColor.secondaryLabel]
        )
        line2.append(NSAttributedString(
            string: " \(compat.badge) \(compat.label)",
            attributes: [.font: UIFont.systemFont(ofSize: 12, weight: .semibold), .foregroundColor: compat.color]
        ))
        text.append(line2)

        label.attributedText = text
        return label
    }

    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        selectedIndex = row
        updateUI()
    }
}

// MARK: - UITextFieldDelegate

extension ViewController: UITextFieldDelegate {

    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        return textField != modelPickerField
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if textField == inputField { sendTapped() }
        textField.resignFirstResponder()
        return true
    }
}

// MARK: - Comparison Model Selection VC

private final class ComparisonModelSelectionVC: UITableViewController {
    private let modelLabels: [String]
    private let downloadedIndices: [Int]
    private var selectedSet: Set<Int> = []
    private let completion: ([Int]) -> Void

    init(modelLabels: [String], downloadedIndices: [Int], completion: @escaping ([Int]) -> Void) {
        self.modelLabels = modelLabels
        self.downloadedIndices = downloadedIndices
        self.completion = completion
        super.init(style: .insetGrouped)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "비교할 모델 선택 (2개 이상)"
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            title: "취소", style: .plain, target: self, action: #selector(cancelTapped))
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            title: "비교 시작", style: .done, target: self, action: #selector(startTapped))
        navigationItem.rightBarButtonItem?.isEnabled = false
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        downloadedIndices.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = UITableViewCell(style: .subtitle, reuseIdentifier: nil)
        let idx = downloadedIndices[indexPath.row]
        cell.textLabel?.text = modelLabels[idx]
        cell.accessoryType = selectedSet.contains(idx) ? .checkmark : .none
        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        let idx = downloadedIndices[indexPath.row]
        if selectedSet.contains(idx) { selectedSet.remove(idx) } else { selectedSet.insert(idx) }
        tableView.reloadRows(at: [indexPath], with: .none)
        navigationItem.rightBarButtonItem?.isEnabled = selectedSet.count >= 2
        title = selectedSet.count >= 2 ? "비교할 모델 선택 (\(selectedSet.count)개)" : "비교할 모델 선택 (2개 이상)"
    }

    @objc private func cancelTapped() { dismiss(animated: true) }

    @objc private func startTapped() {
        let ordered = downloadedIndices.filter { selectedSet.contains($0) }
        dismiss(animated: true) { self.completion(ordered) }
    }
}

// MARK: - Comparison Results VC

private final class ComparisonResultsVC: UIViewController {
    private let prompt: String
    private let systemPromptText: String
    private let results: [(label: String, text: String, tps: Double)]
    private let renderer: (String) -> NSAttributedString

    init(prompt: String,
         systemPrompt: String,
         results: [(label: String, text: String, tps: Double)],
         renderer: @escaping (String) -> NSAttributedString) {
        self.prompt           = prompt
        self.systemPromptText = systemPrompt
        self.results = results
        self.renderer = renderer
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        title = "모델 비교 (\(results.count)개)"
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            title: "닫기", style: .done, target: self, action: #selector(closeTapped))

        let scrollView = UIScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scrollView)

        let stack = UIStackView()
        stack.axis = .vertical
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        scrollView.addSubview(stack)

        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            stack.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 16),
            stack.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor, constant: -16),
            stack.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor, constant: -16),
            stack.widthAnchor.constraint(equalTo: scrollView.widthAnchor, constant: -32),
        ])

        // 질문 카드 (탭하면 실제 LLM 입력 확인)
        let promptCard = makeCard()
        let cardStack = UIStackView()
        cardStack.axis = .vertical
        cardStack.spacing = 6
        cardStack.translatesAutoresizingMaskIntoConstraints = false
        promptCard.addSubview(cardStack)
        NSLayoutConstraint.activate([
            cardStack.topAnchor.constraint(equalTo: promptCard.topAnchor, constant: 12),
            cardStack.leadingAnchor.constraint(equalTo: promptCard.leadingAnchor, constant: 14),
            cardStack.trailingAnchor.constraint(equalTo: promptCard.trailingAnchor, constant: -14),
            cardStack.bottomAnchor.constraint(equalTo: promptCard.bottomAnchor, constant: -12),
        ])
        let promptLabel = UILabel()
        promptLabel.text = "Q. \(prompt)"
        promptLabel.font = .systemFont(ofSize: 14, weight: .semibold)
        promptLabel.textColor = .secondaryLabel
        promptLabel.numberOfLines = 0
        cardStack.addArrangedSubview(promptLabel)
        let hintLabel = UILabel()
        hintLabel.text = "탭하여 실제 LLM 입력 확인 ↗"
        hintLabel.font = .systemFont(ofSize: 11)
        hintLabel.textColor = .tertiaryLabel
        cardStack.addArrangedSubview(hintLabel)
        promptCard.isUserInteractionEnabled = true
        promptCard.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(showInput)))
        stack.addArrangedSubview(promptCard)

        // 모델별 결과 카드
        for result in results {
            stack.addArrangedSubview(makeResultCard(result))
        }
    }

    private func makeResultCard(_ result: (label: String, text: String, tps: Double)) -> UIView {
        let card = makeCard()
        let inner = UIStackView()
        inner.axis = .vertical
        inner.spacing = 8
        inner.translatesAutoresizingMaskIntoConstraints = false
        card.addSubview(inner)
        NSLayoutConstraint.activate([
            inner.topAnchor.constraint(equalTo: card.topAnchor, constant: 12),
            inner.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 14),
            inner.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -14),
            inner.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -12),
        ])

        // 헤더: 모델명 + tok/s
        let headerRow = UIStackView()
        headerRow.axis = .horizontal
        headerRow.spacing = 8

        let nameLabel = UILabel()
        nameLabel.text = result.label
        nameLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        headerRow.addArrangedSubview(nameLabel)

        if result.tps > 0 {
            let tpsLabel = UILabel()
            tpsLabel.text = String(format: "%.1f tok/s", result.tps)
            tpsLabel.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
            tpsLabel.textColor = .secondaryLabel
            tpsLabel.setContentHuggingPriority(.required, for: .horizontal)
            headerRow.addArrangedSubview(tpsLabel)
        }
        inner.addArrangedSubview(headerRow)

        let sep = UIView()
        sep.backgroundColor = .separator
        sep.translatesAutoresizingMaskIntoConstraints = false
        sep.heightAnchor.constraint(equalToConstant: 0.5).isActive = true
        inner.addArrangedSubview(sep)

        // 응답 텍스트 (마크다운 렌더링)
        let textView = UITextView()
        textView.isEditable = false
        textView.isScrollEnabled = false
        textView.font = .systemFont(ofSize: 14)
        textView.backgroundColor = .clear
        textView.textContainerInset = .zero
        textView.textContainer.lineFragmentPadding = 0
        textView.attributedText = renderer(result.text)
        inner.addArrangedSubview(textView)

        return card
    }

    private func makeCard() -> UIView {
        let v = UIView()
        v.backgroundColor = .secondarySystemBackground
        v.layer.cornerRadius = 12
        v.clipsToBounds = true
        return v
    }

    @objc private func closeTapped() { dismiss(animated: true) }

    @objc private func showInput() {
        let vc = LLMInputInspectorVC(system: systemPromptText, user: prompt)
        let nav = UINavigationController(rootViewController: vc)
        nav.modalPresentationStyle = .pageSheet
        if let sheet = nav.sheetPresentationController {
            sheet.detents = [.medium(), .large()]
            sheet.prefersGrabberVisible = true
        }
        present(nav, animated: true)
    }
}

// MARK: - LLM Input Inspector

private final class LLMInputInspectorVC: UIViewController {
    private let systemText: String
    private let userText: String

    init(system: String, user: String) {
        self.systemText = system
        self.userText   = user
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "실제 LLM 입력"
        view.backgroundColor = .systemBackground
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            title: "닫기", style: .done, target: self, action: #selector(closeTapped))

        let scroll = UIScrollView()
        scroll.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scroll.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
        ])

        let stack = UIStackView()
        stack.axis = .vertical
        stack.spacing = 20
        stack.translatesAutoresizingMaskIntoConstraints = false
        scroll.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: scroll.topAnchor, constant: 16),
            stack.leadingAnchor.constraint(equalTo: scroll.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: scroll.trailingAnchor, constant: -16),
            stack.bottomAnchor.constraint(equalTo: scroll.bottomAnchor, constant: -16),
            stack.widthAnchor.constraint(equalTo: scroll.widthAnchor, constant: -32),
        ])

        stack.addArrangedSubview(makeSection(title: "SYSTEM", body: systemText, color: .systemBlue))
        stack.addArrangedSubview(makeSection(title: "USER", body: userText, color: .systemGreen))
    }

    private func makeSection(title: String, body: String, color: UIColor) -> UIView {
        let container = UIView()
        container.backgroundColor = color.withAlphaComponent(0.07)
        container.layer.cornerRadius = 10
        container.clipsToBounds = true

        let badge = UILabel()
        badge.text = title
        badge.font = .monospacedSystemFont(ofSize: 11, weight: .bold)
        badge.textColor = color

        let bodyLabel = UILabel()
        bodyLabel.text = body
        bodyLabel.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        bodyLabel.textColor = .label
        bodyLabel.numberOfLines = 0

        let inner = UIStackView(arrangedSubviews: [badge, bodyLabel])
        inner.axis = .vertical
        inner.spacing = 6
        inner.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(inner)
        NSLayoutConstraint.activate([
            inner.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            inner.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 14),
            inner.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -14),
            inner.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12),
        ])
        return container
    }

    @objc private func closeTapped() { dismiss(animated: true) }
}
