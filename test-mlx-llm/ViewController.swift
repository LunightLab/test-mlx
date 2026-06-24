import UIKit
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
    private var accumulatedText = ""

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
        tf.placeholder = "질문 또는 뉴스 URL을 입력하세요..."
        tf.borderStyle = .roundedRect
        tf.returnKeyType = .send
        tf.delegate = self
        return tf
    }()

    private lazy var sendButton: UIButton = {
        var cfg = UIButton.Configuration.filled()
        cfg.title = "전송"
        let btn = UIButton(configuration: cfg)
        btn.addTarget(self, action: #selector(sendTapped), for: .touchUpInside)
        btn.isEnabled = false
        btn.setContentHuggingPriority(.required, for: .horizontal)
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
        if #available(iOS 26, *) {
            applyGlassEffect(to: sendButton)
            applyGlassEffect(to: pickerDoneButton)
        }
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
        if isDownloaded(models[index].config) { return "📥" }
        return "☁️"
    }

    private func updateLoadButton() {
        let idx = selectedIndex
        var title: String
        var color: UIColor = .tintColor
        var enabled = true

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
            title = "로드"
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
        deleteButton.isEnabled = isDownloaded(models[idx].config) || loadedModelIndex == idx
    }

    private func updateThinkToggle() {
        let info = models[selectedIndex]
        if info.supportsThinking {
            thinkLabel.text = "Think 모드"
            thinkLabel.textColor = .label
            thinkSwitch.isEnabled = true
        } else {
            thinkLabel.text = "Think 모드  (미지원)"
            thinkLabel.textColor = .tertiaryLabel
            thinkSwitch.isEnabled = false
            thinkSwitch.isOn = false
            isThinkingEnabled = false
        }
    }

    // MARK: - Picker Actions

    @objc private func thinkingToggled(_ sender: UISwitch) {
        isThinkingEnabled = sender.isOn
    }

    @objc private func pickerDone() {
        modelPickerField.resignFirstResponder()
        updateUI()
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
        // Release previous model to free RAM before loading the new one
        if modelContainer != nil {
            modelContainer = nil
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

    // MARK: - Background Download

    private func startDownload(at index: Int) {
        guard case .id(let idString, let revision) = models[index].config.id,
              let repoID = Repo.ID(rawValue: idString) else { return }

        downloadingIndex = index
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
                        let pct = Float(progress.fractionCompleted)
                        self.progressView.progress = pct
                        self.statusLabel.text = String(format: "⬇️ %@ %.0f%%", self.models[index].label, pct * 100)
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

                // Auto-load if no model is active; otherwise just notify
                if self.loadedModelIndex == nil {
                    self.loadFromCache(at: index)
                } else {
                    self.updateUI()
                    self.statusLabel.text = "✅ \(self.models[index].label) 다운로드 완료. '로드'를 눌러 전환하세요."
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
        loadedModelIndex = nil
        sendButton.isEnabled = false
        accumulatedText = ""
        statusLabel.text = "모델 언로드됨"
        updateUI()
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

        let config = models[index].config
        switch config.id {
        case .directory:
            break
        case .id(let idString, _):
            if let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
                let dirName = "models--" + idString.replacingOccurrences(of: "/", with: "--")
                let modelDir = cachesDir
                    .appendingPathComponent("huggingface/hub")
                    .appendingPathComponent(dirName)
                try? FileManager.default.removeItem(at: modelDir)
            }
        }

        statusLabel.text = "🗑 \(models[index].label) 삭제됨"
        updateUI()
    }

    // MARK: - Article Extraction

    private func extractAndSummarize(urlString: String) {
        guard modelContainer != nil else {
            statusLabel.text = "❌ 먼저 모델을 로드해주세요."
            return
        }
        generateTask?.cancel()
        sendButton.isEnabled = false
        accumulatedText = ""
        outputTextView.text = ""
        statusLabel.text = "🔍 기사 본문 추출 중..."

        Task {
            do {
                let article = try await ArticleExtractor.extract(from: urlString)
                let info = "📰 \(article.title)" + (article.author.map { "  ·  \($0)" } ?? "")
                self.statusLabel.text = info
                self.generate(prompt: article.summaryPrompt)
            } catch {
                self.statusLabel.text = "❌ \(error.localizedDescription)"
                self.sendButton.isEnabled = true
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
        if text.lowercased().hasPrefix("http://") || text.lowercased().hasPrefix("https://") {
            extractAndSummarize(urlString: text)
        } else {
            generate(prompt: text)
        }
    }

    private func generate(prompt: String) {
        guard let container = modelContainer else { return }
        let maxTokens = models[selectedIndex].maxTokens

        generateTask?.cancel()
        accumulatedText = ""
        sendButton.isEnabled = false
        outputTextView.attributedText = nil
        outputTextView.text = ""

        generateTask = Task {
            do {
                let additionalContext: [String: any Sendable]? = isThinkingEnabled ? nil : ["enable_thinking": false]
                let userInput = UserInput(
                    chat: [.system(systemPrompt), .user(prompt)],
                    additionalContext: additionalContext
                )
                let lmInput = try await container.prepare(input: userInput)
                let stream = try await container.generate(
                    input: lmInput,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.7)
                )

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
                self.outputTextView.text = "오류: \(error.localizedDescription)"
                self.outputTextView.textColor = .systemRed
            }

            self.sendButton.isEnabled = self.modelContainer != nil
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

    func pickerView(_ pickerView: UIPickerView, rowHeightForComponent component: Int) -> CGFloat { 44 }

    func pickerView(_ pickerView: UIPickerView, viewForRow row: Int, forComponent component: Int, reusing view: UIView?) -> UIView {
        let label = (view as? UILabel) ?? UILabel()
        label.textAlignment = .center

        let info = models[row]
        let icon = modelIcon(for: row)

        var statusText: String
        if loadedModelIndex == row        { statusText = "로드됨" }
        else if loadingIndex == row       { statusText = "로드 중..." }
        else if downloadingIndex == row   { statusText = "다운로드 중..." }
        else if isDownloaded(info.config) { statusText = "다운로드됨" }
        else                              { statusText = "미다운로드" }

        let text = NSMutableAttributedString()
        text.append(NSAttributedString(
            string: "\(icon)  \(info.label)",
            attributes: [.font: UIFont.systemFont(ofSize: 15, weight: .medium)]
        ))
        text.append(NSAttributedString(
            string: "  ·  \(info.size)  ·  \(statusText)",
            attributes: [.font: UIFont.systemFont(ofSize: 13), .foregroundColor: UIColor.secondaryLabel]
        ))
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
