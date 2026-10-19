import SwiftUI
import PhotosUI

enum Engine: String, CaseIterable, Identifiable {
    case custom = "自訂模型"
    case vision = "Apple Vision"
    var id: String { rawValue }
}

/// 任何設定改變都會自動重新辨識
private struct RunKey: Equatable {
    var version: Int
    var engine: Engine
    var settings: OCRSettings
    var useVariants: Bool
    var options: PreprocessOptions
}

struct ContentView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var original: UIImage?
    @State private var processed: UIImage?
    @State private var imageVersion = 0

    // 自訂 ONNX 模型只載入一次
    private static let model = Result(catching: { try OnnxOCR() })

    // 用 AppStorage 保存設定，下次開 App 不用重設
    @AppStorage("engine") private var engine: Engine = .custom
    @AppStorage("mode") private var mode: CharsetMode = .letters
    @AppStorage("caseMode") private var caseMode: CaseMode = .lower
    @AppStorage("useLength") private var useLength = true
    @AppStorage("length") private var length = 4
    @AppStorage("useFast") private var useFast = true
    @AppStorage("useVariants") private var useVariants = true
    @State private var options = PreprocessOptions()

    @State private var candidates: [OCRCandidate] = []
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var showCamera = false
    @State private var copied: String?

    private var settings: OCRSettings {
        OCRSettings(mode: mode,
                    expectedLength: useLength ? length : nil,
                    caseMode: caseMode,
                    useFastLevel: useFast)
    }

    private var runKey: RunKey {
        RunKey(version: imageVersion, engine: engine, settings: settings,
               useVariants: useVariants, options: options)
    }

    var body: some View {
        NavigationStack {
            Form {
                sourceSection
                resultSection
                engineSection
                if engine == .vision {
                    settingsSection
                    preprocessSection
                }
            }
            .navigationTitle("驗證碼辨識")
            .onChange(of: pickerItem) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        load(image)
                    }
                }
            }
            .task(id: runKey) { await runOCR() }
            .sheet(isPresented: $showCamera) {
                CameraPicker { load($0) }.ignoresSafeArea()
            }
        }
    }

    // MARK: - Sections

    private var sourceSection: some View {
        Section("影像來源") {
            HStack {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Label("相簿", systemImage: "photo")
                }
                Spacer()
                Button { showCamera = true } label: {
                    Label("拍照", systemImage: "camera")
                }
                .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
                Spacer()
                Button { pasteImage() } label: {
                    Label("貼上", systemImage: "doc.on.clipboard")
                }
            }
            .buttonStyle(.borderless)

            if let original { ImageRow(title: "原圖", image: original) }
            if let processed, options.enabled, engine == .vision { ImageRow(title: "前處理後（實際送去辨識的圖）", image: processed) }
        }
    }

    private var resultSection: some View {
        Section {
            if isRunning { ProgressView("辨識中…") }
            if let errorMessage {
                Text(errorMessage).foregroundStyle(.secondary)
            }
            ForEach(Array(candidates.prefix(5).enumerated()), id: \.element.id) { index, candidate in
                Button {
                    UIPasteboard.general.string = candidate.text
                    copied = candidate.text
                } label: {
                    HStack {
                        Text(candidate.text)
                            .font(.system(index == 0 ? .title : .body, design: .monospaced))
                            .bold(index == 0)
                        if candidate.isVote {
                            Text("投票")
                                .font(.caption2).bold()
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(.green.opacity(0.2), in: Capsule())
                                .foregroundStyle(.green)
                        }
                        Spacer()
                        if engine == .vision {
                            Text("\(candidate.votes) 票")
                                .foregroundStyle(.secondary)
                        }
                        Image(systemName: copied == candidate.text ? "checkmark" : "doc.on.doc")
                    }
                }
            }
        } header: {
            Text("結果（點一下複製）")
        }
    }

    private var engineSection: some View {
        Section {
            Picker("辨識引擎", selection: $engine) {
                ForEach(Engine.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
        } footer: {
            if engine == .custom, case .failure(let error) = Self.model {
                Text("模型載入失敗：\(error.localizedDescription)").foregroundStyle(.red)
            } else if engine == .custom {
                Text("使用 App 內建的 custom.onnx 模型，直接辨識原圖，不需要前處理。")
            }
        }
    }

    private var settingsSection: some View {
        Section {
            Picker("字元類型", selection: $mode) {
                ForEach(CharsetMode.allCases) { Text($0.rawValue).tag($0) }
            }
            if mode != .digits {
                Picker("大小寫", selection: $caseMode) {
                    ForEach(CaseMode.allCases) { Text($0.rawValue).tag($0) }
                }
            }
            Toggle("固定長度", isOn: $useLength)
            if useLength {
                Stepper("長度：\(length)", value: $length, in: 1...12)
            }
            Toggle("同時用快速模式辨識", isOn: $useFast)
            Toggle("多種前處理交叉投票", isOn: $useVariants)
        } header: {
            Text("辨識設定")
        } footer: {
            Text("投票需要開啟「固定長度」。交叉投票會多跑幾次辨識，準確率較高但稍慢。")
        }
    }

    private var preprocessSection: some View {
        Section("影像前處理") {
            Toggle("啟用前處理", isOn: $options.enabled)
            if options.enabled {
                VStack(alignment: .leading) {
                    Text("放大到高度：\(Int(options.targetHeight)) px")
                    Slider(value: $options.targetHeight, in: 60...400, step: 20)
                }
                VStack(alignment: .leading) {
                    Text("對比：\(options.contrast, specifier: "%.1f")")
                    Slider(value: $options.contrast, in: 0.5...3)
                }
                Toggle("去雜訊（中值濾波）", isOn: $options.denoise)
                Toggle("反相（深底淺字時開啟）", isOn: $options.invert)
                Toggle("二值化", isOn: $options.binarize)
                if options.binarize {
                    VStack(alignment: .leading) {
                        Text("門檻：\(options.threshold, specifier: "%.2f")")
                        Slider(value: $options.threshold, in: 0.1...0.9)
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private func load(_ image: UIImage) {
        original = image
        processed = nil
        candidates = []
        copied = nil
        errorMessage = nil
        imageVersion += 1
    }

    private func pasteImage() {
        if let image = UIPasteboard.general.image {
            load(image)
        } else {
            errorMessage = "剪貼簿裡沒有圖片"
        }
    }

    private func runOCR() async {
        guard let original else { return }

        // 簡單防抖：拖動滑桿時不要每一格都辨識
        try? await Task.sleep(for: .milliseconds(250))
        if Task.isCancelled { return }

        isRunning = true
        defer { isRunning = false }

        if engine == .custom {
            await runCustomModel(on: original)
            return
        }

        let opts = options
        let variants = useVariants
        let images: [UIImage] = await Task.detached(priority: .userInitiated) {
            var list = [ImagePreprocessor.process(original, options: opts)]
            if variants {
                // 二值化開關相反的版本
                var flipped = opts
                flipped.enabled = true
                flipped.binarize.toggle()
                list.append(ImagePreprocessor.process(original, options: flipped))
                // 不同放大倍率的版本
                var bigger = opts
                bigger.enabled = true
                bigger.targetHeight = opts.targetHeight * 1.5
                list.append(ImagePreprocessor.process(original, options: bigger))
                // 完全不處理的原圖
                var plain = opts
                plain.enabled = false
                list.append(ImagePreprocessor.process(original, options: plain))
            }
            return list
        }.value
        if Task.isCancelled { return }
        processed = images.first

        do {
            let result = try await OCRService.recognize(images: images, settings: settings)
            if Task.isCancelled { return }
            candidates = result
            errorMessage = result.isEmpty ? "沒有辨識到文字，試著調整前處理參數" : nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

extension ContentView {
    private func runCustomModel(on image: UIImage) async {
        processed = nil
        switch Self.model {
        case .failure(let error):
            candidates = []
            errorMessage = "模型載入失敗：\(error.localizedDescription)"
        case .success(let model):
            do {
                let text = try await Task.detached(priority: .userInitiated) {
                    try model.recognize(image)
                }.value
                if Task.isCancelled { return }
                candidates = text.isEmpty ? [] : [OCRCandidate(text: text, score: 1)]
                errorMessage = text.isEmpty ? "模型沒有辨識出文字" : nil
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

private struct ImageRow: View {
    let title: String
    let image: UIImage

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Image(uiImage: image)
                .resizable()
                .interpolation(.none)
                .scaledToFit()
                .frame(maxHeight: 120)
        }
    }
}
