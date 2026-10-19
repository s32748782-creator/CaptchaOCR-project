import SwiftUI
import PhotosUI

/// 任何設定改變都會自動重新辨識
private struct RunKey: Equatable {
    var version: Int
    var mode: CharsetMode
    var length: Int?
    var uppercase: Bool
    var options: PreprocessOptions
}

struct ContentView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var original: UIImage?
    @State private var processed: UIImage?
    @State private var imageVersion = 0

    @State private var mode: CharsetMode = .alphanumeric
    @State private var useLength = true
    @State private var length = 4
    @State private var uppercase = true
    @State private var options = PreprocessOptions()

    @State private var candidates: [OCRCandidate] = []
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var showCamera = false
    @State private var copied: String?

    private var runKey: RunKey {
        RunKey(version: imageVersion, mode: mode,
               length: useLength ? length : nil,
               uppercase: uppercase, options: options)
    }

    var body: some View {
        NavigationStack {
            Form {
                sourceSection
                resultSection
                settingsSection
                preprocessSection
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
            if let processed, options.enabled { ImageRow(title: "前處理後（實際送去辨識的圖）", image: processed) }
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
                        Spacer()
                        Text(String(format: "%.0f%%", candidate.confidence * 100))
                            .foregroundStyle(.secondary)
                        Image(systemName: copied == candidate.text ? "checkmark" : "doc.on.doc")
                    }
                }
            }
        } header: {
            Text("結果（點一下複製）")
        }
    }

    private var settingsSection: some View {
        Section("辨識設定") {
            Picker("字元類型", selection: $mode) {
                ForEach(CharsetMode.allCases) { Text($0.rawValue).tag($0) }
            }
            Toggle("固定長度", isOn: $useLength)
            if useLength {
                Stepper("長度：\(length)", value: $length, in: 1...12)
            }
            if mode != .digits {
                Toggle("轉成大寫", isOn: $uppercase)
            }
        }
    }

    private var preprocessSection: some View {
        Section("影像前處理") {
            Toggle("啟用前處理", isOn: $options.enabled)
            if options.enabled {
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
        try? await Task.sleep(for: .milliseconds(200))
        if Task.isCancelled { return }

        isRunning = true
        defer { isRunning = false }

        let opts = options
        let image = await Task.detached(priority: .userInitiated) {
            ImagePreprocessor.process(original, options: opts)
        }.value
        if Task.isCancelled { return }
        processed = image

        do {
            let result = try await OCRService.recognize(
                image,
                mode: mode,
                expectedLength: useLength ? length : nil,
                uppercase: uppercase
            )
            if Task.isCancelled { return }
            candidates = result
            errorMessage = result.isEmpty ? "沒有辨識到文字，試著調整前處理參數" : nil
        } catch {
            errorMessage = error.localizedDescription
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
