import UIKit
import Vision

enum CharsetMode: String, CaseIterable, Identifiable {
    case digits = "純數字"
    case alphanumeric = "英數混合"
    case letters = "純英文"

    var id: String { rawValue }

    private static let digitChars = "0123456789"
    private static let letterChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    var allowed: CharacterSet {
        switch self {
        case .digits: return CharacterSet(charactersIn: Self.digitChars)
        case .letters: return CharacterSet(charactersIn: Self.letterChars)
        case .alphanumeric: return CharacterSet(charactersIn: Self.digitChars + Self.letterChars)
        }
    }
}

enum CaseMode: String, CaseIterable, Identifiable {
    case keep = "不轉換"
    case lower = "全小寫"
    case upper = "全大寫"
    var id: String { rawValue }
}

struct OCRSettings: Equatable {
    var mode: CharsetMode
    var expectedLength: Int?
    var caseMode: CaseMode
    var useFastLevel: Bool
}

struct OCRCandidate: Identifiable, Hashable {
    let id = UUID()
    let text: String
    let score: Float
    var votes: Int = 1
    var isVote = false
}

enum OCRError: LocalizedError {
    case invalidImage
    var errorDescription: String? { "無法讀取影像" }
}

enum OCRService {

    /// 對多張（不同前處理的）影像、多種辨識模式各跑一次，再合併投票
    static func recognize(images: [UIImage], settings: OCRSettings) async throws -> [OCRCandidate] {
        let cgImages = images.compactMap(\.cgImage)
        guard !cgImages.isEmpty else { throw OCRError.invalidImage }

        return try await Task.detached(priority: .userInitiated) {
            let levels: [VNRequestTextRecognitionLevel] =
                settings.useFastLevel ? [.accurate, .fast] : [.accurate]
            var raw: [(String, Float)] = []
            for cg in cgImages {
                for level in levels {
                    raw += try OCRService.rawCandidates(cgImage: cg, level: level)
                }
            }
            return OCRService.rank(raw, settings: settings)
        }.value
    }

    /// 單次 Vision 辨識，回傳未清理的候選字串
    static func rawCandidates(cgImage: CGImage,
                              level: VNRequestTextRecognitionLevel) throws -> [(String, Float)] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = level
        request.usesLanguageCorrection = false    // 驗證碼不是單字，關掉自動校正
        request.recognitionLanguages = ["en-US"]

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let observations = (request.results ?? [])
            .sorted { $0.boundingBox.minX < $1.boundingBox.minX }

        var raw: [(String, Float)] = []
        // 每個區塊取第一名串起來（字距大時會被切成多塊）
        let tops = observations.compactMap { $0.topCandidates(1).first }
        if !tops.isEmpty {
            let avg = tops.map(\.confidence).reduce(0, +) / Float(tops.count)
            raw.append((tops.map(\.string).joined(), avg))
        }
        // 每個區塊的前 5 名候選
        for observation in observations {
            for candidate in observation.topCandidates(5) {
                raw.append((candidate.string, candidate.confidence))
            }
        }
        return raw
    }

    /// 清理、合併相同答案、逐字投票
    static func rank(_ raw: [(String, Float)], settings: OCRSettings) -> [OCRCandidate] {
        var scores: [String: Float] = [:]
        var votes: [String: Int] = [:]

        for (text, confidence) in raw {
            let cleaned = normalize(text, settings: settings)
            guard !cleaned.isEmpty else { continue }
            var score = max(confidence, 0.05)
            if let n = settings.expectedLength, cleaned.count != n {
                score *= 0.3   // 長度不符大幅降分
            }
            scores[cleaned, default: 0] += score
            votes[cleaned, default: 0] += 1
        }

        var results = scores.map {
            OCRCandidate(text: $0.key, score: $0.value, votes: votes[$0.key] ?? 1)
        }
        results.sort { $0.score > $1.score }

        // 逐字投票：所有長度正確的候選，每個位置分別選得分最高的字元
        if let n = settings.expectedLength {
            let fits = results.filter { $0.text.count == n }
            if fits.count >= 2 {
                var columns = Array(repeating: [Character: Float](), count: n)
                for candidate in fits {
                    for (i, ch) in candidate.text.enumerated() {
                        columns[i][ch, default: 0] += candidate.score
                    }
                }
                let voted = String(columns.compactMap { col in
                    col.max { $0.value < $1.value }?.key
                })
                if voted.count == n {
                    results.removeAll { $0.text == voted }
                    let total = fits.reduce(0) { $0 + $1.votes }
                    results.insert(OCRCandidate(text: voted, score: .infinity,
                                                votes: total, isVote: true), at: 0)
                }
            }
        }
        return results
    }

    static func normalize(_ text: String, settings: OCRSettings) -> String {
        let cleaned = clean(text, mode: settings.mode)
        switch settings.caseMode {
        case .keep: return cleaned
        case .lower: return cleaned.lowercased()
        case .upper: return cleaned.uppercased()
        }
    }

    /// 修正常見誤判並過濾不允許的字元
    static func clean(_ text: String, mode: CharsetMode) -> String {
        var t = text.replacingOccurrences(of: " ", with: "")

        switch mode {
        case .digits:
            let map: [Character: Character] = [
                "O": "0", "o": "0", "D": "0", "Q": "0",
                "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
                "Z": "2", "z": "2", "S": "5", "s": "5",
                "B": "8", "G": "6", "b": "6", "T": "7",
                "g": "9", "q": "9"
            ]
            t = String(t.map { map[$0] ?? $0 })
        case .letters:
            let map: [Character: Character] = [
                "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "|": "l"
            ]
            t = String(t.map { map[$0] ?? $0 })
        case .alphanumeric:
            break
        }

        let allowed = mode.allowed
        return String(t.unicodeScalars.filter { allowed.contains($0) })
    }
}
