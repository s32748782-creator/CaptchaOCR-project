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

struct OCRCandidate: Identifiable, Hashable {
    let id = UUID()
    let text: String
    let confidence: Float
}

enum OCRError: LocalizedError {
    case invalidImage
    var errorDescription: String? { "無法讀取影像" }
}

enum OCRService {

    /// 呼叫 Vision 做文字辨識，回傳依分數排序的候選答案
    static func recognize(_ image: UIImage,
                          mode: CharsetMode,
                          expectedLength: Int?,
                          uppercase: Bool) async throws -> [OCRCandidate] {
        guard let cgImage = image.cgImage else { throw OCRError.invalidImage }

        return try await Task.detached(priority: .userInitiated) {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate      // 準確模式
            request.usesLanguageCorrection = false    // 驗證碼不是單字，關掉自動校正
            request.recognitionLanguages = ["en-US"]

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])

            // 由左到右排序文字區塊
            let observations = (request.results ?? [])
                .sorted { $0.boundingBox.minX < $1.boundingBox.minX }

            return OCRService.buildCandidates(from: observations,
                                              mode: mode,
                                              expectedLength: expectedLength,
                                              uppercase: uppercase)
        }.value
    }

    static func buildCandidates(from observations: [VNRecognizedTextObservation],
                                mode: CharsetMode,
                                expectedLength: Int?,
                                uppercase: Bool) -> [OCRCandidate] {
        var raw: [(String, Float)] = []

        // A. 每個區塊取第一名，串起來（驗證碼字距大時會被切成多塊）
        let tops = observations.compactMap { $0.topCandidates(1).first }
        if !tops.isEmpty {
            let joined = tops.map(\.string).joined()
            let avg = tops.map(\.confidence).reduce(0, +) / Float(tops.count)
            raw.append((joined, avg))
        }

        // B. 每個區塊的前 5 名候選
        for observation in observations {
            for candidate in observation.topCandidates(5) {
                raw.append((candidate.string, candidate.confidence))
            }
        }

        var seen = Set<String>()
        var results: [OCRCandidate] = []
        for (text, confidence) in raw {
            var cleaned = clean(text, mode: mode)
            if uppercase && mode != .digits { cleaned = cleaned.uppercased() }
            guard !cleaned.isEmpty, seen.insert(cleaned).inserted else { continue }

            var score = confidence
            if let n = expectedLength, cleaned.count != n {
                score *= 0.5   // 長度不符就降分
            }
            results.append(OCRCandidate(text: cleaned, confidence: score))
        }
        return results.sorted { $0.confidence > $1.confidence }
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
                "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"
            ]
            t = String(t.map { map[$0] ?? $0 })
        case .alphanumeric:
            break
        }

        let allowed = mode.allowed
        return String(t.unicodeScalars.filter { allowed.contains($0) })
    }
}
