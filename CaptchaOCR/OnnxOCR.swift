import UIKit
import onnxruntime_objc

/// 執行 dddd_trainer / ddddocr 匯出的自訂 ONNX 模型
/// 前處理與 ddddocr 相同：等比縮放到指定高度 → RGB → ImageNet 正規化 → NCHW
final class OnnxOCR: @unchecked Sendable {

    private struct Config: Decodable {
        let charset: [String]
        let image: [Int]      // [寬, 高]，寬 = -1 代表依比例縮放
        let word: Bool
        let channel: Int
    }

    enum ModelError: LocalizedError {
        case missingFile(String)
        case badImage
        case badOutput

        var errorDescription: String? {
            switch self {
            case .missingFile(let name): return "找不到模型檔案：\(name)"
            case .badImage: return "無法讀取影像"
            case .badOutput: return "模型輸出格式無法解析"
            }
        }
    }

    private let env: ORTEnv
    private let session: ORTSession
    private let inputName: String
    private let outputName: String
    private let charset: [String]
    private let height: Int
    private let fixedWidth: Int?
    private let channels: Int

    init(modelName: String = "custom", configName: String = "charsets") throws {
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "onnx") else {
            throw ModelError.missingFile("\(modelName).onnx")
        }
        guard let configURL = Bundle.main.url(forResource: configName, withExtension: "json") else {
            throw ModelError.missingFile("\(configName).json")
        }
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
        charset = config.charset
        height = config.image.count > 1 ? config.image[1] : 48
        fixedWidth = (config.image.first ?? -1) > 0 ? config.image[0] : nil
        channels = config.channel

        env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        inputName = try session.inputNames().first ?? "input1"
        outputName = try session.outputNames().first ?? "output"
    }

    func recognize(_ image: UIImage) throws -> String {
        let upright = ImagePreprocessor.normalized(image)
        guard let cg = upright.cgImage, cg.height > 0 else { throw ModelError.badImage }

        // 1. 等比縮放到模型需要的高度
        let h = height
        let w = fixedWidth ?? max(1, Int(Double(cg.width) * Double(h) / Double(cg.height)))

        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let ctx = CGContext(data: buffer.baseAddress,
                                      width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw ModelError.badImage }

        // 2. 正規化並排成 NCHW
        let plane = w * h
        var input = [Float](repeating: 0, count: plane * channels)
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        for i in 0..<plane {
            let r = Float(pixels[i * 4]) / 255
            let g = Float(pixels[i * 4 + 1]) / 255
            let b = Float(pixels[i * 4 + 2]) / 255
            if channels == 1 {
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                input[i] = (gray - 0.456) / 0.224
            } else {
                input[i] = (r - mean[0]) / std[0]
                input[plane + i] = (g - mean[1]) / std[1]
                input[2 * plane + i] = (b - mean[2]) / std[2]
            }
        }

        let data = NSMutableData(bytes: input, length: input.count * MemoryLayout<Float>.stride)
        let tensor = try ORTValue(tensorData: data,
                                  elementType: .float,
                                  shape: [1, NSNumber(value: channels), NSNumber(value: h), NSNumber(value: w)])

        // 3. 推論
        let outputs = try session.run(withInputs: [inputName: tensor],
                                      outputNames: [outputName],
                                      runOptions: nil)
        guard let output = outputs[outputName] else { throw ModelError.badOutput }
        let info = try output.tensorTypeAndShapeInfo()
        let raw = try output.tensorData() as Data

        // 4. 取出每個時間步的字元索引
        var indices: [Int] = []
        switch info.elementType {
        case .int64:
            indices = raw.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)).map { Int($0) } }
        case .int32:
            indices = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)).map { Int($0) } }
        case .float:
            // 若模型輸出的是機率而非索引，自己做 argmax
            let classes = info.shape.last?.intValue ?? charset.count
            let probs = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            guard classes > 0 else { throw ModelError.badOutput }
            stride(from: 0, to: probs.count - classes + 1, by: classes).forEach { start in
                let row = probs[start..<start + classes]
                if let best = row.indices.max(by: { row[$0] < row[$1] }) {
                    indices.append(best - start)
                }
            }
        default:
            throw ModelError.badOutput
        }

        // 5. CTC 解碼：合併連續重複、去掉空白（索引 0）
        var result = ""
        var last = 0
        for index in indices {
            if index != last && index != 0 && index < charset.count {
                result += charset[index]
            }
            last = index
        }
        return result
    }
}
