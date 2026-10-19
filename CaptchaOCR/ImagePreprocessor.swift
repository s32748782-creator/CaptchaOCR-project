import UIKit
import CoreImage
import CoreImage.CIFilterBuiltins

/// 前處理參數：驗證碼通常很小、有雜訊、有彩色干擾線，先處理過再丟給 OCR 會準很多
struct PreprocessOptions: Equatable {
    var enabled = true
    var targetHeight: CGFloat = 160   // 太小的圖先放大到這個高度
    var contrast: Float = 1.6         // 對比強度
    var denoise = true                // 中值濾波去雜點
    var binarize = false              // 二值化（黑白）
    var threshold: Float = 0.5        // 二值化門檻
    var invert = false                // 深底淺字時反相
}

enum ImagePreprocessor {
    private static let context = CIContext()

    /// 把照片方向轉正，避免 OCR 讀到旋轉的圖
    static func normalized(_ image: UIImage) -> UIImage {
        if image.imageOrientation == .up { return image }
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        return UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }

    static func process(_ image: UIImage, options: PreprocessOptions) -> UIImage {
        let upright = normalized(image)
        guard options.enabled, let cg = upright.cgImage else { return upright }
        var ci = CIImage(cgImage: cg)

        // 1. 放大（Lanczos 插值，保留邊緣）
        let height = ci.extent.height
        if height < options.targetHeight {
            let scale = CIFilter.lanczosScaleTransform()
            scale.inputImage = ci
            scale.scale = Float(options.targetHeight / height)
            scale.aspectRatio = 1
            ci = scale.outputImage ?? ci
        }

        // 2. 灰階 + 提高對比
        let color = CIFilter.colorControls()
        color.inputImage = ci
        color.saturation = 0
        color.contrast = options.contrast
        ci = color.outputImage ?? ci

        // 3. 去雜訊
        if options.denoise {
            let median = CIFilter.median()
            median.inputImage = ci
            ci = median.outputImage ?? ci
        }

        // 4. 反相
        if options.invert {
            let invert = CIFilter.colorInvert()
            invert.inputImage = ci
            ci = invert.outputImage ?? ci
        }

        // 5. 二值化
        if options.binarize {
            let threshold = CIFilter.colorThreshold()
            threshold.inputImage = ci
            threshold.threshold = options.threshold
            ci = threshold.outputImage ?? ci
        }

        // 6. 加白邊，文字貼邊時 Vision 比較容易漏字
        let extent = ci.extent.integral
        let background = CIImage(color: .white).cropped(to: extent.insetBy(dx: -24, dy: -24))
        ci = ci.cropped(to: extent).composited(over: background)

        guard let output = context.createCGImage(ci, from: ci.extent) else { return upright }
        return UIImage(cgImage: output)
    }
}
