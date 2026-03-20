import Foundation
import UniformTypeIdentifiers

enum AttachmentService {
    static func readText(from url: URL, maxBytes: Int = 64_000) -> String {
        guard let data = try? Data(contentsOf: url) else { return "" }
        let capped = data.count > maxBytes ? data.prefix(maxBytes) : data
        return String(data: capped, encoding: .utf8) ?? ""
    }
}

