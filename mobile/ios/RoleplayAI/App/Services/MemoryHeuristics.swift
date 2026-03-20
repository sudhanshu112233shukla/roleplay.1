import Foundation

struct MemoryCandidate {
    let text: String
    let score: Double
}

enum MemoryHeuristics {
    private static let patterns: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: "\\bmy name is\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bi am\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bi live\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bi like\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bremember\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bnever forget\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bsecret\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bquest\\b", options: [.caseInsensitive]),
        try! NSRegularExpression(pattern: "\\bartifact\\b", options: [.caseInsensitive]),
    ]

    static func candidate(from userText: String) -> MemoryCandidate? {
        let t = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return nil }

        var score = 0.0
        let range = NSRange(location: 0, length: (t as NSString).length)
        for p in patterns {
            if p.firstMatch(in: t, options: [], range: range) != nil { score += 0.2 }
        }
        if t.count > 80 { score += 0.1 }
        if !t.contains("?") { score += 0.1 }
        guard score >= 0.3 else { return nil }
        return MemoryCandidate(text: t, score: min(score, 1.0))
    }
}

