import Foundation

struct ExtractedArticle {
    let title: String
    let content: String
    let date: String?
    let author: String?
    let url: String

    var summaryPrompt: String {
        var meta = "제목: \(title)"
        if let author { meta += "\n기자: \(author)" }
        if let date   { meta += "\n날짜: \(date)" }

        return """
        다음은 뉴스 기사의 본문입니다. 반드시 아래 지침을 따라 내용을 정리하세요.

        [지침]
        • 기사에 담긴 사실과 구체적 정보(인명·기관명·수치·날짜)만 전달합니다.
        • 댓글 정책, SNS 공유 안내, 관련기사·추천기사 링크, 저작권 고지, 광고 문구는 완전히 무시합니다.
        • "이 기사는", "요약하자면", "결론적으로" 같은 메타 표현 없이 사실을 직접 서술합니다.
        • 독자에게 처음 전달하는 뉴스 브리핑처럼, 핵심 사실을 간결하고 명확하게 작성합니다.
        • 불확실하거나 추측성 내용은 그대로 불확실하다고 표현합니다.
        • 출력은 한국어로만 작성합니다.

        \(meta)

        \(content)
        """
    }
}

enum ArticleExtractorError: LocalizedError {
    case invalidURL, downloadFailed, encodingFailed, contentNotFound

    var errorDescription: String? {
        switch self {
        case .invalidURL:      return "유효하지 않은 URL입니다."
        case .downloadFailed:  return "페이지를 불러오지 못했습니다."
        case .encodingFailed:  return "페이지 인코딩 변환에 실패했습니다."
        case .contentNotFound: return "기사 본문을 찾을 수 없습니다."
        }
    }
}

// MARK: - ArticleExtractor

enum ArticleExtractor {

    static func extract(from urlString: String) async throws -> ExtractedArticle {
        guard let url = URL(string: urlString) else { throw ArticleExtractorError.invalidURL }

        var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        req.setValue(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
            forHTTPHeaderField: "User-Agent"
        )
        req.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        req.setValue("ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await URLSession.shared.data(for: req)
        guard !data.isEmpty else { throw ArticleExtractorError.downloadFailed }

        let html = decodeHTML(data: data, response: response)
        guard !html.isEmpty else { throw ArticleExtractorError.encodingFailed }

        let title   = extractTitle(html)
        let author  = extractAuthor(html)
        let date    = extractDate(html)
        let content = extractContent(html)

        guard !content.isEmpty else { throw ArticleExtractorError.contentNotFound }

        return ExtractedArticle(title: title, content: content, date: date, author: author, url: urlString)
    }

    // MARK: - Encoding Detection

    private static func decodeHTML(data: Data, response: URLResponse?) -> String {
        // 1. Content-Type header
        if let http = response as? HTTPURLResponse,
           let ct = http.value(forHTTPHeaderField: "Content-Type"),
           let cs = charset(from: ct) {
            if let s = decode(data, ianaName: cs) { return s }
        }
        // 2. UTF-8 fast path
        if let s = String(data: data, encoding: .utf8) {
            if s.contains("euc-kr") || s.contains("EUC-KR") || s.contains("ks_c_5601") {
                return decode(data, ianaName: "EUC-KR") ?? s
            }
            return s
        }
        // 3. EUC-KR fallback (Korean sites)
        return decode(data, ianaName: "EUC-KR") ?? ""
    }

    private static func charset(from contentType: String) -> String? {
        guard let r = contentType.range(of: "charset=", options: .caseInsensitive) else { return nil }
        return String(contentType[r.upperBound...]).components(separatedBy: CharacterSet(charactersIn: "; ")).first
    }

    private static func decode(_ data: Data, ianaName: String) -> String? {
        let cfEncoding = CFStringConvertIANACharSetNameToEncoding(ianaName as CFString)
        guard cfEncoding != kCFStringEncodingInvalidId else { return nil }
        let nsEncoding = CFStringConvertEncodingToNSStringEncoding(cfEncoding)
        return String(data: data, encoding: String.Encoding(rawValue: nsEncoding))
    }

    // MARK: - Metadata

    private static func extractTitle(_ html: String) -> String {
        let patterns = [
            #"<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\'<>]+)["\']"#,
            #"<meta[^>]+content=["\']([^"\'<>]+)["\'][^>]+property=["\']og:title["\']"#,
            #"<title[^>]*>([^<]+)</title>"#,
            #"<h1[^>]*>([^<]+)</h1>"#,
        ]
        for p in patterns {
            if let v = first(in: html, pattern: p) {
                let t = v.strippingTags().htmlDecoded.trimmed
                if !t.isEmpty { return t }
            }
        }
        return "제목 없음"
    }

    private static func extractAuthor(_ html: String) -> String? {
        let patterns = [
            #"<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\'<>]+)["\']"#,
            #"<meta[^>]+content=["\']([^"\'<>]+)["\'][^>]+name=["\']author["\']"#,
            #"<meta[^>]+property=["\']article:author["\'][^>]+content=["\']([^"\'<>]+)["\']"#,
            #"class=["\'][^"\']*(?:byline|reporter|journalist|author)[^"\']*["\'][^>]*>([^<]{2,40})<"#,
        ]
        for p in patterns {
            if let v = first(in: html, pattern: p) {
                let t = v.strippingTags().htmlDecoded.trimmed
                if !t.isEmpty && t.count < 50 { return t }
            }
        }
        return nil
    }

    private static func extractDate(_ html: String) -> String? {
        let patterns = [
            #"<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']"#,
            #"<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']"#,
            #"<time[^>]+datetime=["\']([^"\']+)["\']"#,
            #""datePublished"\s*:\s*"([^"]+)""#,
        ]
        for p in patterns {
            if let v = first(in: html, pattern: p)?.trimmed, !v.isEmpty { return v }
        }
        return nil
    }

    // MARK: - Content Extraction (Readability-like)

    private static func extractContent(_ html: String) -> String {
        let cleaned = removeNoise(html)

        // Priority 1: <article> container
        if let text = textFromContainer(cleaned, tag: "article"), text.count > 200 {
            return postClean(text)
        }
        // Priority 2: <main> container
        if let text = textFromContainer(cleaned, tag: "main"), text.count > 200 {
            return postClean(text)
        }
        // Priority 3: div with content/article/body class
        if let text = textFromContentDiv(cleaned), text.count > 200 {
            return postClean(text)
        }
        // Fallback: score all <p> blocks
        return postClean(paragraphFallback(cleaned))
    }

    private static func removeNoise(_ html: String) -> String {
        var h = html
        // Remove self-contained noise blocks
        let blockTags = ["script", "style", "nav", "header", "footer", "aside",
                         "noscript", "form", "iframe", "figure"]
        for tag in blockTags {
            h = h.replacingOccurrences(
                of: "<\(tag)[\\s\\S]*?</\(tag)>",
                with: " ", options: [.regularExpression, .caseInsensitive]
            )
        }
        // Remove comments
        h = h.replacingOccurrences(of: "<!--[\\s\\S]*?-->", with: " ", options: [.regularExpression])
        return h
    }

    private static func textFromContainer(_ html: String, tag: String) -> String? {
        guard let inner = first(in: html, pattern: "<\(tag)[^>]*>([\\s\\S]*?)</\(tag)>") else { return nil }
        let paragraphs = all(in: inner, pattern: #"<(?:p|h[1-6])[^>]*>([\s\S]*?)</(?:p|h[1-6])>"#)
            .map { $0.strippingTags().htmlDecoded.trimmed }
            .filter { $0.count > 20 }
        let joined = paragraphs.joined(separator: "\n\n")
        return joined.isEmpty ? nil : joined
    }

    private static func textFromContentDiv(_ html: String) -> String? {
        let pattern = #"<div[^>]+(?:class|id)=["\'][^"\']*(?:article|content|article-body|news-body|story)[^"\']*["\'][^>]*>([\s\S]*?)</div>"#
        guard let inner = first(in: html, pattern: pattern) else { return nil }
        let paragraphs = all(in: inner, pattern: #"<p[^>]*>([\s\S]*?)</p>"#)
            .map { $0.strippingTags().htmlDecoded.trimmed }
            .filter { $0.count > 20 }
        let joined = paragraphs.joined(separator: "\n\n")
        return joined.isEmpty ? nil : joined
    }

    // MARK: - Post-processing

    static func postClean(_ text: String) -> String {
        // Keywords that indicate noise lines (Korean news boilerplate)
        let noiseKeywords: [String] = [
            // Copyright / legal
            "무단 전재", "재배포 금지", "저작권", "ⓒ", "©", "All rights reserved",
            // Comment / community
            "댓글 정책", "댓글을 달려면", "로그인 후", "욕설·비방", "명예훼손",
            // Social / share
            "SNS 공유", "카카오스토리", "트위터로 공유", "페이스북 공유",
            "URL 복사", "스크랩", "인쇄하기", "공유하기",
            // Subscription / notification
            "구독 신청", "뉴스레터", "알림 받기", "구독하기",
            // Related articles UI text
            "관련기사", "추천기사", "더 보기", "영상으로 보기",
            // Common filler
            "기사를 읽어드립니다", "TTS",
        ]

        let lines = text.components(separatedBy: "\n")
        let filtered = lines.filter { line in
            let t = line.trimmed
            guard t.count >= 10 else { return false }   // 너무 짧은 줄 제거
            for kw in noiseKeywords {
                if t.contains(kw) { return false }
            }
            return true
        }

        // Collapse 3+ consecutive blank lines into 2
        var result: [String] = []
        var blankCount = 0
        for line in filtered {
            if line.trimmed.isEmpty {
                blankCount += 1
                if blankCount <= 2 { result.append(line) }
            } else {
                blankCount = 0
                result.append(line)
            }
        }
        return result.joined(separator: "\n").trimmed
    }

    private static func paragraphFallback(_ html: String) -> String {
        // Score each <p> by text length and link density
        struct Candidate { let text: String; let score: Double }

        let candidates: [Candidate] = all(in: html, pattern: #"<p[^>]*>([\s\S]*?)</p>"#)
            .compactMap { block -> Candidate? in
                let text = block.strippingTags().htmlDecoded.trimmed
                guard text.count > 30 else { return nil }
                let linkText = all(in: block, pattern: #"<a[^>]*>([\s\S]*?)</a>"#)
                    .map { $0.strippingTags() }.joined()
                let linkDensity = Double(linkText.count) / Double(text.count)
                guard linkDensity < 0.5 else { return nil }
                let score = Double(text.count) * (1 - linkDensity)
                return Candidate(text: text, score: score)
            }

        // Group by score clusters, return best consecutive group
        var groups: [[Candidate]] = []
        var current: [Candidate] = []
        for c in candidates {
            if c.score > 50 {
                current.append(c)
            } else {
                if !current.isEmpty { groups.append(current); current = [] }
            }
        }
        if !current.isEmpty { groups.append(current) }

        let best = groups.max(by: { a, b in
            a.reduce(0, { $0 + $1.score }) < b.reduce(0, { $0 + $1.score })
        })
        return best?.map(\.text).joined(separator: "\n\n") ?? ""
    }

    // MARK: - Regex Helpers

    private static func first(in text: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else { return nil }
        let ns = text as NSString
        guard let m = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              m.numberOfRanges > 1,
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return String(text[r])
    }

    private static func all(in text: String, pattern: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else { return [] }
        let ns = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).compactMap { m in
            let idx = m.numberOfRanges > 1 ? 1 : 0
            guard let r = Range(m.range(at: idx), in: text) else { return nil }
            return String(text[r])
        }
    }
}

// MARK: - String Helpers

private extension String {
    func strippingTags() -> String {
        self.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    var htmlDecoded: String {
        var s = self
        let map: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&apos;", "'"), ("&nbsp;", " "),
            ("&#39;", "'"), ("&middot;", "·"), ("&hellip;", "…"),
        ]
        for (e, c) in map { s = s.replacingOccurrences(of: e, with: c) }
        // Numeric entities &#NNN;
        if let r = try? NSRegularExpression(pattern: #"&#(\d+);"#) {
            let ns = s as NSString
            let matches = r.matches(in: s, range: NSRange(location: 0, length: ns.length)).reversed()
            for m in matches {
                if let numRange = Range(m.range(at: 1), in: s),
                   let code = UInt32(s[numRange]),
                   let scalar = Unicode.Scalar(code) {
                    s.replaceSubrange(Range(m.range, in: s)!, with: String(scalar))
                }
            }
        }
        return s
    }

    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
