package com.roleplayai.app.ai

private val patterns = listOf(
    Regex("\\bmy name is\\b", RegexOption.IGNORE_CASE),
    Regex("\\bi am\\b", RegexOption.IGNORE_CASE),
    Regex("\\bi live\\b", RegexOption.IGNORE_CASE),
    Regex("\\bi like\\b", RegexOption.IGNORE_CASE),
    Regex("\\bremember\\b", RegexOption.IGNORE_CASE),
    Regex("\\bnever forget\\b", RegexOption.IGNORE_CASE),
    Regex("\\bsecret\\b", RegexOption.IGNORE_CASE),
    Regex("\\bquest\\b", RegexOption.IGNORE_CASE),
    Regex("\\bartifact\\b", RegexOption.IGNORE_CASE),
)

data class MemoryCandidate(val text: String, val score: Float)

object MemoryHeuristics {
    fun candidate(userText: String): MemoryCandidate? {
        val t = userText.trim()
        if (t.isEmpty()) return null
        var score = 0f
        for (p in patterns) {
            if (p.containsMatchIn(t)) score += 0.2f
        }
        if (t.length > 80) score += 0.1f
        if (!t.contains("?")) score += 0.1f
        return if (score >= 0.3f) MemoryCandidate(t, score.coerceAtMost(1f)) else null
    }
}

