package com.roleplayai.app.model

import kotlinx.serialization.Serializable

enum class Role { USER, ASSISTANT, SYSTEM }

@Serializable
data class CharacterProfile(
    val id: String,
    val name: String,
    val personality: String,
    val world: String,
    val tone: String,
    val avatarKey: String = "wizard"
)

data class ChatMessage(
    val id: String,
    val sessionId: String,
    val role: Role,
    val content: String,
    val createdAtMs: Long
)

data class ChatSession(
    val id: String,
    val title: String,
    val characterId: String,
    val createdAtMs: Long,
    val updatedAtMs: Long
)

data class MemoryItem(
    val id: String,
    val sessionId: String,
    val text: String,
    val score: Float,
    val createdAtMs: Long
)
