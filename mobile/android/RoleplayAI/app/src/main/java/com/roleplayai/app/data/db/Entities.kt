package com.roleplayai.app.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey val id: String,
    val title: String,
    val characterId: String,
    val createdAtMs: Long,
    val updatedAtMs: Long
)

@Entity(
    tableName = "messages",
    indices = [Index("sessionId")]
)
data class MessageEntity(
    @PrimaryKey val id: String,
    val sessionId: String,
    val role: String,
    val content: String,
    val createdAtMs: Long
)

@Entity(
    tableName = "memory_items",
    indices = [Index("sessionId")]
)
data class MemoryEntity(
    @PrimaryKey val id: String,
    val sessionId: String,
    val text: String,
    val score: Float,
    val createdAtMs: Long
)

@Entity(tableName = "characters")
data class CharacterEntity(
    @PrimaryKey val id: String,
    val name: String,
    val personality: String,
    val world: String,
    val tone: String
)

