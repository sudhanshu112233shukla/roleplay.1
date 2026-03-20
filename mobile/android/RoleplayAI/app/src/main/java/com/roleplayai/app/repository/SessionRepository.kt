package com.roleplayai.app.repository

import com.roleplayai.app.data.db.MessageDao
import com.roleplayai.app.data.db.MessageEntity
import com.roleplayai.app.data.db.SessionDao
import com.roleplayai.app.data.db.SessionEntity
import kotlinx.coroutines.flow.Flow
import java.util.UUID

class SessionRepository(
    private val sessionDao: SessionDao,
    private val messageDao: MessageDao
) {
    fun observeSessions(): Flow<List<SessionEntity>> = sessionDao.observeSessions()

    fun observeMessages(sessionId: String): Flow<List<MessageEntity>> = messageDao.observeMessages(sessionId)

    suspend fun createSession(characterId: String, title: String = "New chat"): String {
        val now = System.currentTimeMillis()
        val id = UUID.randomUUID().toString()
        sessionDao.upsert(
            SessionEntity(
                id = id,
                title = title,
                characterId = characterId,
                createdAtMs = now,
                updatedAtMs = now
            )
        )
        return id
    }

    suspend fun renameSession(sessionId: String, title: String) {
        val existing = sessionDao.get(sessionId) ?: return
        sessionDao.upsert(existing.copy(title = title, updatedAtMs = System.currentTimeMillis()))
    }

    suspend fun deleteSession(sessionId: String) {
        messageDao.deleteBySession(sessionId)
        sessionDao.delete(sessionId)
    }

    suspend fun addMessage(sessionId: String, role: String, content: String) {
        val now = System.currentTimeMillis()
        messageDao.insert(
            MessageEntity(
                id = UUID.randomUUID().toString(),
                sessionId = sessionId,
                role = role,
                content = content,
                createdAtMs = now
            )
        )
        // UpdatedAt is managed on next app load; keep minimal for now.
    }

    suspend fun recentMessages(sessionId: String, limit: Int): List<MessageEntity> {
        return messageDao.recent(sessionId, limit)
    }
}
