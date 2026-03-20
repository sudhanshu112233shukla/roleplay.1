package com.roleplayai.app.repository

import com.roleplayai.app.data.db.MemoryDao
import com.roleplayai.app.data.db.MemoryEntity
import kotlinx.coroutines.flow.Flow
import java.util.UUID

class MemoryRepository(
    private val memoryDao: MemoryDao
) {
    fun observeMemory(sessionId: String): Flow<List<MemoryEntity>> = memoryDao.observeMemory(sessionId)

    suspend fun addMemory(sessionId: String, text: String, score: Float) {
        memoryDao.insert(
            MemoryEntity(
                id = UUID.randomUUID().toString(),
                sessionId = sessionId,
                text = text,
                score = score,
                createdAtMs = System.currentTimeMillis()
            )
        )
    }

    suspend fun deleteMemory(id: String) {
        memoryDao.delete(id)
    }

    suspend fun top(sessionId: String, limit: Int): List<MemoryEntity> {
        return memoryDao.top(sessionId, limit)
    }
}
