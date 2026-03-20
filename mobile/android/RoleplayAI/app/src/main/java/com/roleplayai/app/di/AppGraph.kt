package com.roleplayai.app.di

import android.content.Context
import com.roleplayai.app.ai.AiEngineFactory
import com.roleplayai.app.data.db.AppDatabase
import com.roleplayai.app.repository.CharacterRepository
import com.roleplayai.app.repository.MemoryRepository
import com.roleplayai.app.repository.SessionRepository

class AppGraph(context: Context) {
    private val db: AppDatabase = AppDatabase.create(context.applicationContext)

    val sessions: SessionRepository = SessionRepository(db.sessionDao(), db.messageDao())
    val memory: MemoryRepository = MemoryRepository(db.memoryDao())
    val characters: CharacterRepository = CharacterRepository(db.characterDao())
    val aiEngines: AiEngineFactory = AiEngineFactory(context.applicationContext)

    // Simple in-memory navigation state for the sample app. Replace with SavedStateHandle in production.
    var activeSessionId: String? = null
    var activeCharacterId: String = "wizard"
}
