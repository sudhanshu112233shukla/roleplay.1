package com.roleplayai.app.repository

import com.roleplayai.app.data.db.CharacterDao
import com.roleplayai.app.data.db.CharacterEntity
import kotlinx.coroutines.flow.Flow

class CharacterRepository(
    private val characterDao: CharacterDao
) {
    fun observeCharacters(): Flow<List<CharacterEntity>> = characterDao.observeCharacters()

    suspend fun upsert(character: CharacterEntity) {
        characterDao.upsert(character)
    }

    suspend fun ensureDefaults() {
        // Simple baseline; user can create more in-app.
        characterDao.upsert(
            CharacterEntity(
                id = "wizard",
                name = "Wizard",
                personality = "Ancient magical teacher; wise and mysterious.",
                world = "Fantasy",
                tone = "Wise, descriptive, emotionally consistent."
            )
        )
        characterDao.upsert(
            CharacterEntity(
                id = "detective",
                name = "Detective",
                personality = "Sharp, skeptical, detail-oriented investigator.",
                world = "Noir city",
                tone = "Concise, probing questions, grounded."
            )
        )
    }

    suspend fun get(id: String): CharacterEntity? = characterDao.get(id)
}
