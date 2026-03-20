package com.roleplayai.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.data.db.CharacterEntity
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class CharacterViewModel(private val graph: AppGraph) : ViewModel() {
    val characters: StateFlow<List<CharacterEntity>> =
        graph.characters.observeCharacters().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val activeCharacterId: String get() = graph.activeCharacterId

    fun select(characterId: String) {
        graph.activeCharacterId = characterId
    }

    fun create(id: String, name: String, personality: String, world: String, tone: String) {
        viewModelScope.launch {
            val requested = id.trim()
            val base = slugify(if (requested.isNotEmpty()) requested else name)
            var uniqueId = base
            var i = 2
            while (graph.characters.get(uniqueId) != null) {
                uniqueId = "${base}_$i"
                i += 1
            }

            graph.characters.upsert(
                CharacterEntity(
                    id = uniqueId,
                    name = name,
                    personality = personality,
                    world = world,
                    tone = tone
                )
            )
            graph.activeCharacterId = uniqueId
        }
    }

    private fun slugify(s: String): String {
        val lowered = s.trim().lowercase()
        val sb = StringBuilder(lowered.length)
        var lastWasUnderscore = false
        for (ch in lowered) {
            val out = when {
                ch.isLetterOrDigit() -> ch
                ch == ' ' || ch == '-' || ch == '_' -> '_'
                else -> '_'
            }
            if (out == '_') {
                if (!lastWasUnderscore) sb.append('_')
                lastWasUnderscore = true
            } else {
                sb.append(out)
                lastWasUnderscore = false
            }
        }
        return sb.toString().trim('_').ifBlank { "character" }
    }
}
