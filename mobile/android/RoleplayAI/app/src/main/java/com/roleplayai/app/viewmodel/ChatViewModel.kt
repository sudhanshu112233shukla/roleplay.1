package com.roleplayai.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.roleplayai.app.ai.AiEngine
import com.roleplayai.app.ai.GenerationSettings
import com.roleplayai.app.ai.MemoryHeuristics
import com.roleplayai.app.ai.PromptAssembler
import com.roleplayai.app.ai.WorldState
import com.roleplayai.app.ai.EmotionState
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.data.db.CharacterEntity
import com.roleplayai.app.data.db.MessageEntity
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ChatUiState(
    val isLocal: Boolean = true,
    val engineName: String = "Local",
    val streamingText: String = "",
    val isStreaming: Boolean = false
)

class ChatViewModel(private val graph: AppGraph) : ViewModel() {
    private var streamJob: Job? = null

    private val _activeSessionId = MutableStateFlow(graph.activeSessionId)
    val activeSessionId: StateFlow<String?> = _activeSessionId.asStateFlow()

    val messages: StateFlow<List<MessageEntity>> = _activeSessionId
        .flatMapLatest { sid ->
            if (sid.isNullOrBlank()) {
                kotlinx.coroutines.flow.flowOf(emptyList())
            } else {
                graph.sessions.observeMessages(sid)
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    fun refreshSession() {
        _activeSessionId.value = graph.activeSessionId
    }

    fun send(userText: String) {
        val sid = graph.activeSessionId ?: return
        streamJob?.cancel()

        viewModelScope.launch {
            graph.sessions.addMessage(sid, "user", userText)
        }

        // Engine selection is settings-driven; default to llama.cpp id (falls back to MockEngine until wired).
        val engine: AiEngine = graph.aiEngines.create("llamacpp")

        streamJob = viewModelScope.launch {
            val character = graph.characters.get(graph.activeCharacterId) ?: CharacterEntity(
                id = graph.activeCharacterId,
                name = "Character",
                personality = "roleplay",
                world = "world",
                tone = "tone"
            )

            val memories = graph.memory.top(sid, limit = 12).map { it.text }
            val recent = graph.sessions.recentMessages(sid, limit = 24).reversed()
            val pairs = mutableListOf<Pair<String, String>>()
            var pendingUser: String? = null
            for (m in recent) {
                if (m.role == "user") pendingUser = m.content
                if (m.role == "assistant" && pendingUser != null) {
                    pairs.add(pendingUser!! to m.content)
                    pendingUser = null
                }
            }

            val prompt = PromptAssembler.build(
                character = character,
                world = WorldState(),
                emotion = EmotionState(),
                memories = memories,
                history = pairs,
                userInput = userText
            )

            val buf = StringBuilder()
            engine.generateStream(prompt, GenerationSettings()).collect { token ->
                buf.append(token)
                // UI consumes streaming text directly.
                _state.value = _state.value.copy(
                    streamingText = buf.toString(),
                    isStreaming = true,
                    engineName = engine.name,
                )
            }
            val finalText = buf.toString().trim()
            _state.value = _state.value.copy(streamingText = "", isStreaming = false)
            if (finalText.isNotEmpty()) {
                graph.sessions.addMessage(sid, "assistant", finalText)
            }

            val cand = MemoryHeuristics.candidate(userText)
            if (cand != null) {
                graph.memory.addMemory(sid, cand.text, cand.score)
            }
        }
    }
}
