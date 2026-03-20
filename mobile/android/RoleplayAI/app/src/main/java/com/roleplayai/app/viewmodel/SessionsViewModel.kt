package com.roleplayai.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.data.db.SessionEntity
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SessionsViewModel(private val graph: AppGraph) : ViewModel() {
    val sessions: StateFlow<List<SessionEntity>> =
        graph.sessions.observeSessions().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    init {
        viewModelScope.launch { graph.characters.ensureDefaults() }
    }

    fun createAndOpenSession(onOpened: () -> Unit) {
        viewModelScope.launch {
            val sessionId = graph.sessions.createSession(characterId = graph.activeCharacterId)
            graph.activeSessionId = sessionId
            onOpened()
        }
    }

    fun openSession(sessionId: String, onOpened: () -> Unit) {
        graph.activeSessionId = sessionId
        onOpened()
    }

    fun deleteSession(sessionId: String) {
        viewModelScope.launch { graph.sessions.deleteSession(sessionId) }
    }
}

