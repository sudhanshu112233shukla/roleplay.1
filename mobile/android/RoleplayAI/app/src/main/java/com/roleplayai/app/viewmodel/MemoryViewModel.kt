package com.roleplayai.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.data.db.MemoryEntity
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class MemoryViewModel(private val graph: AppGraph) : ViewModel() {
    private val sessionIdFlow = kotlinx.coroutines.flow.MutableStateFlow(graph.activeSessionId)

    val items: StateFlow<List<MemoryEntity>> = sessionIdFlow
        .flatMapLatest { sid ->
            if (sid.isNullOrBlank()) kotlinx.coroutines.flow.flowOf(emptyList())
            else graph.memory.observeMemory(sid)
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun refreshSession() {
        sessionIdFlow.value = graph.activeSessionId
    }

    fun delete(id: String) {
        viewModelScope.launch { graph.memory.deleteMemory(id) }
    }
}

