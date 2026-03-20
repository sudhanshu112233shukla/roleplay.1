package com.roleplayai.app.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.viewmodel.graphViewModel
import com.roleplayai.app.viewmodel.SessionsViewModel

@Composable
fun SessionsScreen(
    graph: AppGraph,
    onOpenChat: () -> Unit,
    onOpenSettings: () -> Unit
) {
    val vm: SessionsViewModel = graphViewModel(graph) { SessionsViewModel(graph) }
    val sessions by vm.sessions.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text("Sessions") },
            actions = {
                IconButton(onClick = onOpenSettings) { Text("Settings") }
            }
        )

        Row(modifier = Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { vm.createAndOpenSession(onOpenChat) }) {
                Text("New Chat")
            }
            Button(onClick = onOpenChat, enabled = graph.activeSessionId != null) {
                Text("Resume")
            }
        }

        LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
            items(sessions, key = { it.id }) { s ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { vm.openSession(s.id, onOpenChat) }
                        .padding(vertical = 12.dp)
                ) {
                    Text(s.title, style = MaterialTheme.typography.titleMedium)
                    Text("Character: ${s.characterId}", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}
