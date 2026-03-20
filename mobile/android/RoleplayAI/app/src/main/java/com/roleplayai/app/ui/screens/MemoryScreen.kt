package com.roleplayai.app.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.viewmodel.graphViewModel
import com.roleplayai.app.viewmodel.MemoryViewModel

@Composable
fun MemoryScreen(graph: AppGraph, onBack: () -> Unit) {
    val vm: MemoryViewModel = graphViewModel(graph) { MemoryViewModel(graph) }
    LaunchedEffect(Unit) { vm.refreshSession() }
    val items by vm.items.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text("Memory") },
            navigationIcon = { Button(onClick = onBack) { Text("Back") } }
        )

        LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            items(items, key = { it.id }) { m ->
                Column(modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
                    Text(m.text, style = MaterialTheme.typography.bodyMedium)
                    Text("score=${m.score}", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = { vm.delete(m.id) }) { Text("Delete") }
                }
            }
        }
    }
}
