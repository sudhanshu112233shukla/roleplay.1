package com.roleplayai.app.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.viewmodel.graphViewModel
import com.roleplayai.app.viewmodel.CharacterViewModel

@Composable
fun CharacterScreen(graph: AppGraph, onBack: () -> Unit) {
    val vm: CharacterViewModel = graphViewModel(graph) { CharacterViewModel(graph) }
    val characters by vm.characters.collectAsState()

    var id by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var personality by remember { mutableStateOf("") }
    var world by remember { mutableStateOf("") }
    var tone by remember { mutableStateOf("") }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text("Characters") },
            navigationIcon = { Button(onClick = onBack) { Text("Back") } }
        )

        LazyColumn(modifier = Modifier.weight(1f).padding(16.dp)) {
            items(characters, key = { it.id }) { c ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { vm.select(c.id) }
                        .padding(vertical = 12.dp)
                ) {
                    Text(c.name, style = MaterialTheme.typography.titleMedium)
                    Text("World: ${c.world}", style = MaterialTheme.typography.bodySmall)
                    if (c.id == vm.activeCharacterId) {
                        Text("Selected", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }

        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            Text("Create Character", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(value = id, onValueChange = { id = it }, label = { Text("Character id (optional)") })
            OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") })
            OutlinedTextField(value = personality, onValueChange = { personality = it }, label = { Text("Personality") })
            OutlinedTextField(value = world, onValueChange = { world = it }, label = { Text("World") })
            OutlinedTextField(value = tone, onValueChange = { tone = it }, label = { Text("Tone") })
            Button(
                onClick = {
                    if (name.trim().isNotEmpty()) {
                        vm.create(id.trim(), name.trim(), personality.trim(), world.trim(), tone.trim())
                        id = ""; name = ""; personality = ""; world = ""; tone = ""
                    }
                },
                modifier = Modifier.padding(top = 8.dp)
            ) { Text("Save") }
        }
    }
}
