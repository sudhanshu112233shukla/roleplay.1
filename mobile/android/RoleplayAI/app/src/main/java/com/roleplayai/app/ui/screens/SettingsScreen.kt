package com.roleplayai.app.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun SettingsScreen(onBack: () -> Unit) {
    var temperature by remember { mutableFloatStateOf(0.8f) }
    var maxTokens by remember { mutableFloatStateOf(180f) }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text("Settings") },
            navigationIcon = { Button(onClick = onBack) { Text("Back") } }
        )
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Offline mode: ON")
            Text("Temperature: ${"%.2f".format(temperature)}")
            Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0f..1.5f)
            Text("Max tokens: ${maxTokens.toInt()}")
            Slider(value = maxTokens, onValueChange = { maxTokens = it }, valueRange = 32f..512f)
            Text("Model selection and engine selection are wired in repository layer next.")
        }
    }
}

