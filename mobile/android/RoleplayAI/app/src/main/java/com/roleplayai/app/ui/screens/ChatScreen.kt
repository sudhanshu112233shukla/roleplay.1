package com.roleplayai.app.ui.screens

import android.content.Intent
import android.provider.OpenableColumns
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.viewmodel.graphViewModel
import com.roleplayai.app.viewmodel.ChatViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    graph: AppGraph,
    onOpenCharacters: () -> Unit,
    onOpenMemory: () -> Unit,
    onBack: () -> Unit
) {
    val vm: ChatViewModel = graphViewModel(graph) { ChatViewModel(graph) }

    LaunchedEffect(Unit) { vm.refreshSession() }

    val messages by vm.messages.collectAsState()
    val state by vm.state.collectAsState()
    var input by remember { mutableStateOf("") }
    val context = LocalContext.current
    val tts = remember {
        TextToSpeech(context.applicationContext) { }
    }
    DisposableEffect(Unit) {
        onDispose { tts.shutdown() }
    }

    val voiceLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
        val data = res.data
        val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
        val text = results?.firstOrNull()?.trim().orEmpty()
        if (text.isNotEmpty()) {
            input = if (input.isBlank()) text else (input.trimEnd() + "\n" + text)
        }
    }

    val attachmentLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val name = runCatching {
            context.contentResolver.query(uri, null, null, null, null)?.use { c ->
                val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (c.moveToFirst() && idx >= 0) c.getString(idx) else "file"
            }
        }.getOrNull() ?: "file"

        val text = runCatching {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                val bytes = stream.readBytes()
                // Safety cap: avoid huge attachments in prompt.
                val capped = if (bytes.size > 64_000) bytes.copyOfRange(0, 64_000) else bytes
                String(capped, Charsets.UTF_8)
            }
        }.getOrNull().orEmpty()

        val block = buildString {
            appendLine()
            appendLine("[Attachment: $name]")
            appendLine(text.trim())
        }.trim()
        input = if (input.isBlank()) block else (input.trimEnd() + "\n\n" + block)
    }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = {
                Column {
                    Text("Chat", style = MaterialTheme.typography.titleMedium)
                    Text(state.engineName, style = MaterialTheme.typography.bodySmall)
                }
            },
            navigationIcon = { Button(onClick = onBack) { Text("Back") } },
            actions = {
                Button(onClick = onOpenCharacters) { Text("Character") }
                Spacer(modifier = Modifier.padding(4.dp))
                Button(onClick = onOpenMemory) { Text("Memory") }
                Spacer(modifier = Modifier.padding(4.dp))
                Button(
                    onClick = {
                        val last = messages.lastOrNull { it.role == "assistant" }?.content?.trim().orEmpty()
                        if (last.isNotEmpty()) {
                            tts.speak(last, TextToSpeech.QUEUE_FLUSH, null, "roleplay_tts")
                        }
                    }
                ) { Text("Speak") }
            }
        )

        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            items(messages, key = { it.id }) { m ->
                val alignRight = m.role == "user"
                MessageBubble(text = m.content, isUser = alignRight)
            }
            if (state.isStreaming && state.streamingText.isNotBlank()) {
                item(key = "streaming") {
                    MessageBubble(text = state.streamingText, isUser = false, isStreaming = true)
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message") }
            )
            Button(
                onClick = {
                    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
                    }
                    voiceLauncher.launch(intent)
                }
            ) { Text("Voice") }

            Button(onClick = { attachmentLauncher.launch(arrayOf("*/*")) }) { Text("Attach") }
            Button(
                onClick = {
                    val t = input.trim()
                    if (t.isNotEmpty()) {
                        input = ""
                        vm.send(t)
                    }
                }
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
private fun MessageBubble(text: String, isUser: Boolean, isStreaming: Boolean = false) {
    val bg = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val fg = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) androidx.compose.ui.Alignment.End else androidx.compose.ui.Alignment.Start
    ) {
        androidx.compose.material3.Surface(color = bg, shape = MaterialTheme.shapes.medium) {
            Text(
                text = if (isStreaming) "$text|" else text,
                color = fg,
                modifier = Modifier.padding(12.dp)
            )
        }
    }
}
