package com.roleplayai.app.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.roleplayai.app.di.AppGraph
import com.roleplayai.app.ui.screens.ChatScreen
import com.roleplayai.app.ui.screens.CharacterScreen
import com.roleplayai.app.ui.screens.MemoryScreen
import com.roleplayai.app.ui.screens.SettingsScreen
import com.roleplayai.app.ui.screens.SessionsScreen

object Routes {
    const val Sessions = "sessions"
    const val Chat = "chat"
    const val Characters = "characters"
    const val Memory = "memory"
    const val Settings = "settings"
}

@Composable
fun AppRoot() {
    val nav = rememberNavController()
    val context = LocalContext.current
    val graph = remember { AppGraph(context) }
    Surface(color = MaterialTheme.colorScheme.background) {
        Box(modifier = Modifier.fillMaxSize()) {
            NavHost(navController = nav, startDestination = Routes.Sessions) {
                composable(Routes.Sessions) {
                    SessionsScreen(
                        graph = graph,
                        onOpenChat = { nav.navigate(Routes.Chat) },
                        onOpenSettings = { nav.navigate(Routes.Settings) }
                    )
                }
                composable(Routes.Chat) {
                    ChatScreen(
                        graph = graph,
                        onOpenCharacters = { nav.navigate(Routes.Characters) },
                        onOpenMemory = { nav.navigate(Routes.Memory) },
                        onBack = { nav.popBackStack() }
                    )
                }
                composable(Routes.Characters) { CharacterScreen(graph = graph, onBack = { nav.popBackStack() }) }
                composable(Routes.Memory) { MemoryScreen(graph = graph, onBack = { nav.popBackStack() }) }
                composable(Routes.Settings) { SettingsScreen(onBack = { nav.popBackStack() }) }
            }
        }
    }
}
