package com.roleplayai.app.viewmodel

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import com.roleplayai.app.di.AppGraph

@Composable
inline fun <reified VM : ViewModel> graphViewModel(
    graph: AppGraph,
    crossinline creator: () -> VM
): VM {
    val factory = remember(graph) {
        object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                @Suppress("UNCHECKED_CAST")
                return creator() as T
            }
        }
    }
    return viewModel(factory = factory)
}

