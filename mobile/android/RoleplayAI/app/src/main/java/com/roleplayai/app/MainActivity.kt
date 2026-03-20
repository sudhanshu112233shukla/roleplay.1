package com.roleplayai.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.roleplayai.app.ui.AppRoot
import com.roleplayai.app.ui.theme.RoleplayTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            RoleplayTheme {
                AppRoot()
            }
        }
    }
}

