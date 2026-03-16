package com.example.smartthingsdevices

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.smartthingsdevices.data.DeviceRepository
import com.example.smartthingsdevices.network.RetrofitClient
import com.example.smartthingsdevices.ui.DeviceListScreen
import com.example.smartthingsdevices.viewmodel.DeviceViewModel
import com.example.smartthingsdevices.viewmodel.DeviceViewModelFactory

class MainActivity : ComponentActivity() {

    private val viewModel: DeviceViewModel by viewModels {
        DeviceViewModelFactory(
            repository = DeviceRepository(RetrofitClient.apiService)
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    DeviceListScreen(viewModel = viewModel)
                }
            }
        }
    }
}
