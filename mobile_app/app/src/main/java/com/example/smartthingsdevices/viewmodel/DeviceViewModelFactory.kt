package com.example.smartthingsdevices.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.smartthingsdevices.data.DeviceRepository

class DeviceViewModelFactory(
    private val repository: DeviceRepository
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(DeviceViewModel::class.java)) {
            "Unknown ViewModel class: ${modelClass.name}"
        }
        return DeviceViewModel(repository) as T
    }
}
