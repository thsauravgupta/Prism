package com.example.smartthingsdevices.data

import com.example.smartthingsdevices.model.Device
import com.example.smartthingsdevices.network.SmartThingsApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class DeviceRepository(
    private val apiService: SmartThingsApiService
) {

    suspend fun fetchDevices(patToken: String): List<Device> = withContext(Dispatchers.IO) {
        apiService.getDevices(token = "Bearer $patToken").items
    }
}
