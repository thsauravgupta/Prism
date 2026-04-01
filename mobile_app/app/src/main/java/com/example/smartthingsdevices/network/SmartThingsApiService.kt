package com.example.smartthingsdevices.network

import com.example.smartthingsdevices.model.DeviceResponse
import retrofit2.http.GET
import retrofit2.http.Header

interface SmartThingsApiService {

    @GET("v1/devices")
    suspend fun getDevices(
        @Header("Authorization") token: String
    ): DeviceResponse
}
