package com.example.smartthingsdevices.ui

import com.example.smartthingsdevices.model.Device
import com.example.smartthingsdevices.model.PredictiveModelOption

data class DeviceScreenState(
    val deviceUiState: UiState<List<Device>> = UiState.Success(emptyList()),
    val availableModels: List<PredictiveModelOption> = PredictiveModelOption.entries,
    val selectedModel: PredictiveModelOption = PredictiveModelOption.ON_DEVICE_OPTIMIZED
) {
    val selectedModelApiValue: String
        get() = selectedModel.apiValue
}
