package com.example.smartthingsdevices.ui

import com.example.smartthingsdevices.model.Device
import com.example.smartthingsdevices.model.PredictionResult
import com.example.smartthingsdevices.model.PredictiveModelOption

data class DeviceScreenState(
    val deviceUiState: UiState<List<Device>> = UiState.Success(emptyList()),
    val predictionState: UiState<PredictionResult> = UiState.Success(
        PredictionResult(modelName = "", predictions = emptyList(), inferenceTimeMs = 0)
    ),
    val availableModels: List<PredictiveModelOption> = PredictiveModelOption.entries,
    val selectedModel: PredictiveModelOption = PredictiveModelOption.TWO_LEVEL_ARCH
) {
    val selectedModelApiValue: String
        get() = selectedModel.apiValue

    /** True when devices have been fetched successfully and list is non-empty. */
    val canRunPrediction: Boolean
        get() = deviceUiState is UiState.Success && (deviceUiState as UiState.Success).data.isNotEmpty()
}
