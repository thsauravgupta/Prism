package com.example.smartthingsdevices.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartthingsdevices.data.DeviceRepository
import com.example.smartthingsdevices.inference.GnnInferenceEngine
import com.example.smartthingsdevices.inference.InferenceEngineFactory
import com.example.smartthingsdevices.inference.TwoLevelInferenceEngine
import com.example.smartthingsdevices.model.PredictionResult
import com.example.smartthingsdevices.model.PredictiveModelOption
import com.example.smartthingsdevices.ui.DeviceScreenState
import com.example.smartthingsdevices.ui.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import retrofit2.HttpException
import java.io.IOException

class DeviceViewModel(
    application: Application,
    private val repository: DeviceRepository
) : AndroidViewModel(application) {

    private val _screenState = MutableStateFlow(DeviceScreenState())
    val screenState: StateFlow<DeviceScreenState> = _screenState.asStateFlow()

    fun selectModel(model: PredictiveModelOption) {
        _screenState.update { it.copy(selectedModel = model) }
    }

    fun fetchDevices(patToken: String) {
        val trimmedToken = patToken.trim()
        if (trimmedToken.isBlank()) {
            _screenState.update {
                it.copy(deviceUiState = UiState.Error("Personal Access Token is required."))
            }
            return
        }

        viewModelScope.launch {
            _screenState.update { it.copy(deviceUiState = UiState.Loading) }

            val nextUiState = try {
                UiState.Success(repository.fetchDevices(trimmedToken))
            } catch (exception: HttpException) {
                UiState.Error(
                    message = "Request failed: ${exception.code()} ${exception.message().orEmpty()}".trim()
                )
            } catch (_: IOException) {
                UiState.Error("Network error. Check your connection and try again.")
            } catch (exception: Exception) {
                UiState.Error(exception.message ?: "Something went wrong.")
            }

            _screenState.update { it.copy(deviceUiState = nextUiState) }
        }
    }

    /** Run on-device prediction using the currently selected model. */
    fun runPrediction() {
        val currentState = _screenState.value
        if (!currentState.canRunPrediction) return

        val devices = (currentState.deviceUiState as UiState.Success).data
        val selectedModel = currentState.selectedModel
        val appContext = getApplication<Application>().applicationContext

        viewModelScope.launch {
            _screenState.update { it.copy(predictionState = UiState.Loading) }

            val result = withContext(Dispatchers.Default) {
                try {
                    val demoSequence = InferenceEngineFactory.generateDemoSequence(
                        option = selectedModel,
                        numDevicesFetched = devices.size
                    )

                    val prediction: PredictionResult = when (selectedModel) {
                        PredictiveModelOption.TWO_LEVEL_ARCH -> {
                            val engine = InferenceEngineFactory.getEngine(
                                selectedModel, appContext
                            ) as TwoLevelInferenceEngine
                            engine.predict(demoSequence)
                        }
                        PredictiveModelOption.KG_GNN -> {
                            val engine = InferenceEngineFactory.getEngine(
                                selectedModel, appContext
                            ) as GnnInferenceEngine
                            engine.predict(demoSequence)
                        }
                    }
                    UiState.Success(prediction)
                } catch (e: Exception) {
                    UiState.Error(e.message ?: "Prediction failed.")
                }
            }

            _screenState.update { it.copy(predictionState = result) }
        }
    }
}
