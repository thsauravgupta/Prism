package com.example.smartthingsdevices.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartthingsdevices.data.DeviceRepository
import com.example.smartthingsdevices.model.PredictiveModelOption
import com.example.smartthingsdevices.ui.DeviceScreenState
import com.example.smartthingsdevices.ui.UiState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.IOException

class DeviceViewModel(
    private val repository: DeviceRepository
) : ViewModel() {

    private val _screenState = MutableStateFlow(DeviceScreenState())
    val screenState: StateFlow<DeviceScreenState> = _screenState.asStateFlow()

    fun selectModel(model: PredictiveModelOption) {
        _screenState.update { currentState ->
            currentState.copy(selectedModel = model)
        }
    }

    fun fetchDevices(patToken: String) {
        val trimmedToken = patToken.trim()
        if (trimmedToken.isBlank()) {
            _screenState.update { currentState ->
                currentState.copy(
                    deviceUiState = UiState.Error("Personal Access Token is required.")
                )
            }
            return
        }

        viewModelScope.launch {
            _screenState.update { currentState ->
                currentState.copy(deviceUiState = UiState.Loading)
            }

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

            _screenState.update { currentState ->
                currentState.copy(deviceUiState = nextUiState)
            }
        }
    }
}
