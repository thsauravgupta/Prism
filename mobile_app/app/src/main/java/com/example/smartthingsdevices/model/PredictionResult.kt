package com.example.smartthingsdevices.model

/**
 * Holds the output of one on-device model inference run.
 */
data class PredictionResult(
    val modelName: String,
    val predictions: List<DevicePrediction>,
    val inferenceTimeMs: Long
)

/**
 * A single predicted device with its confidence score.
 */
data class DevicePrediction(
    val deviceIndex: Int,
    val deviceName: String,
    val confidence: Float
)
