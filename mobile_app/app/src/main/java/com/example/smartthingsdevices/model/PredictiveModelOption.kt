package com.example.smartthingsdevices.model

/**
 * Enumerates the predictive model options supported by the frontend.
 *
 * The [apiValue] must be treated as the backend contract value when this
 * selection is attached to future REST or gRPC requests.
 */
enum class PredictiveModelOption(
    val displayName: String,
    val apiValue: String,
    val description: String
) {
    CLOUD_PRIMARY(
        displayName = "Cloud Primary",
        apiValue = "cloud_primary",
        description = "Routes inference to the primary cloud model for the highest-capacity predictions."
    ),
    ON_DEVICE_OPTIMIZED(
        displayName = "On-Device Optimized",
        apiValue = "on_device_optimized",
        description = "Prefers local inference to reduce latency and preserve operation during network loss."
    ),
    HYBRID_BALANCED(
        displayName = "Hybrid Balanced",
        apiValue = "hybrid_balanced",
        description = "Uses a balanced strategy that can combine device-side and backend-assisted processing."
    )
}
