package com.example.smartthingsdevices.model

/**
 * Enumerates the predictive model options available for on-device inference.
 */
enum class PredictiveModelOption(
    val displayName: String,
    val apiValue: String,
    val description: String
) {
    TWO_LEVEL_ARCH(
        displayName = "Two-Level Architecture",
        apiValue = "two_level_arch",
        description = "LSTM routine predictor fused with heuristic scores for next-device prediction."
    ),
    KG_GNN(
        displayName = "Knowledge Graph GNN",
        apiValue = "kg_gnn",
        description = "Graph neural network using device co-occurrence knowledge graph for prediction."
    )
}
