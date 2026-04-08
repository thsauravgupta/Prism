package com.example.smartthingsdevices.inference

import android.content.Context
import com.example.smartthingsdevices.model.DevicePrediction
import com.example.smartthingsdevices.model.DeviceTypes
import com.example.smartthingsdevices.model.PredictionResult
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

/**
 * Runs inference using the Two-Level Architecture:
 *   LSTM routine predictor  →  heuristic scores  →  hybrid fuser  →  top-K devices
 *
 * Loads `lstm_quantized.ptl` and `hybrid_fuser.ptl` from Android assets.
 * The XGBoost re-ranker is omitted on-device (no native Android runtime);
 * the fuser therefore blends LSTM + heuristic with adjusted weights.
 */
class TwoLevelInferenceEngine(private val context: Context) {

    private var lstmModule: Module? = null
    private var fuserModule: Module? = null

    companion object {
        private const val LSTM_ASSET = "lstm_quantized.ptl"
        private const val FUSER_ASSET = "hybrid_fuser.ptl"
        private const val SEQ_LEN = 9
        private const val INPUT_SIZE = 5
        private const val NUM_DEVICES = DeviceTypes.NUM_DEVICE_TYPES
        private const val TOP_K = 5
    }

    /** Lazily load both ptl modules. */
    private fun ensureLoaded() {
        if (lstmModule == null) {
            lstmModule = LiteModuleLoader.load(assetFilePath(LSTM_ASSET))
        }
        if (fuserModule == null) {
            fuserModule = LiteModuleLoader.load(assetFilePath(FUSER_ASSET))
        }
    }

    /**
     * Run prediction on a simulated action history.
     *
     * @param actionHistory  flattened float array of shape [SEQ_LEN * INPUT_SIZE].
     *                       Each row is (day_of_week, hour, device, control, device_control),
     *                       pre-normalised to [0, 1].
     * @return top-K device predictions with confidence and timing.
     */
    fun predict(actionHistory: FloatArray): PredictionResult {
        ensureLoaded()

        val startTime = System.nanoTime()

        // ── 1. LSTM inference ─────────────────────────────────────────────
        val inputTensor = Tensor.fromBlob(
            actionHistory,
            longArrayOf(1, SEQ_LEN.toLong(), INPUT_SIZE.toLong())
        )
        val lstmOutput = lstmModule!!.forward(IValue.from(inputTensor)).toTensor()
        val lstmScores = lstmOutput.dataAsFloatArray          // [NUM_DEVICES]

        // Apply softmax to get probabilities
        val lstmProbs = softmax(lstmScores)

        // ── 2. Heuristic scores (recency + frequency from the sequence) ──
        val heuristicScores = computeHeuristicScores(actionHistory)

        // ── 3. Fuse scores ────────────────────────────────────────────────
        // The fuser expects 3 tensors: (heuristic, xgboost, lstm) each [1, NUM_DEVICES]
        // Since we don't have XGBoost on-device, pass zeros for the xgb channel.
        val hTensor = Tensor.fromBlob(heuristicScores, longArrayOf(1, NUM_DEVICES.toLong()))
        val xTensor = Tensor.fromBlob(FloatArray(NUM_DEVICES), longArrayOf(1, NUM_DEVICES.toLong()))
        val lTensor = Tensor.fromBlob(lstmProbs, longArrayOf(1, NUM_DEVICES.toLong()))

        val fuserOutput = fuserModule!!.forward(
            IValue.from(hTensor),
            IValue.from(xTensor),
            IValue.from(lTensor)
        ).toTensor()

        val topIndices = fuserOutput.dataAsLongArray   // [TOP_K]

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        // Build result with confidence from LSTM probabilities
        val predictions = topIndices.map { idx ->
            val i = idx.toInt()
            DevicePrediction(
                deviceIndex = i,
                deviceName = DeviceTypes.nameFor(i),
                confidence = lstmProbs.getOrElse(i) { 0f }
            )
        }

        return PredictionResult(
            modelName = "Two-Level Architecture",
            predictions = predictions,
            inferenceTimeMs = elapsedMs
        )
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max() ?: 0f
        val exps = FloatArray(logits.size) { kotlin.math.exp(logits[it] - max) }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }

    /**
     * Simple recency + frequency heuristic that mirrors
     * `Two_level_Arch/src/feature_engineering.py`.
     */
    private fun computeHeuristicScores(flatHistory: FloatArray): FloatArray {
        val scores = FloatArray(NUM_DEVICES)
        for (t in 0 until SEQ_LEN) {
            val base = t * INPUT_SIZE
            val deviceIdx = (flatHistory[base + 2] * (NUM_DEVICES - 1)).toInt()
                .coerceIn(0, NUM_DEVICES - 1)
            // Recency: later steps count more
            val recency = (t + 1).toFloat() / SEQ_LEN
            scores[deviceIdx] += recency
            // Frequency: each occurrence adds 1/SEQ_LEN
            scores[deviceIdx] += 1f / SEQ_LEN
        }
        // Normalise to [0, 1]
        val maxScore = scores.max() ?: 1f
        if (maxScore > 0f) {
            for (i in scores.indices) scores[i] /= maxScore
        }
        return scores
    }

    /** Copy an asset to internal storage and return its absolute path. */
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}
