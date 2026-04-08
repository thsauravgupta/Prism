package com.example.smartthingsdevices.inference

import android.content.Context
import com.example.smartthingsdevices.model.PredictiveModelOption

/**
 * Factory that returns the correct inference engine for the chosen model.
 */
object InferenceEngineFactory {

    private var twoLevelEngine: TwoLevelInferenceEngine? = null
    private var gnnEngine: GnnInferenceEngine? = null

    fun getEngine(
        option: PredictiveModelOption,
        context: Context
    ): Any = when (option) {
        PredictiveModelOption.TWO_LEVEL_ARCH -> {
            if (twoLevelEngine == null) twoLevelEngine = TwoLevelInferenceEngine(context)
            twoLevelEngine!!
        }
        PredictiveModelOption.KG_GNN -> {
            if (gnnEngine == null) gnnEngine = GnnInferenceEngine(context)
            gnnEngine!!
        }
    }

    /**
     * Generates a simulated action-history sequence for demo purposes.
     * Returns a flat FloatArray of size [SEQ_LEN * 5].
     *
     * For the Two-Level Arch the values are normalised to [0, 1].
     * For the KG-GNN model the values are raw integer indices.
     */
    fun generateDemoSequence(
        option: PredictiveModelOption,
        numDevicesFetched: Int
    ): FloatArray {
        val seqLen = 9
        val inputSize = 5
        val result = FloatArray(seqLen * inputSize)
        val random = java.util.Random(42)

        // Simulate a plausible daily routine:
        // morning activities across a few devices
        val numDevices = numDevicesFetched.coerceAtLeast(5).coerceAtMost(38)

        for (t in 0 until seqLen) {
            val base = t * inputSize
            val dayOfWeek = random.nextInt(7)
            val hour = (6 + t) % 8                       // morning window 6-14 mapped to bins 0-7
            val deviceId = random.nextInt(numDevices)
            val control = random.nextInt(10)
            val deviceControl = random.nextInt(20)

            when (option) {
                PredictiveModelOption.TWO_LEVEL_ARCH -> {
                    // Normalised to [0, 1] — matching feature_engineering.py
                    result[base + 0] = dayOfWeek / 6f
                    result[base + 1] = hour / 7f
                    result[base + 2] = deviceId.toFloat() / (numDevices - 1).coerceAtLeast(1)
                    result[base + 3] = control.toFloat() / 9f
                    result[base + 4] = deviceControl.toFloat() / 19f
                }
                PredictiveModelOption.KG_GNN -> {
                    // Raw integer indices — TFLite model expects int32
                    result[base + 0] = dayOfWeek.toFloat()
                    result[base + 1] = hour.toFloat()
                    result[base + 2] = deviceId.toFloat()
                    result[base + 3] = control.toFloat()
                    result[base + 4] = deviceControl.toFloat()
                }
            }
        }
        return result
    }
}
