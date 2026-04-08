package com.example.smartthingsdevices.inference

import android.content.Context
import com.example.smartthingsdevices.model.DevicePrediction
import com.example.smartthingsdevices.model.DeviceTypes
import com.example.smartthingsdevices.model.PredictionResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs inference using the KG-GNN TFLite model.
 *
 * The model expects 5 separate input tensors (matching gnn_model.py):
 *   - dayofweek  [1, 9]  int32
 *   - hour       [1, 9]  int32
 *   - device     [1, 9]  int32
 *   - unknown    [1, 9]  float32
 *   - device_control [1, 9] int32
 *
 * Output: softmax probabilities over all device types [1, vocab_dev].
 */
class GnnInferenceEngine(private val context: Context) {

    private var interpreter: Interpreter? = null

    companion object {
        private const val MODEL_ASSET = "gnn_model.tflite"
        private const val SEQ_LEN = 9
        private const val INPUT_SIZE = 5
        private const val TOP_K = 5
    }

    /** Lazily load the TFLite interpreter. */
    private fun ensureLoaded() {
        if (interpreter == null) {
            val model = loadModelFile()
            interpreter = Interpreter(model)
        }
    }

    /**
     * Run prediction on a simulated action-history.
     *
     * @param actionHistory  flattened float array of shape [SEQ_LEN * INPUT_SIZE].
     *                       Each row is (day_of_week, hour, device, control, device_control).
     *                       Values are raw integer indices (NOT normalised).
     */
    fun predict(actionHistory: FloatArray): PredictionResult {
        ensureLoaded()
        val interp = interpreter!!

        val startTime = System.nanoTime()

        // ── 1. Decompose flat array into 5 separate input buffers ────────
        val dayOfWeek = IntArray(SEQ_LEN)
        val hour = IntArray(SEQ_LEN)
        val device = IntArray(SEQ_LEN)
        val unknown = FloatArray(SEQ_LEN)
        val deviceControl = IntArray(SEQ_LEN)

        for (t in 0 until SEQ_LEN) {
            val base = t * INPUT_SIZE
            dayOfWeek[t] = actionHistory[base + 0].toInt()
            hour[t] = actionHistory[base + 1].toInt()
            device[t] = actionHistory[base + 2].toInt()
            unknown[t] = actionHistory[base + 3]
            deviceControl[t] = actionHistory[base + 4].toInt()
        }

        // Wrap into [1, SEQ_LEN] arrays as the model expects batch dim
        val inputDow = Array(1) { dayOfWeek }
        val inputHour = Array(1) { hour }
        val inputDev = Array(1) { device }
        val inputUnk = Array(1) { unknown }
        val inputDC = Array(1) { deviceControl }

        // ── 2. Determine output shape from the interpreter ───────────────
        val outputTensorIndex = 0
        val outputShape = interp.getOutputTensor(outputTensorIndex).shape()
        val numClasses = outputShape[1]  // vocab_dev

        val outputBuffer = Array(1) { FloatArray(numClasses) }

        // ── 3. Build ordered input map matching signature order ───────────
        //    gnn_model.py: inputs=[in_dow, in_hr, in_dev, in_unknown, in_ctrl]
        //    TFLite signature preserves this order.
        val inputs: Array<Any> = arrayOf(inputDow, inputHour, inputDev, inputUnk, inputDC)
        val outputs = mutableMapOf<Int, Any>(0 to outputBuffer)

        interp.runForMultipleInputsOutputs(inputs, outputs)

        val probs = outputBuffer[0]

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        // ── 4. Extract top-K ─────────────────────────────────────────────
        val indexed = probs.mapIndexed { idx, score -> idx to score }
            .sortedByDescending { it.second }
            .take(TOP_K)

        val predictions = indexed.map { (idx, score) ->
            DevicePrediction(
                deviceIndex = idx,
                deviceName = DeviceTypes.nameFor(idx),
                confidence = score
            )
        }

        return PredictionResult(
            modelName = "Knowledge Graph GNN",
            predictions = predictions,
            inferenceTimeMs = elapsedMs
        )
    }

    /** Memory-map the TFLite model from assets. */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFd = context.assets.openFd(MODEL_ASSET)
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }
}
