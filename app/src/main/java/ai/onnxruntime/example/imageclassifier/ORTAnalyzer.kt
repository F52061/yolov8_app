// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.*
import kotlin.math.exp


internal data class DetectionResult(
    var boxes: MutableList<FloatArray> = mutableListOf(),  // [x1, y1, x2, y2]
    var scores: MutableList<Float> = mutableListOf(),
    var classIds: MutableList<Int> = mutableListOf(),
    var processTimeMs: Long = 0
)

internal class ORTAnalyzer(
        private val ortEnv: OrtEnvironment,
        private val ortSession: OrtSession?,
        private val callBack: (DetectionResult) -> Unit
) : ImageAnalysis.Analyzer {

    private val confidenceThreshold = 0.5f
    private val nmsThreshold = 0.45f
    private val numClasses = 80
    private val inputSize = 640  // YOLOv8默认输入尺寸

    // Rotate the image of the input bitmap
    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    // YOLOv8输出解析和非极大值抑制(NMS)
    private fun parseYoloOutput(output: Array<Array<FloatArray>>): DetectionResult {
        val result = DetectionResult()
        
        // YOLOv8输出格式: [1, 84, 8400] 其中84 = 4(框) + 80(类别)
        val predictions = output[0]
        val numPredictions = predictions[0].size
        
        val validPredictions = mutableListOf<Prediction>()
        
        // 解析所有预测
        for (i in 0 until numPredictions) {
            // 获取类别置信度最大值
            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until numClasses) {
                val score = predictions[c + 4][i]
                if (score > maxScore) {
                    maxScore = score
                    maxClassId = c
                }
            }
            
            // 应用置信度阈值
            if (maxScore >= confidenceThreshold) {
                // YOLOv8输出的是中心点坐标和宽高
                val cx = predictions[0][i]
                val cy = predictions[1][i]
                val w = predictions[2][i]
                val h = predictions[3][i]
                
                // 转换为左上角和右下角坐标
                val x1 = cx - w / 2
                val y1 = cy - h / 2
                val x2 = cx + w / 2
                val y2 = cy + h / 2
                
                validPredictions.add(Prediction(x1, y1, x2, y2, maxScore, maxClassId))
            }
        }
        
        // 非极大值抑制(NMS)
        val selectedIndices = nms(validPredictions)
        
        for (idx in selectedIndices) {
            val pred = validPredictions[idx]
            result.boxes.add(floatArrayOf(pred.x1, pred.y1, pred.x2, pred.y2))
            result.scores.add(pred.confidence)
            result.classIds.add(pred.classId)
        }
        
        return result
    }
    
    // 非极大值抑制算法
    private fun nms(predictions: List<Prediction>): List<Int> {
        // 按置信度排序
        val indices = predictions.indices.sortedByDescending { predictions[it].confidence }
        val selected = mutableListOf<Int>()
        val suppressed = BooleanArray(predictions.size)
        
        for (i in indices) {
            if (suppressed[i]) continue
            
            selected.add(i)
            
            for (j in indices) {
                if (i == j || suppressed[j]) continue
                
                val box1 = predictions[i]
                val box2 = predictions[j]
                
                // 只处理相同类别的框
                if (box1.classId != box2.classId) continue
                
                // 计算IoU
                val x1 = maxOf(box1.x1, box2.x1)
                val y1 = maxOf(box1.y1, box2.y1)
                val x2 = minOf(box1.x2, box2.x2)
                val y2 = minOf(box1.y2, box2.y2)
                
                val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
                val area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
                val area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
                val union = area1 + area2 - intersection
                
                val iou = if (union > 0) intersection / union else 0f
                
                if (iou > nmsThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        return selected
    }
    
    data class Prediction(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val confidence: Float, val classId: Int
    )



    override fun analyze(image: ImageProxy) {
        val session = ortSession ?: run {
            image.close()
            return
        }
        
        try {
            // 将输入图像转换为bitmap并resize到640x640用于YOLOv8模型输入
            val imgBitmap = image.toBitmap()
            val rawBitmap = Bitmap.createScaledBitmap(imgBitmap, inputSize, inputSize, false)
            val bitmap = rawBitmap.rotate(image.imageInfo.rotationDegrees.toFloat())

            if (bitmap != null) {
                var result = DetectionResult()

                val imgData = preProcess(bitmap)
                val inputName = session.inputNames.iterator().next()
                val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
                
                val tensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
                val startTime = SystemClock.uptimeMillis()
                
                tensor.use {
                    val output = session.run(Collections.singletonMap(inputName, tensor))
                    output.use {
                        result.processTimeMs = SystemClock.uptimeMillis() - startTime
                        @Suppress("UNCHECKED_CAST")
                        val rawOutput = output.get(0).value as Array<Array<FloatArray>>
                        result = parseYoloOutput(rawOutput)
                    }
                }
                callBack(result)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            image.close()
        }
    }

    // We can switch analyzer in the app, need to make sure the native resources are freed
    protected fun finalize() {
        ortSession?.close()
    }
}