// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import java.lang.Runnable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val labelData: List<String> by lazy { readLabels() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        ortEnv = OrtEnvironment.getEnvironment()
        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .build()
                    .also {
                        it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                    }

            imageCapture = ImageCapture.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            setORTAnalyzer()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    override fun onRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<out String>,
            grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                        this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT
                ).show()
                finish()
            }

        }
    }

    private fun updateUI(result: DetectionResult) {
        runOnUiThread {
            if (result.boxes.isEmpty()) {
                // 没有检测到目标
                binding.detectedItem1.text = "未检测到目标"
                binding.detectedItemValue1.text = ""
                binding.detectedItem2.text = ""
                binding.detectedItemValue2.text = ""
                binding.detectedItem3.text = ""
                binding.detectedItemValue3.text = ""
                binding.inferenceTimeValue.text = result.processTimeMs.toString() + "ms"
                binding.percentMeter.progress = 0
                return@runOnUiThread
            }
            
            // 显示检测到的前3个目标
            val maxDetections = minOf(3, result.boxes.size)
            binding.percentMeter.progress = 100
            
            for (i in 0 until maxDetections) {
                val className = labelData[result.classIds[i]]
                val confidence = result.scores[i] * 100
                val box = result.boxes[i]
                
                when (i) {
                    0 -> {
                        binding.detectedItem1.text = "$className (${confidence.toInt()}%)"
                        binding.detectedItemValue1.text = ""
                    }
                    1 -> {
                        binding.detectedItem2.text = "$className (${confidence.toInt()}%)"
                        binding.detectedItemValue2.text = ""
                    }
                    2 -> {
                        binding.detectedItem3.text = "$className (${confidence.toInt()}%)"
                        binding.detectedItemValue3.text = ""
                    }
                }
            }
            
            // 清除未使用的显示项
            for (i in maxDetections until 3) {
                when (i) {
                    1 -> {
                        binding.detectedItem2.text = ""
                        binding.detectedItemValue2.text = ""
                    }
                    2 -> {
                        binding.detectedItem3.text = ""
                        binding.detectedItemValue3.text = ""
                    }
                }
            }
            
            binding.inferenceTimeValue.text = "${result.processTimeMs}ms (检测到 ${result.boxes.size} 个目标)"
        }
    }

    // Read COCO class labels
    private fun readLabels(): List<String> {
        val lines = resources.openRawResource(R.raw.txt_coco80).bufferedReader().readLines()
        // 解析格式为 "0: person" 的标签文件，只保留类别名称
        return lines.map { line ->
            val parts = line.split(":", limit = 2)
            if (parts.size == 2) parts[1].trim() else line.trim()
        }
    }

    // Read ort model into a ByteArray, run in background
    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        // YOLOv8模型
        resources.openRawResource(R.raw.yolov8).readBytes()
    }

    // Create a new ORT session in background
    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        ortEnv?.createSession(readModel())
    }

    // Create a new ORT session and then change the ImageAnalysis.Analyzer
    // This part is done in background to avoid blocking the UI
    private fun setORTAnalyzer(){
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            val session = createOrtSession()
            if (session != null && ortEnv != null) {
                imageAnalysis?.setAnalyzer(
                        backgroundExecutor,
                        ORTAnalyzer(ortEnv!!, session, ::updateUI)
                )
            }
        }
    }

    companion object {
        public const val TAG = "ORTImageClassifier"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
