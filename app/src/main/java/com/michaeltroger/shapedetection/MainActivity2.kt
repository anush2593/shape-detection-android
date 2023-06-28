package com.michaeltroger.shapedetection

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.appcompat.widget.AppCompatButton
import androidx.appcompat.widget.AppCompatTextView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class MainActivity2 : ComponentActivity() {

    companion object {

        private val PICK_IMAGE_REQUEST_CODE = 1
    }

    private lateinit var imageView: ImageView
    private lateinit var detectButton: AppCompatButton
    private lateinit var tvTotalCount: AppCompatTextView
    private var circleCount = 0
    private var triangleCount = 0
    private var rectangleCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_my2)

        // Initialize OpenCV
        OpenCVLoader.initDebug()

        imageView = findViewById(R.id.imageView)
        detectButton = findViewById(R.id.selectImage)
        tvTotalCount = findViewById(R.id.tvTotalCount)
        detectButton.setOnClickListener {
            openGallery()
        }
    }

    private fun openGallery() {
        circleCount = 0
        rectangleCount = 0
        triangleCount = 0
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, PICK_IMAGE_REQUEST_CODE)
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            val imageUri = data?.data
            imageUri?.let {
                val imageBitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageUri)

                val targetWidth: Int
                val targetHeight: Int

                val maxDimension = 1280
                if (imageBitmap.width >= imageBitmap.height) {
                    targetWidth = maxDimension
                    targetHeight = (maxDimension.toFloat() / imageBitmap.width * imageBitmap.height).toInt()
                } else {
                    targetWidth = (maxDimension.toFloat() / imageBitmap.height * imageBitmap.width).toInt()
                    targetHeight = maxDimension
                }

                val resizedBitmap = Bitmap.createScaledBitmap(imageBitmap, targetWidth, targetHeight, false)

                detectShapes(resizedBitmap)
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private fun detectShapes(imageBitmap: Bitmap) {
        // Convert Bitmap to Mat
        val rgbaMat = Mat()
        Utils.bitmapToMat(imageBitmap, rgbaMat)

        // Convert color space to HSV
        val hsvMat = Mat()
        Imgproc.cvtColor(rgbaMat, hsvMat, Imgproc.COLOR_RGB2HSV)

        /*       // 1 Define color range for light pink to red
               val lowerRed = Scalar(140.0, 50.0, 50.0)
               val upperRed = Scalar(180.0, 255.0, 255.0)

               // Threshold the image to obtain a binary mask of the red regions
               val mask = Mat()
               Core.inRange(hsvMat, lowerRed, upperRed, mask)*/

        /*    // 2 Convert the image to the LAB color space
            val labMat = Mat()
            Imgproc.cvtColor(hsvMat, labMat, Imgproc.COLOR_HSV2BGR)
            Imgproc.cvtColor(labMat, labMat, Imgproc.COLOR_BGR2Lab)

    // Define color ranges for light pink in LAB color space
            val lowerPink = Scalar(130.0, 120.0, 130.0)
            val upperPink = Scalar(180.0, 200.0, 180.0)

    // Threshold the LAB image to obtain a binary mask of the light pink regions in skin tones
            val mask = Mat()
            Core.inRange(labMat, lowerPink, upperPink, mask)*/

        // 3 Define color ranges for light pink to red in skin tones
        val lowerPink1 = Scalar(0.0, 50.0, 120.0) // Lower range for light pink
        val upperPink1 = Scalar(5.0, 255.0, 255.0) // Upper range for light pink
        val lowerPink2 = Scalar(130.0, 50.0, 50.0) // Lower range for light pink (additional range due to circular Hue values)
        val upperPink2 = Scalar(180.0, 255.0, 255.0) // Upper range for light pink (additional range due to circular Hue values)

        // Threshold the image to obtain a binary mask of the light pink regions in skin tones
        val mask = Mat()
        val mask1 = Mat()
        val mask2 = Mat()
        Core.inRange(hsvMat, lowerPink1, upperPink1, mask1)
        Core.inRange(hsvMat, lowerPink2, upperPink2, mask2)
        Core.bitwise_or(mask1, mask2, mask)

        // Find contours in the binary mask
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // Process each contour and draw shape on the original image
        for (contour in contours) {
            val epsilon = 0.06 * Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approxCurve = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approxCurve, epsilon, true)

            // Get number of vertices
            val vertices = approxCurve.total().toInt()
            val contourArea = Imgproc.contourArea(contour)

            // Calculate the ratio of contour area to the area of a perfect circle with the same perimeter
            val circleArea = (Math.PI * Math.pow(epsilon, 2.0))
            val areaRatio = contourArea / circleArea

            // Define a threshold for the area ratio to determine if it's a circle
            val circleThreshold = 3.5

            // Check if the shape is a circle based on the number of vertices and the area ratio
            val isCircle = areaRatio >= circleThreshold

            Log.d("isCircle", "areaRatio $areaRatio circleThreshold $circleThreshold")
            if (Math.abs(contourArea) < 200) {
                continue
            }

            if (vertices < 2) continue
            // Draw shape based on the number of vertices
            when (vertices) {
                3 -> {
                    drawTriangle(rgbaMat, contour)
                }
                4 -> {
                    val perimeter = 0.6 * Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
                    val numPoints = contour.rows()
                    val circularity = Math.PI * (numPoints / (perimeter * perimeter))
                    Log.d("areaRatio", areaRatio.toString())

                    Log.d("circularity", circularity.toString())
                    if (circularity >= 0.01) {
                        Log.d("isCircle", "Draw circle 4 vertices")

                        drawCircle(rgbaMat, contour)
                    } else {
                        drawRectangle(rgbaMat, contour)
                    }
                }
                in 5..20 -> {
                    drawCircle(rgbaMat, contour)
                }
            }
        }

        // Convert Mat back to Bitmap
        val resultBitmap = Bitmap.createBitmap(rgbaMat.cols(), rgbaMat.rows(), Bitmap.Config.RGB_565)
        Utils.matToBitmap(rgbaMat, resultBitmap)

        // Display the result image
        imageView.setImageBitmap(resultBitmap)
        tvTotalCount.text = "Rectangles $rectangleCount \nTriangleCount $triangleCount \nCircle $circleCount"
    }

    private fun drawCircle(rgbaMat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(rgbaMat, listOf(contour), 0, Scalar(0.0, 0.0, 255.0), 3)
        val circleLabel = "Circle"
        val labelPosition = contour.toArray().first() // Get the first point of the contour
        Imgproc.putText(
            rgbaMat,
            circleLabel,
            Point(labelPosition.x, labelPosition.y),
            Core.FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0.0, 0.0, 255.0),
            1
        )
        circleCount++
    }

    private fun drawRectangle(rgbaMat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(rgbaMat, listOf(contour), 0, Scalar(255.0, 0.0, 0.0), 3)
        val rectangleLabel = "Rectangle"
        val labelPosition = contour.toArray().first() // Get the first point of the contour
        Imgproc.putText(
            rgbaMat,
            rectangleLabel,
            Point(labelPosition.x, labelPosition.y),
            Core.FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(255.0, 0.0, 0.0),
            1
        )
        rectangleCount++
    }

    private fun drawTriangle(rgbaMat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(rgbaMat, listOf(contour), 0, Scalar(0.0, 255.0, 0.0), 3)
        val triangleLabel = "Triangle"
        val labelPosition = contour.toArray().first() // Get the first point of the contour
        Imgproc.putText(
            rgbaMat,
            triangleLabel,
            Point(labelPosition.x, labelPosition.y),
            Core.FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0.0, 255.0, 0.0),
            1
        )
        triangleCount++
    }
}
