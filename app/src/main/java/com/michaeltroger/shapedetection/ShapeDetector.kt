package com.michaeltroger.shapedetection

import android.graphics.Bitmap
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters

class ShapeDetector {

    private var annotatedImage: Bitmap? = null

    init {
        OpenCVLoader.initDebug()
    }

    fun detectShapes(inputFrame: CvCameraViewFrame): Mat {
        // get the camera frame as RGB image
        val rgb = inputFrame.rgba()

        // convert the image to HSV color space
        val hsv = Mat()
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        // COLOR_RGBA2HSV
        // define the lower and upper bounds for pink to red color range
        val lowerPink = Scalar(150.0, 50.0, 50.0)
        val upperRed = Scalar(180.0, 255.0, 255.0)

        // create a binary mask for pixels within the specified color range
        val mask = Mat()
        Core.inRange(hsv, lowerPink, upperRed, mask)

        // apply morphological operations to enhance the mask
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        val morphedMask = Mat()
        Imgproc.morphologyEx(mask, morphedMask, Imgproc.MORPH_OPEN, kernel)

        // find contours and store them all as a list
        val contours: MutableList<MatOfPoint> = ArrayList()
        val hierarchy = Mat()
        Imgproc.findContours(morphedMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // the image to output on the screen in the end
        val dst = rgb.clone()

        // loop over all found contours
        for (contour in contours) {
            // approximate the contour to a polygon
            val approxCurve = MatOfPoint2f()
            val contourPerimeter = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approxCurve, 0.02 * contourPerimeter, true)

            // get the number of vertices of the polygon
            val numVertices = approxCurve.total().toInt()
            val contourArea = Imgproc.contourArea(contour)

            // ignore small contours and non-convex shapes
            if (Math.abs(contourArea) < 150) {
                continue
            }

            // triangle detection
            if (numVertices == 3) {
                setLabel(dst, "Triangle", contour)
            }
            // rectangle detection
            else if (numVertices == 4) {
                setLabel(dst, "Rectangle", contour)
            }
            // circle detection
            else if (numVertices > 6) {
                setLabel(dst, "CIR", contour)
            }
        }

        // return the matrix / image to show on the screen
        return dst
    }
    /* fun detectShapes(image: Bitmap): List<Shape> {
         val shapes = mutableListOf<Shape>()
         val mat = Mat()
         Utils.bitmapToMat(image, mat)

         // Convert to HSV color space
         Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2HSV)

         // Define red color range in HSV
 //        val lowerRed = Scalar(0.0, 100.0, 100.0)
 //        val upperRed = Scalar(10.0, 255.0, 255.0)

         val lowerPink = Scalar(150.0, 50.0, 50.0)
         val upperRed = Scalar(180.0, 255.0, 255.0)

         // Threshold the image to extract red regions
         Core.inRange(mat, lowerPink, upperRed, mat)

         // Find contours
         val contours = mutableListOf<MatOfPoint>()
         val hierarchy = Mat()
         Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

         for (contour in contours) {
             // approximate the contour to a polygon
             val approxCurve = MatOfPoint2f()
             val contourPerimeter = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
             Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approxCurve, 0.02 * contourPerimeter, true)

             // get the number of vertices of the polygon
             val numVertices = approxCurve.total().toInt()
             val contourArea = Imgproc.contourArea(contour)

             // ignore small contours and non-convex shapes
             if (Math.abs(contourArea) < 150) {
                 continue
             }

             when (numVertices) {
                 3 -> shapes.add(Triangle(approxCurve))
                 4 -> shapes.add(Rectangle(approxCurve))
                 in 5..10 -> shapes.add(Circle(approxCurve))
             }
         }

         // Annotate the original image with detected shapes
         annotatedImage = drawShapesOnImage(image, shapes)

         return shapes
     }


 */

    private fun drawShapesOnImage(image: Bitmap, shapes: List<Shape>): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(image, mat)
        val color = Scalar(0.0, 255.0, 0.0) // Green color for annotation

        for (shape in shapes) {
            val points = shape.getPoints()
            val pointsMat = Converters.vector_Point2f_to_Mat(points)

            val fontFace = Core.FONT_HERSHEY_SIMPLEX
            val fontScale = 0.5
            val thickness = 2
            val baseline = IntArray(1)
            var label = ""
            label = when (shape) {
                is Circle -> "CIRCLEEEEE"
                is Triangle -> "TRI"
                else -> "RECT"
            }

            val textSize = Imgproc.getTextSize(label, fontFace, fontScale, thickness, baseline)

            val r = Imgproc.boundingRect(pointsMat)
            val center = Point((r.x + r.width / 2).toDouble(), (r.y + r.height / 2).toDouble())
            val bottomLeft = Point(center.x - textSize.width / 2, center.y + textSize.height / 2)

            Imgproc.rectangle(mat, r.tl(), r.br(), color, 2)
            Imgproc.putText(mat, label, bottomLeft, fontFace, fontScale, color, thickness)
        }

        val annotatedImage = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, annotatedImage)

        return annotatedImage
    }

    fun getAnnotatedImage(): Bitmap? {
        return annotatedImage
    }

    private fun setLabel(dst: Mat, label: String, cnt: MatOfPoint) {
        val fontFace = Core.FONT_HERSHEY_SIMPLEX
        val fontScale = 0.5
        val thickness = 2
        val baseline = IntArray(1)

        val textSize = Imgproc.getTextSize(label, fontFace, fontScale, thickness, baseline)

        val r = Imgproc.boundingRect(cnt)
        val center = Point((r.x + r.width / 2).toDouble(), (r.y + r.height / 2).toDouble())
        val bottomLeft = Point(center.x - textSize.width / 2, center.y + textSize.height / 2)

        Imgproc.rectangle(dst, r.tl(), r.br(), Scalar(255.0, 0.0, 0.0), 2)
        Imgproc.putText(dst, label, bottomLeft, fontFace, fontScale, Scalar(255.0, 0.0, 0.0), thickness)
    }
}
