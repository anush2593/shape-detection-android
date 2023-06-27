package com.michaeltroger.shapedetection

import android.Manifest
import android.app.ActivityManager
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.michaeltroger.shapedetection.views.OverlayView
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.lang.Math.abs

/**
 * the main activity - entry to the application
 */
class MainActivity : ComponentActivity(), CvCameraViewListener2 {

    /**
     * the camera view
     */
    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    /**
     * for displaying Toast info messages
     */
    private val toast: Toast? = null

    /**
     * responsible for displaying images on top of the camera picture
     */
    private var overlayView: OverlayView? = null

    /**
     * image thresholded to black and white
     */
    private var bw: Mat? = null

    /**
     * image converted to HSV
     */
    private var hsv: Mat? = null

    /**
     * the image thresholded for the lower HSV red range
     */
    private var lowerRedRange: Mat? = null

    /**
     * the image thresholded for the upper HSV red range
     */
    private var upperRedRange: Mat? = null

    /**
     * the downscaled image (for removing noise)
     */
    private var downscaled: Mat? = null

    /**
     * the upscaled image (for removing noise)
     */
    private var upscaled: Mat? = null

    /**
     * the image changed by findContours
     */
    private var contourImage: Mat? = null

    /**
     * the activity manager needed for getting the memory info
     * which is necessary for getting the memory usage
     */
    private var activityManager: ActivityManager? = null

    /**
     * responsible for getting memory information
     */
    private var mi: ActivityManager.MemoryInfo? = null

    /**
     * the found contour as hierarchy vector
     */
    private var hierarchyOutputVector: Mat? = null

    /**
     * approximated polygonal curve with specified precision
     */
    private var approxCurve: MatOfPoint2f? = null
    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    bw = Mat()
                    hsv = Mat()
                    lowerRedRange = Mat()
                    upperRedRange = Mat()
                    downscaled = Mat()
                    upscaled = Mat()
                    contourImage = Mat()
                    hierarchyOutputVector = Mat()
                    approxCurve = MatOfPoint2f()
                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                onPermissionGranted()
            } else {
                checkPermissonAndInitialize()
            }
        }

    private fun checkPermissonAndInitialize() {
        if (ContextCompat.checkSelfPermission(baseContext, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            onPermissionGranted()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    public override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_my)

        // get the OverlayView responsible for displaying images on top of the camera
        overlayView = findViewById<View>(R.id.overlay_view) as OverlayView
        mOpenCvCameraView = findViewById<View>(R.id.java_camera_view) as CameraBridgeViewBase

        checkPermissonAndInitialize()
    }

    private fun onPermissionGranted() {
        mOpenCvCameraView!!.setMaxFrameSize(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT)

        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
        mi = ActivityManager.MemoryInfo()
        activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
        toast?.cancel()
    }

    public override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        // nothing to do
    }

    override fun onCameraViewStopped() {
        // nothing to do
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
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
            if (abs(contourArea) < 150) {
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

    /**
     * display a label in the center of the given contur (in the given image)
     * @param im the image to which the label is applied
     * @param label the label / text to display
     * @param contour the contour to which the label should apply
     */

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

    /**
     * makes an logcat/console output with the string detected
     * displays also a TOAST message and finally sends the command to the overlay
     * @param content the content of the detected barcode
     */
    private fun doSomethingWithContent(content: String) {
        Log.d(TAG, "content: $content") // for debugging in console
        val refresh = Handler(Looper.getMainLooper())
        refresh.post { overlayView!!.changeCanvas(content) }
    }

    companion object {

        /**
         * class name for debugging with logcat
         */
        private val TAG = MainActivity::class.java.name

        /**
         * detect only red objects
         */
        private const val DETECT_RED_OBJECTS_ONLY = true

        /**
         * the lower red HSV range (lower limit)
         */
        private val HSV_LOW_RED1 = Scalar(0.0, 100.0, 100.0)

        /**
         * the lower red HSV range (upper limit)
         */
        private val HSV_LOW_RED2 = Scalar(10.0, 255.0, 255.0)

        /**
         * the upper red HSV range (lower limit)
         */
        private val HSV_HIGH_RED1 = Scalar(160.0, 100.0, 100.0)

        /**
         * the upper red HSV range (upper limit)
         */
        private val HSV_HIGH_RED2 = Scalar(179.0, 255.0, 255.0)

        /**
         * frame size width
         */
        private const val FRAME_SIZE_WIDTH = 640

        /**
         * frame size height
         */
        private const val FRAME_SIZE_HEIGHT = 480

        /**
         * whether or not to use the database to display
         * an image on top of the camera
         * when false the objects are labeled with writing
         */
        private const val DISPLAY_IMAGES = false

        /**
         * Helper function to find a cosine of angle between vectors
         * from pt0->pt1 and pt0->pt2
         */
        private fun angle(pt1: Point, pt2: Point, pt0: Point): Double {
            val dx1 = pt1.x - pt0.x
            val dy1 = pt1.y - pt0.y
            val dx2 = pt2.x - pt0.x
            val dy2 = pt2.y - pt0.y
            return (
                (dx1 * dx2 + dy1 * dy2) /
                    Math.sqrt(
                        (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
                    )
                )
        }
    }

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
    }
}
