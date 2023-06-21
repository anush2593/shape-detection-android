package com.michaeltroger.shapedetection.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.drawable.Drawable
import com.michaeltroger.shapedetection.R
import android.graphics.PorterDuff
import android.media.MediaPlayer
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.appcompat.content.res.AppCompatResources.getDrawable

/**
 * The overlay view is responsible for displaying
 * information on top of the camera
 * @param mContext the context
 * @param attrs the attributes
 * @author Michael Troger
 */
class OverlayView(private val mContext: Context, attrs: AttributeSet?) : View(mContext, attrs) {
    /**
     * holds the rectangle image
     */
    private var rectangle: Drawable? = null

    /**
     * holds the triangle image
     */
    private var triangle: Drawable? = null

    /**
     * holds the triangle image
     */
    private var circle: Drawable? = null

    /**
     * command by which the canvas should be changed
     * e.g. to switch the image
     */
    private var changingType: String? = null

    /**
     * the sound of a bear
     */
    private var bearSound: MediaPlayer? = null

    /**
     * the sound of a chicken
     */
    private var chickenSound: MediaPlayer? = null

    /**
     * sounds are preloaded so that they must no be loaded in the main loop
     */
    private fun loadSounds() {
        bearSound = MediaPlayer.create(mContext, R.raw.bear)
        chickenSound = MediaPlayer.create(mContext, R.raw.chicken)
    }

    /**
     * images are preloaded so that they must no be loaded in the main loop
     */
    private fun loadImages() {
        rectangle = getDrawable(context, R.drawable.rectangle)
        triangle = getDrawable(context, R.drawable.triangle)
        circle = getDrawable(context, R.drawable.circle)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (changingType != null) {
            when (changingType) {
                "rectangle" -> {
                    clearCanvas(canvas)
                    drawRectangle(canvas)
                    bearSound!!.start()
                }
                "triangle" -> {
                    clearCanvas(canvas)
                    drawTriangle(canvas)
                    chickenSound!!.start()
                }

                "circle" -> {
                    clearCanvas(canvas)
                    drawCircle(canvas)
                    chickenSound!!.start()
                }
            }
        }
    }

    private fun drawRectangle(canvas: Canvas) {
        val imageBounds = canvas.clipBounds
        rectangle!!.bounds = imageBounds
        rectangle!!.draw(canvas)
    }

    private fun drawTriangle(canvas: Canvas) {
        val imageBounds = canvas.clipBounds
        triangle!!.bounds = imageBounds
        triangle!!.draw(canvas)
    }

    private fun drawCircle(canvas: Canvas) {
        val imageBounds = canvas.clipBounds
        circle!!.bounds = imageBounds
        circle!!.draw(canvas)
    }

    private fun clearCanvas(canvas: Canvas) {
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
    }

    /**
     * change the canvas by given command
     * @param changingType the command as String
     */
    fun changeCanvas(changingType: String) {
        // force redraw with the given command
        // but only if same command has not been used before
        if (this.changingType == null
                || this.changingType != changingType) {
            this.invalidate()
            this.changingType = changingType
        }
    }

    companion object {
        /**
         * class name for debugging with logcat
         */
        private val TAG = OverlayView::class.java.name
    }

    init {
        // preload sounds and images
        loadSounds()
        loadImages()
        Log.d(TAG, "started :)")
    }
}