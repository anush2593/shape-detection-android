package com.michaeltroger.shapedetection

import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

abstract class Shape(private val points: List<Point>) {
    fun getPoints(): List<Point> {
        return points
    }
}

class Triangle(points: MatOfPoint2f) : Shape(points.toList())

class Rectangle(points: MatOfPoint2f) : Shape(points.toList())

class Circle(points: MatOfPoint2f) : Shape(points.toList())
