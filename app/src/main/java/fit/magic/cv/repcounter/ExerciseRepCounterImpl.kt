// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import android.icu.text.IDNA.Info
import android.util.Log
import android.widget.Switch
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import fit.magic.cv.PoseLandmarkerHelper
import kotlin.math.abs

enum class Posture {
    Stand, LeftStep, RightStep
}
data class Landmark(val x: Float, val y: Float, val z: Float)

class ExerciseRepCounterImpl : ExerciseRepCounter() {
    // Predefined angles for knee-down posture normalized with 360
    private val leftKneeDownAngles = listOf(0.25, 0.25, 0.25, 0.5)
    private val rightKneeDownAngles = listOf(0.25, 0.5, 0.25, 0.25)

    // Smooth parameters
    private var smoothedProgress = 0.0
    private val smoothingFactor = 0.3

    private var currentPosture = Posture.Stand

    override fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {

        // Check if results are available
        resultBundle?.results?.firstOrNull()?.worldLandmarks()?.getOrNull(0)?.let { worldLandmarks ->
            if (worldLandmarks.size >= 33) {
                try {
                    // Indices representing body parts in order:
                    // 27 - left ankle, 28 - right ankle, 25 - left knee, 26 - right knee,
                    // 23 - left hip, 24 - right hip, 11 - left shoulder, 12 - right shoulder
                    val indices = listOf(27, 28, 25, 26, 23, 24, 11, 12)

                    // Extract predefined landmarks
                    if (indices.all { it < worldLandmarks.size }) {
                        val landmarks = indices.map { index ->
                            val landmark = worldLandmarks[index]
                            Landmark(landmark.x(), landmark.y(), landmark.z())
                        }

                        // Assign body landmarks
                        val leftAnkle = landmarks[0]
                        val rightAnkle = landmarks[1]
                        val leftKnee = landmarks[2]
                        val rightKnee = landmarks[3]
                        val leftHip = landmarks[4]
                        val rightHip = landmarks[5]
                        val leftShoulder = landmarks[6]
                        val rightShoulder = landmarks[7]

                        // Calculate joint angles
                        val currentAngles = listOf(
                            calculate3PointAngle(leftAnkle, leftKnee, leftHip),
                            calculate3PointAngle(leftKnee, leftHip, leftShoulder),
                            calculate3PointAngle(rightAnkle, rightKnee, rightHip),
                            calculate3PointAngle(rightKnee, rightHip, rightShoulder)
                        )

                        // Calculate cosine similarity with predefined angles
                        val cosineSimLeftSide = cosineSimilarity(leftKneeDownAngles, currentAngles)
                        val cosineSimRightSide = cosineSimilarity(rightKneeDownAngles, currentAngles)

                        // Update smoothed progress
                        val bestStepSim = maxOf(cosineSimLeftSide, cosineSimRightSide)
                        val progressDiff = maxOf((bestStepSim - 0.95) / 0.04, 0.0)
                        smoothedProgress = (smoothingFactor * progressDiff) + ((1.0 - smoothingFactor) * smoothedProgress)
                        sendProgressUpdate(smoothedProgress.toFloat())

                        // Update current posture based on angles
                        updatePosture(cosineSimLeftSide, cosineSimRightSide, smoothedProgress)
                    } else {
                        Log.e("ERROR", "Index out of bounds for worldLandmarks size: ${worldLandmarks.size}")
                    }
                } catch (e: Exception) {
                    Log.e("ERROR", "Failed to process landmarks: ${e.message}")
                }
            } else {
                Log.w("WARNING", "worldLandmarks is null, empty, or insufficient in size.")
            }
        }
    }

    // Updates the posture based on similarity scores
    private fun updatePosture(cosineSimLeft: Double, cosineSimRight: Double, smoothedProgress: Double) {
        when {
            // Transition to right step
            cosineSimRight > 0.985 && currentPosture == Posture.LeftStep && smoothedProgress > 0.99 -> {
                incrementRepCount()
                currentPosture = Posture.RightStep
            }
            // Transition to left step
            cosineSimLeft > 0.985 && currentPosture == Posture.RightStep && smoothedProgress > 0.99 -> {
                incrementRepCount()
                currentPosture = Posture.LeftStep
            }
            // Start stepping
            abs(cosineSimLeft - cosineSimRight) > 0.07 && currentPosture == Posture.Stand -> {
                currentPosture = if (cosineSimRight > cosineSimLeft) Posture.LeftStep else Posture.RightStep
            }
            // Return to standing posture
            cosineSimLeft < 0.96 && cosineSimRight < 0.96 && currentPosture != Posture.Stand -> {
                currentPosture = Posture.Stand
            }
        }
    }

    // Calculates cosine similarity between two vectors
    private fun cosineSimilarity(vec1: List<Double>, vec2: List<Double>): Double {
        val dotProduct = vec1.zip(vec2).sumOf { it.first * it.second }
        val magnitude1 = Math.sqrt(vec1.sumOf { it * it })
        val magnitude2 = Math.sqrt(vec2.sumOf { it * it })
        return if (magnitude1 > 0 && magnitude2 > 0) dotProduct / (magnitude1 * magnitude2) else 0.0
    }

    // Calculates vector from two points
    private fun calculateVector(p1: Landmark, p2: Landmark): Landmark {
        return Landmark(
            x = p2.x - p1.x,
            y = p2.y - p1.y,
            z = p2.z - p1.z
        )
    }

    // Calculates the angle between two vectors
    private fun calculateAngle(v1: Landmark, v2: Landmark): Double {
        val dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        val magnitude1 = Math.sqrt((v1.x * v1.x + v1.y * v1.y + v1.z * v1.z).toDouble())
        val magnitude2 = Math.sqrt((v2.x * v2.x + v2.y * v2.y + v2.z * v2.z).toDouble())
        return Math.toDegrees(Math.acos(dotProduct / (magnitude1 * magnitude2))) / 360.0
    }

    // Calculates the angle at p2 from three points
    private fun calculate3PointAngle(p1: Landmark, p2: Landmark, p3: Landmark): Double {
        return calculateAngle(calculateVector(p1, p2), calculateVector(p3, p2))
    }
}


