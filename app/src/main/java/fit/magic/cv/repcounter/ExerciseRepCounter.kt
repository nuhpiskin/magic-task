// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.repcounter

import fit.magic.cv.PoseLandmarkerHelper

abstract class ExerciseRepCounter {

    private var listener: ExerciseEventListener? = null

    private var repCount = 0

    abstract fun setResults(resultBundle: PoseLandmarkerHelper.ResultBundle)

    fun setListener(listener: ExerciseEventListener?) {
        this.listener = listener
    }

    /*
    Increments the rep count by 1.
     */
    fun incrementRepCount() {
        repCount++
        listener?.repCountUpdated(repCount)
    }

    /*
    Resets the rep count to 0.
    */
    fun resetRepCount() {
        repCount = 0
        listener?.repCountUpdated(repCount)
    }

    /*
    Updates the progress bar. Should be a value between 0 and 1.
     */
    fun sendProgressUpdate(progress: Float) {
        listener?.progressUpdated(progress)
    }

    /*
    Displays a feedback message.
     */
    fun sendFeedbackMessage(message: String) {
        listener?.showFeedback(message)
    }
}

interface ExerciseEventListener {
    fun repCountUpdated(count: Int)

    fun progressUpdated(progress: Float)

    fun showFeedback(feedback: String)
}