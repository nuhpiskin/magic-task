// Copyright (c) 2024 Magic Tech Ltd

package fit.magic.cv.exercise_info

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.TextView

import fit.magic.cv.R

class ExerciseInfoView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {

    private lateinit var tvFeedback: TextView
    private lateinit var tvRepCount: TextView

    private lateinit var progress: ProgressBar

    init {
        init(context)
    }

    private fun init(context: Context) {
        val view = LayoutInflater.from(context).inflate(R.layout.view_exercise_info, this, true)

        tvFeedback = view.findViewById(R.id.text_view_feedback)
        tvRepCount = view.findViewById(R.id.text_view_rep_count)

        progress = view.findViewById(R.id.rep_progress)
    }

    fun setFeedback(feedback: String) {
        tvFeedback.text = feedback
    }

    fun setRepCount(count: Int) {
        tvRepCount.text = count.toString()
    }

    fun setProgress(value: Float) {
        val coercedValue = value.coerceIn(0.0f, 1.0f)
        progress.progress = (coercedValue * 100).toInt()
    }
}
