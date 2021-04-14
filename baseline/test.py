def _sample_indices_adv(sample_length, start_idx, frame_count, input_freq, output_freq):
    if input_freq < output_freq:
        # repetition is used to fill in the gaps
        repeat = output_freq // input_freq 
        indices = []
        for x in range(sample_length // repeat):
            next_idx = (start_idx + x) if (start_idx + x) < frame_count else (frame_count - 1)
            indices.extend([next_idx] * repeat)
    else:
        step = input_freq // output_freq
        indices = [(start_idx + x * step) if (start_idx + x * step) <frame_count else (frame_count - step) for x in range(sample_length)]
    return indices

print(_sample_indices_adv(10, 0, 60, 6, 6))