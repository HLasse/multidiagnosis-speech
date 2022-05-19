import numpy as np
import math

# test against librosa.util.frame

sig = np.arange(0, 20)
sampling_rate = 1
frame_length = 5
frame_stride = 2
zero_padding = True


def stack_frames(
    sig,
    sampling_rate,
    frame_length,
    frame_stride,
    filter=lambda x: np.ones((x,)),
    zero_padding=True,
    keep_short_signals=True,
    remove_zero_padding=False,
):
    """Frame a signal into overlapping frames.
    Args:
        sig (array): The audio signal to frame of size (N,).
        sampling_rate (int): The sampling frequency of the signal.
        frame_length (float): The length of the frame in second.
        frame_stride (float): The stride between frames.
        filter (array): The time-domain filter for applying to each frame.
            By default it is one so nothing will be changed.
        zero_padding (bool): If the samples is not a multiple of
            frame_length(number of frames sample), zero padding will
            be done for generating last frame.
        keep_short_signal:  Return the original signal if shorter than frame_length.
        remove_zero_padding: Remove trailing zeros from last element following zero padding
    Returns:
            array: Stacked_frames-Array of frames of size (number_of_frames x frame_len).
    """

    # Check dimension
    s = "Signal dimention should be of the format of (N,) but it is %s instead"
    assert sig.ndim == 1, s % str(sig.shape)

    # Initial necessary values
    length_signal = sig.shape[0]
    frame_sample_length = int(
        np.round(sampling_rate * frame_length)
    )  # Defined by the number of samples
    frame_stride = float(np.round(sampling_rate * frame_stride))

    signal_length = len(sig) / sampling_rate
    if signal_length < frame_length:
        if keep_short_signals:
            if zero_padding:
                # Uncomment if you want to zero pad sequences shorter than frame_length
                # len_sig = int(frame_sample_length)
                # additive_zeros = np.zeros((len_sig - length_signal,))
                # sig = np.concatenate((sig, additive_zeros))
                return np.expand_dims(sig, axis=0)
            return np.expand_dims(sig, axis=0)
        else:
            raise ValueError(
                f"Signal is shorter than frame length {signal_length} vs {frame_length}. Set `keep_short_signal` to True to return the original signal in such cases."
            )

    # Zero padding is done for allocating space for the last frame.
    if zero_padding:
        # Calculation of number of frames
        # numframes = (int(math.ceil((length_signal
        #                             - frame_sample_length) / frame_stride)))

        # below zero pads the last, above discards the last signal
        numframes = int(
            math.ceil(
                (length_signal - (frame_sample_length - frame_stride)) / frame_stride
            )
        )
        # Zero padding
        len_sig = int(numframes * frame_stride + frame_sample_length)
        additive_zeros = np.zeros((len_sig - length_signal,))
        signal = np.concatenate((sig, additive_zeros))

    else:
        # No zero padding! The last frame which does not have enough
        # samples(remaining samples <= frame_sample_length), will be dropped!
        numframes = int(
            math.floor((length_signal - frame_sample_length) / frame_stride)
        )

        # new length
        len_sig = int((numframes - 1) * frame_stride + frame_sample_length)
        signal = sig[0:len_sig]

    # Getting the indices of all frames.
    indices = (
        np.tile(np.arange(0, frame_sample_length), (numframes, 1))
        + np.tile(
            np.arange(0, numframes * frame_stride, frame_stride),
            (frame_sample_length, 1),
        ).T
    )
    indices = np.array(indices, dtype=np.int32)

    # Extracting the frames based on the allocated indices.
    frames = signal[indices]

    # Apply the windows function
    window = np.tile(filter(frame_sample_length), (numframes, 1))
    Extracted_Frames = frames * window

    # doesn't work - can't change the shape of a signle array
    if remove_zero_padding:
        Extracted_Frames[-1] = np.trim_zeros(Extracted_Frames[-1], trim="b")

    return Extracted_Frames


l = stack_frames(
    sig, sampling_rate, frame_length, frame_stride, remove_zero_padding=False
)

stack_frames(sig, 1, 4, 1)
