def rootMeanSquareError(error: list):

    rms_error = 0

    for entry in error:
        rms_error += entry ** 2

    rms_error = rms_error / len(error)

    rms_error = rms_error ** (1 / 2)

    return rms_error