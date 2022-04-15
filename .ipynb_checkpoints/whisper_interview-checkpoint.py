# Note, the main function requires matplotlib version > 3.4.0
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import shutil
from pathlib import Path
from scipy.signal import sawtooth


def decimate_by_2(input_array):
    """
    :param input_array: array-like input array (i.e. list or np.ndarray, etc.) to be decimated
                        by 2, keeping every other sample
    :return: decimated_output: output with every other sample removed (keeping only even indexes)
    """
    # I could use standard scipy functions to decimate, but this is a simple enough operation to implement
    # I'm using try/except here to handle cases in case the input is not array-like and cannot be indexed
    try:
        decimated_output = input_array[::2]  # Slice the input for all values from beginning to end incrementing by 2
    except TypeError:
        print('\tThe input cannot be properly indexed as a list or array-like structure and has not been modified.')
        print('\tError Input = ', input_array)
        print('\tError Output = ', input_array)
        return input_array

    # return the decimated output
    return decimated_output


def downsample_by_2(input_array, filter_file='kaiser_filter_32.txt',
                     url='https://00f74ba44b59ef176df449a69dcfad30158094170b-apidata.googleusercontent.com/download'
                         '/storage/v1/b/whisper-public/o/kaiser_filter_32.txt?jk=AFshE3XUKEtpzK3y4bW'
                         'ijQ1JCOZo6QM8ZEb1VFAjkwaj-mJVqqbILiCMjAKAxaqzvCXYnowW9JmHJEGg2zrOkkCGdTQ84dqWpO-Lr22'
                         '25pzc6m_SPa9xXYyw0d1fCFJMzuQYkYqdJjKiD_XF1qG0rNxiAmLylbw7CBP8SvnFgeoNoz6KBMWHs5qFAJ4fgsLY5u_Y'
                         'e88vvZYXh8xDcptT4fZ4r0-6Fp_jDIvqplNv3Xl7Z_GKMkhiiWg8iOO8Ey-4-c_G4kxelyjqBnRT0r3gYf8gqljppK1rs'
                         '-WQq5zkjFInAZg38utViLt2wIJzqBsvmGeCtNegqmPPE1ir7eA03jQ2m8SvWlXtBuMqiZTIikeyWk9aAwBduz758oNYA'
                         'URgPp_0fnnI0sN31Zu8NO2gp28pdCbP2273WLLGUs5OOiw0Zhpk5aeexlpGPpSw1DTsVHYXnUGKXIImPBzmT7E7Oo_wQq'
                         '3UAbV3KEAZw9QwYny765IsrepXykEIlvh3EGcInQ7ZMSsgbywWVxYRSrS3tpmKEpO_1Vkb-Uf7yIzoAyheuu45JL075u0'
                         'a2iWOAPRrUVqmal07im6FyEsRboJzjTnmQSR-tFCJ-IfVQe3lJsGsjG0VT9t2tMeCI0fngjOOIMblUKCLMBlZDr4m'
                         'tXtfgDlbb8nQk5L11iy6ghPUltxUhCXdgOpKaF9EF4WZPmBO6taf17-Jic__sD-f_O6L8IVAcEyCNMuyAsC'
                         'X9kBsIOempjz3rxFfkSmPsgRBr_8UNnNu5GtrI60BtGmTW9Gf209ES8WAThOCfmd1ci5RnhW6ITpwEPe'
                         '1zGdaoJSAI7Zdnx9w7YSodXOaWXY2n_6rvJlRCUNwAr8olVkX57_T-'
                         'IeInGbdmork20gIMU37txSs-y_xFengvmGRpriT3MT_UjtkZ4qfBuQPIYlS-W1arD2fZcfy6g3Goc_iBbZcVdoh'
                         'uAV56w1z0aEQbT3aj6iVsi1VOjSY4eKLvp3e6MTAacbplE0x8GxGCBIKrcsXB1C15HgZfdNIgpYkJEvzq8gOfX'
                         'Ox2dJqkw&isca=1'):

    """
    :param input_array: array-like input (i.e. list or np.ndarray, etc.) to be low-pass filtered and downsampled by 2
    :param filter_file: text file path which contains the filter coefficients for low-pass filtering
    :param url: optional url for filter coefficients file, in case you want to try a different filter
                Note: the given URL in the instructions redirects to this super long one, so downloading directly
                      did not work until I used the correct full URL to the google drive file.
    :return: downsampled_output
    """

    # check if filter coefficient file exists in the current directory, download if not
    if not Path(filter_file).is_file():
        # Download the file from the given or custom URL
        with urllib.request.urlopen(url) as response, open(filter_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    # Load filter values as np.ndarray from file
    with open(filter_file) as f:
        filter_params = f.read()

    # Convert this string to np.ndarray
    filter_params = filter_params[1:-2]  # get rid of brackets
    filter_params = np.fromstring(filter_params, dtype=float, sep= ',')

    # Do convolution with numpy. To see my own implementation of the convolution operation, see the
    # jupyter notebook example.
    filtered_signal = np.convolve(filter_params, input_array, mode='same')

    # Decimate before returning
    downsampled_output = decimate_by_2(filtered_signal)
    return downsampled_output


if __name__ == '__main__':
    # Create a 1005 sample random signal
    sample_signal = np.random.random(size=(1005,))
    decimated = decimate_by_2(sample_signal)

    print('Length of original signal: ', len(sample_signal))
    print('Length of decimated signal: ', len(decimated))

    # Requires matplotlib 3.4.0 or greater
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, num=1)
    ax1.plot(sample_signal)
    ax2.plot(decimated)

    ax1.set_title('Original Signal')
    ax2.set_title('Decimated Signal by Factor of 2')

    fig.supxlabel('Number of Samples')
    fig.supylabel('Signal Amplitude')

    print('\nThis verifies that the decimate function is working as intended.')

    # Check an input that cannot be indexed
    print('\nNow we want to try an input that cannot be indexed, setting input_array=18.')
    error_case = decimate_by_2(18)

    print('If the error is printed above, then the function works as intended.')

    # Now, let's test our downsample_by_2_function
    print('\nNow, let us generate a noisy 500 samples of a sawtooth wave signal that repeats 5 times')

    # Generate 500 samples between 0 and 1
    time = np.linspace(0, 1, 500)
    sawtooth_wave = 10 * sawtooth(2 * np.pi * 5 * time, width=0.9)

    # Add Gaussian noise and plot noisy signal
    noisy_sawtooth = sawtooth_wave + np.random.normal(size=sawtooth_wave.shape)

    # Apply Kaiser filter and decimate by 2 to downsample:
    downsampled_time = decimate_by_2(time)
    downsampled_sawtooth = downsample_by_2(noisy_sawtooth)

    # Plot final result:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, num=2)
    ax1.plot(sawtooth_wave)
    ax2.plot(noisy_sawtooth)
    ax3.plot(downsampled_sawtooth)

    ax1.set_title('Original Sawtooth Wave')
    ax2.set_title('Sawtooth Wave with Gaussian Noise')
    ax3.set_title('Filtered and Downsampled Sawtooth Wave')

    fig.supxlabel('Number of Samples')
    fig.supylabel('Signal Amplitude')

    # Show both figures
    plt.show()

    print('\nSee the generated plots for the original, noisy, and downsampled signals.')
