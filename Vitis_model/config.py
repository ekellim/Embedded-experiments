import argparse

# used folders
US_FILE_PATH = "./UrbanSound8K"
SPEC_FILE_PATH = "./spectrograms"
SPEC_META_PKL = "spec_meta.pkl"
SPEC_META_CSV = "spec_meta.csv"

# Spectrogram settings
SR = 44100          #sample rate
N_FFT = 1024        #length of the FFT window
HOP_LENGTH = 1024   #number of samples between successive frames  
N_MELS = 128        #number of Mel bands to generate
FRAME_LENGTH = 3    # Length (in seconds) of one spectrogram taken from the audio

MIN_SAMPLES = HOP_LENGTH*128-1

# Debug settings
# BATCH_SIZE = 30
# EPOCHS = 1

# Real training settings:
BATCH_SIZE = 30
EPOCHS = 100

DIVIDER = '-----------------------------------------'

class configuration:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument('-ih', '--input_height',
                        type=int,
                        default='128',
                        help='Input image height in pixels.')
        ap.add_argument('-iw', '--input_width',
                        type=int,
                        default='128',
                        help='Input image width in pixels.')
        ap.add_argument('-ic', '--input_chan',
                        type=int,
                        default='1',
                        help='Number of input image channels.')
        ap.add_argument('-kh', '--keras_hdf5',
                        type=str,
                        default='./model.hdf5',
                        help='path of Keras HDF5 file (floating point model) - must include file name. Default is ./model.hdf5')
        ap.add_argument('-tb', '--tboard',
                        type=str,
                        default='./tb_logs',
                        help='path to folder for saving TensorBoard data. Default is ./tb_logs.')   
        ap.add_argument('-vf', '--val_fold',
                        type=int,
                        default='9',
                        help='Fold to use for testing')
        ap.add_argument('-tf', '--test_fold',
                        type=int,
                        default='10',
                        help='Fold to use for testing')


        # Spectrogram
        ap.add_argument('-SR', '--sampling_rate',
                        type=int,
                        default=44100,
                        help='sampling rate for the audio')
        ap.add_argument('-NF', '--N_FFT',
                        type=int,
                        default=1024,
                        help='Number of FFTs to use for the spectrogram')
        ap.add_argument('-HL', '--hop_length',
                        type=int,
                        default=1024,
                        help='Hop length of the spectrogram')

        ap.add_argument('-q', '--quant_model',
                        type=str,
                        default='./q_model.h5',
                        help='Full path of quantized model. Default is ./q_model.h5')

        # Export application code
        ap.add_argument('-od', '--output_dir',
                        type=str,
                        default="./build/fold10/compile",
                        help="Target directory for the output files.")
        ap.add_argument('-ea', '--export_all',
                        type=int,
                        default=0,
                        help="Flag to indicate all images from all folds should be exported, or only the images from the test fold. Default is false (0), to enable, pass the value \"1\"")
            
        
        args = ap.parse_args()

        self.inputWidth = args.input_width
        self.inputHeight = args.input_height
        self.inputChan = args.input_chan
        self.keras_hdf5 = args.keras_hdf5
        self.tboard = args.tboard
        self.testFold = args.test_fold
        self.valFold = args.val_fold
        self.quantizedModel = args.quant_model
        self.outputDir = args.output_dir
        self.exportAll = bool(args.export_all)
        

    def printTrainSettings(self):
        print (' Command line options:')
        print ('--input_height : ', self.inputHeight)
        print ('--input_width  : ', self.inputWidth)
        print ('--input_chan   : ', self.inputChan)
        print ('--keras_hdf5   : ', self.keras_hdf5)
        print ('--tboard       : ', self.tboard)
        print ('--testFold     : ', self.testFold)
        print ('--valFold      : ', self.valFold)
        print()
        print('Batch size      : ', BATCH_SIZE)
        print('EPOCHS          : ', EPOCHS)


    def printQuantizationSettings(self):
        print ('--keras_hdf5   : ', self.keras_hdf5)
        print ('--q_keras_hdf5 : ', self.quantizedModel)
        print ('--testFold     : ', self.testFold)
        print()
        print('Batch size      : ', BATCH_SIZE)

    def printExportSettings(self):
        print ('--testFold     : ', self.testFold)
        print ('--output_dir   : ', self.outputDir)
        print ('--export_all   : ', self.exportAll)
