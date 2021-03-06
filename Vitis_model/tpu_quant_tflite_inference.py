
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
import pandas as pd
import time
import pickle
import sys
from glob import glob
from sklearn.metrics import f1_score, accuracy_score
from loadDataset import loadUrbansound8k

acc=0
accuracies = []
times = []
length=5
 
for fold in np.arange(1, 11):
      if fold==10:
          f=1
      else:
          f=fold+1
      DATA_PATH='tflite/'
      MODEL_PATH='build_CrossVall/fold{}/keras_model/k_model.h5'.format(f)
      TFLITE_PATH='tflite/fold{}.tflite'.format(f)
      QUANTIZED_TFLITE_PATH='tflite/fold{}_q.tflite'.format(f)
      TPU_QUANTIZED_TFLITE_PATH='tflite/fold{}_q_edgetpu.tflite'.format(f)
      (_), (_), (X_test, y_test) = loadUrbansound8k(f, fold)
      
      acc = 0
      y_pred=np.zeros(len(X_test))
      interpreter = interpreter = tflite.Interpreter(model_path=(TPU_QUANTIZED_TFLITE_PATH),
                                                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      input_shape = input_details[0]['shape']
      for i in range(len(X_test)):
      #      set_input(interpreter, x_test[i])
        input_data = X_test[i].astype(np.float32)
        input_data=input_data.reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        times.append(inference_time * 1000)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred[i]=np.argmax(output_data)
        if np.argmax(output_data) == np.argmax(y_test[i]):
            acc += 1
      Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
      acc = (acc/len(X_test)) * 100
      accuracies.append(acc)
      Y_test = np.argmax(y_test, axis=1)
      print('FOLD', fold, '; acc=', acc)


print("Accuracy mean, std | time mean, std (ms)")
print(acc, '\t', np.std(accuracies), '\t', np.mean(times), '\t', np.std(times), '\t')