import importlib
import inspect
import tensorflow.python.keras.models
import tensorflow

#ensure script is run from the root directory
IMG_SIZE = (224, 224, 3)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
out_dir = './models/'

print()
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print()

from tensorflow.keras.applications import MobileNetV3Small
m = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
m._name = 'MobileNetV3Small'
tensorflow.keras.models.save_model(m, '{0}{1}.hdf5'.format(out_dir, m._name))

from tensorflow.keras.applications import MobileNetV3Large
m = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
m._name = 'MobileNetV3Large'
tensorflow.keras.models.save_model(m, '{0}{1}.hdf5'.format(out_dir, m._name))


import tensorflow.keras.applications

# Get a list of all the model names in keras.applications
all_model_names = ['tensorflow.keras.applications.' + name for name in dir(tensorflow.keras.applications)]
all_model_names2 = ['tensorflow.keras' + name for name in dir(tensorflow.keras.applications)]
# print(all_model_names)
# all_model_names = ['tensorflow.keras.applications.' + name for name in dir(kapp) if
#                    name.islower() and not name.startswith("__")]
imported_modules = []

for module_name in all_model_names:
    try:
        module = importlib.import_module(module_name)
        imported_modules.append(module)
    except ImportError:
        print(f"Failed to import module: {module_name}")
        
for module_name in all_model_names2:
    try:
        module = importlib.import_module(module_name)
        imported_modules.append(module)
    except ImportError:
        print(f"Failed to import module: {module_name}")

for module in imported_modules:
    x = dir(module)
    a = 1
    for function_name in dir(module):
        function = getattr(module, function_name)
        if callable(function):
            print(function_name)
            try:
                m = function(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
                m._name = function_name
                tensorflow.keras.models.save_model(m, '{0}{1}.hdf5'.format(out_dir, function_name))
                # print("Success: {0}".format(function_name))
                # print()
            except Exception as E:
                print("Fail: {0}{1}".format(module, function_name))
                print(E)
                print()
            # Now 'function' is the callable function from the module
            # You can use it as needed
            pass
        

