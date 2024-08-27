# DeepWeeds_classification_tensorflow
Benchmarks tensorflow.keras models on the DeepWeeds dataset. This code is modified from https://github.com/AlexOlsen/DeepWeeds.
The download location of the dataset download is also found at https://github.com/AlexOlsen/DeepWeeds.

Ensure that the Root Directory has the following file structure:

./models
./images
./data_split
./labels
./outputs
deepweeds3.py
download_models.py
split_dataset.py

Firstly create a conda environment which uses tf_2_10.yml.

Then ensure all of the images are placed in ./images. Ensure all of the .csv files seen in this repository are placed in ./labels. Once split_dataset.py is run, the train, validation and test splits are seen in ./data_split. Once download_models.py has been run, the ./models directory will be populated with tensorflow.keras models in the .hdf5 format. 

Other arbitrary deep learning encoders which accept (224x244) input size with .hdf5 format can also be trained. Once the previous steps are complete. Then run deepweeds3.py --idx 0 --step 1 --compute_batch_size 0 --compute_overall_summary 0. To train all models sequentially on a single GPU. If HPC access is available, the --idx and --step arguments can be used to train batch jobs on different GPUs. Allocating 6 different jobs to a HPC would consist of running 6 different .sh files each with a different --idx argument (from 0 to 5) and a --step 6 for all .sh files.

deepweeds3.py can be run with commands such as --compute_overall_summary where 1 creates a .csv file which summarizes the testing performance of previously benchmarked models.


