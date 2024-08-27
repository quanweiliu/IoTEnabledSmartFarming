
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

#script to perform a stratified 60/20/20 train, validation and test split.
#ensure script is run from the root directory

IMAGE_DIRECTORY = "./images/"
LABEL_DIRECTORY = "./labels/"
NEW_DIRECTORY = './data_split/'

train_dir = os.path.join(NEW_DIRECTORY,'train')
val_dir = os.path.join(NEW_DIRECTORY,'val')
test_dir = os.path.join(NEW_DIRECTORY,'test')

new_dirs = [train_dir, val_dir, test_dir]

all_labels = pd.read_csv(os.path.join(LABEL_DIRECTORY, 'labels.csv'))
min_label = all_labels['Label'].min()
max_label = all_labels['Label'].max()

for dir in new_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(min_label,max_label+1):
        if not os.path.exists(os.path.join(dir,str(i))):
            os.makedirs(os.path.join(dir,str(i)))

train_label_file = [os.path.join(LABEL_DIRECTORY, f) for f in os.listdir(LABEL_DIRECTORY) if 'train_subset' in f]
val_label_file = [os.path.join(LABEL_DIRECTORY, f) for f in os.listdir(LABEL_DIRECTORY) if 'val_subset' in f]
test_label_file = [os.path.join(LABEL_DIRECTORY, f) for f in os.listdir(LABEL_DIRECTORY) if 'test_subset' in f]

train_dfs = []
val_dfs = []
test_dfs = []

for f in train_label_file:
    train_dfs.append(pd.read_csv(f))
for f in val_label_file:
    val_dfs.append(pd.read_csv(f))
for f in test_label_file:
    test_dfs.append(pd.read_csv(f))

train_dataframe = pd.concat(train_dfs, axis=0, ignore_index=True)
val_dataframe = pd.concat(val_dfs, axis=0, ignore_index=True)
test_dataframe = pd.concat(test_dfs, axis=0, ignore_index=True)

df = pd.concat([train_dataframe, val_dataframe, test_dataframe])
df = df.drop_duplicates()

train_dataframe, test_dataframe = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=0)
train_dataframe = train_dataframe.reset_index(drop=True)
test_dataframe = test_dataframe.reset_index(drop=True)
train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.25,
                                                  stratify=train_dataframe['Label'], random_state=0)
train_dataframe = train_dataframe.reset_index(drop=True)
val_dataframe = val_dataframe.reset_index(drop=True)

for index, row in train_dataframe.iterrows():
    # Replace 'ImageFilename' with the actual column name
    source_path = os.path.join(IMAGE_DIRECTORY, row['Filename'])  # Update the path accordingly
    shutil.copy(source_path, os.path.join(train_dir, str(row['Label'])))

for index, row in val_dataframe.iterrows():
    source_path = os.path.join(IMAGE_DIRECTORY, row['Filename'])  # Update the path accordingly
    shutil.copy(source_path, os.path.join(val_dir, str(row['Label'])))

for index, row in test_dataframe.iterrows():
    source_path = os.path.join(IMAGE_DIRECTORY, row['Filename'])  # Update the path accordingly
    shutil.copy(source_path, os.path.join(test_dir, str(row['Label'])))
