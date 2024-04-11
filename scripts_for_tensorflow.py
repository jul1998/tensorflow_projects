import os
import shutil
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def create_subset_directory(original_dir, subset_dir, percentage=0.1):
    """
    Create a subset directory containing a specified percentage of images from each class directory.

    Parameters:
    - original_dir (str): The path to the original directory containing class subdirectories.
    - subset_dir (str): The path to the directory where the subset will be created.
    - percentage (float): The percentage of images to select from each class directory.

    Returns:
    None

    This function removes the existing subset directory (if it exists) and creates a new one.
    It then iterates through each class directory in the original directory, selects a random
    subset of images based on the specified percentage, and copies those images to the subset
    class directory.

    Example:
    ```python
    original_train_dir = "/path/to/original/train"
    subset_train_dir = "/path/to/subset/train"
    create_subset_directory(original_train_dir, subset_train_dir, percentage=0.2)
    ```

    In this example, the function removes the existing subset directory, creates a new one,
    and selects 20% of images from each class directory in the original train directory,
    copying them to the corresponding subset class directory.
    """
    # Remove the existing subset directory if it exists
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)

    # Create the subset directory
    os.makedirs(subset_dir)

    # Iterate through each class directory in the original directory
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        subset_class_dir = os.path.join(subset_dir, class_name)

        # Create the subset class directory if it doesn't exist
        os.makedirs(subset_class_dir, exist_ok=True)

        # Get the list of files in the class directory
        files = os.listdir(class_dir)

        # Calculate the number of files to select for the subset
        num_files_to_select = int(len(files) * percentage)

        # Select a random subset of files
        selected_files = random.sample(files, num_files_to_select)

        # Copy the selected files to the subset class directory
        for file_name in selected_files:
            source_path = os.path.join(class_dir, file_name)
            destination_path = os.path.join(subset_class_dir, file_name)
            shutil.copy(source_path, destination_path)



def display_random_image_grid_with_labels(folder_path, num_rows, num_cols):
    """
    Display a grid of randomly selected images from subfolders with labels.

    Parameters:
    - folder_path (str): Path to the folder containing subfolders with images.
    - num_rows (int): Number of rows in the grid.
    - num_cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Get a list of subfolders (class labels)
    class_labels = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    for i in range(num_rows):
        for j in range(num_cols):
            class_label = random.choice(class_labels)
            class_folder_path = os.path.join(folder_path, class_label)

            # Get a list of image files in the selected subfolder
            image_files = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if image_files:
                # Shuffle the list of image files
                random.shuffle(image_files)

                # Select a random image to display
                img_path = os.path.join(class_folder_path, random.choice(image_files))
                img = mpimg.imread(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                axes[i, j].set_title(class_label, fontsize=10)

    plt.tight_layout()
    plt.show()

def select_percentage_of_data(src_dir, dest_dir, percentage):
    """
    Select a specified percentage of data from each subdirectory in the source directory and copy it to the destination directory.

    Parameters:
    - src_dir (str): The path to the source directory containing subdirectories representing categories.
    - dest_dir (str): The path to the destination directory where the selected data will be copied.
    - percentage (float): The percentage of data to select from each subdirectory.

    Returns:
    None

    This function iterates over each subdirectory in the source directory, selects a specified
    percentage of files randomly, and copies them to the corresponding subdirectory in the
    destination directory.

    Example:
    ```python
    source_directory = "/path/to/source/data"
    destination_directory = "/path/to/destination/data"
    select_percentage_of_data(source_directory, destination_directory, percentage=10.0)
    ```

    In this example, the function selects 10% of data from each subdirectory in the source
    directory and copies it to the corresponding subdirectory in the destination directory.
    """
    # Get the list of subdirectories (each subdirectory represents a category)
    categories = os.listdir(src_dir)

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over each category
    for category in categories:
        category_src_dir = os.path.join(src_dir, category)
        category_dest_dir = os.path.join(dest_dir, category)

        # Get the list of all files in the current category
        all_files = os.listdir(category_src_dir)

        # Calculate the number of files to select based on the percentage
        num_files_to_select = int(len(all_files) * (percentage / 100))

        # Randomly select files
        selected_files = random.sample(all_files, num_files_to_select)

        # Create the destination directory for the current category
        os.makedirs(category_dest_dir, exist_ok=True)

        # Copy selected files to the destination directory
        for file in selected_files:
            src_path = os.path.join(category_src_dir, file)
            dest_path = os.path.join(category_dest_dir, file)
            shutil.copy(src_path, dest_path)
   

def select_random_image(class_names, dataset):
  class_name = random.choice(class_names)
  filename = random.choice(os.listdir(dataset + "/" + class_name))
  filepath = val_dir_10_percent + "/" + class_name + "/" + filename
  return filepath


def predictor(img, model, train_dataset, label_names):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype = 'float32')/255.0
    plt.imshow(image)
    image = image.reshape(1, 224,224,3)
    
    
    dict_class = dict(zip(list(range(len(label_names))), label_names))
    clas = model.predict(image).argmax()
    name = dict_class[clas]
    print('The given image is of \nClass: {0} \nAnimal: {1}'.format(clas, name))


def get_sample_from_df(percentage, df):
  # Specify the percentage of data you want to sample (e.g., 10%)
  percentage_to_sample = percentage

  # Use the sample method to get the specified percentage
  sampled_df = df.sample(frac=percentage_to_sample / 100)

  # Display the sampled DataFrame
  return sampled_df


def text_preprocessing(text):
    text = re.sub(r'[^a-zA-Z\s]', "", str(text))
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'http\S+', '', str(text))
    text = whitespace.sub(' ', text)
    text = web_address.sub('', text)
    text = user.sub('', text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


# import nltk
# nltk.download('stopwords')
# import nltk
# nltk.download('punkt')
# import nltk
# nltk.download('wordnet')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def check_df(df, head=5):
      """
      Prints a summary of a dataframe.
    
      Args:
          df: A Pandas DataFrame.
          head: The number of rows to print from the head of the DataFrame.
      """
      print("-" * 80)
      print(f"DataFrame Shape: {df.shape}")
      print("-" * 80)
      print(f"Data types:\n{df.dtypes}")
      print("-" * 80)
      print(f"Head {head}:\n{df.head(head)}")
      print("-" * 80)
      print(f"Tail {head}:\n{df.tail(head)}")
      print("-" * 80)
      print(f"Missing values:\n{df.isnull().sum()}")
      print("-" * 80)
      print(f"Quantiles:\n{df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])}")
      print("-" * 80)


def get_columns(df):
  num_attribs = df.select_dtypes(include=[np.number]).columns.tolist()
  cat_attribs = [col for col in df.columns if col not in num_attribs] 
  return num_attribs, cat_attribs
