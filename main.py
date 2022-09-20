import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import argparse
import os
import keras as ks
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()
'''

def show_dataset_info(pokedex): 
    print(pokedex.info()) 
    print(pokedex.head())

def show_example_images(image_folder): 
    images = sorted(os.listdir(image_folder))
    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()
    for idx, img_file in enumerate(images):
        if idx >= len(axes):
            break
        img = mimg.imread(os.path.join(image_folder, img_file))
        print(f'Image shape = {img.shape}')
        axes[idx].imshow(img)
        axes[idx].set_title(img_file.split('.')[0])
        axes[idx].axis('off')
    plt.show()

def load_pokedex(desciption_file, image_folder):
    pokedex = pd.read_csv(desciption_file)
    pokedex.drop('Type2', axis=1, inplace=True)
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)

    images = sorted(os.listdir(image_folder))
    images = list(map(lambda image_file: os.path.join(image_folder, image_file), images))
    pokedex['Image'] = images

    return pokedex

def prepare_data_for_network(pokedex):
    data_generator = ImageDataGenerator(validation_split=0.1, rescale=1.0/255)
    train_generator = data_generator.flow_from_dataframe(pokedex, x_col='Image', y_col='Type1', subset='training', color_mode='rgba', class_mode='categorical')
    return train_generator


def parse_arguemnts():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file', default='pokemon.csv',
        help='A CSV file with pokemon information')
    parser.add_argument('-i', '--image_folder', default='images',
        help='Folder with pokemon images')
    return parser.parse_args()

def show_generator_results(generator):
    for i in range(10):
        plt.subplot(2, 5, i+1)
        for x, y in generator:
            img = x[0]
            print(f'Generator image shape {img.shape}')
            plt.imshow(img)
            break
    plt.show()


def main():

    args = parse_arguemnts()
    pokedex = load_pokedex(args.description_file, args.image_folder)
    show_dataset_info(pokedex)
    show_example_images(args.image_folder)
    generator = prepare_data_for_network(pokedex)
    show_generator_results(generator)

    model = ks.models.Sequential()

    model.add(ks.layers.Conv2D(34, (3,3), activation='relu', input_shape=(256,256,4)))
    model.add(ks.layers.MaxPooling2D(2,2))

    model.add(ks.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(ks.layers.MaxPooling2D(2,2))

    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(128, activation='relu'))
    model.add(ks.layers.Dense(18, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    history = model.fit_generator(generator, epochs=10)
    plt.plot(history.history['acc'])

if __name__ == '__main__':
    main()