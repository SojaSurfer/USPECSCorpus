import sys
from pathlib import Path

from wordcloud import WordCloud, STOPWORDS
# import imageio.v3 as iio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from analysis import loadMetadata, concatTables, concatTexts

def getCircularMask(grid_size:int) -> np.ndarray:
    # Create the grid
    x, y = np.ogrid[:grid_size, :grid_size]

    # Define the center and radius of the circle
    center = grid_size // 2
    radius = 260  # Increase the radius to upscale the circle

    # Create the mask
    mask = (x - center) ** 2 + (y - center) ** 2 > radius ** 2
    mask = 255 * mask.astype(int)
    return mask

def createWordClouds(show: bool = False) -> None:

    path = Path('analysis') / 'wordClouds'
    metadataDF = loadMetadata()


    mask = getCircularMask(grid_size=600)
    stopwords = set(STOPWORDS) | {'cheers', 'applause', 'thank'}

    # imagePath = '/Users/julian/Downloads/silhouette_trump.jpeg'
    # img = iio.imread(imagePath)

    # threshold = 200
    # img = np.where(img >= threshold, 255, img)

    # alpha = np.where(img[...,0] == 255, 0, 255)


    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]
        texts = concatTexts(df)


        wc = WordCloud(background_color='white', max_words=500, mask=mask, # mask=img,
                    stopwords=stopwords, # contour_width=3, contour_color='black'
                    )

        # generate word cloud
        wc.generate(texts)
        

        if show:
            # show
            fig, axis = plt.subplots(figsize=(10,10))
            axis.imshow(wc)
            plt.axis("off")
            axis.set_title(f'Word Cloud {speaker}')

            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            wc.to_file(path / f'wordCloud_{speaker}.png')


if __name__ == '__main__':
    createWordClouds()