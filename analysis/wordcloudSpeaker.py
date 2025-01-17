import sys
from pathlib import Path

from wordcloud import WordCloud, STOPWORDS
import imageio.v3 as iio
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


def wordCloudPerSpeaker(show: bool = False) -> None:

    path = Path('analysis') / 'wordClouds'
    metadataDF = loadMetadata()


    # mask = getCircularMask(grid_size=600)
    stopwords = set(STOPWORDS) | {'cheers', 'applause', 'thank'}

    imagePath = Path('analysis/wordClouds/Politician_top.png')
    img = iio.imread(imagePath)
    imgBottom = iio.imread(Path('analysis/wordClouds/Politician_bottom.png'))

    threshold = 200
    img = np.where(img >= threshold, 255, img)

    # alpha = np.where(img[...,0] == 255, 0, 255)


    for speaker in metadataDF['speaker'].unique():
        df = metadataDF[metadataDF['speaker'] == speaker]
        texts = concatTexts(df)


        wc = WordCloud(background_color='white', max_words=500, mask=img,
                    stopwords=stopwords, contour_width=5, contour_color='black',
                    )

        # generate word cloud
        wc.generate(texts)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
        ax1.imshow(wc)
        ax1.axis('off')
        ax1.set_title(f'Word Cloud {speaker}')
        ax2.imshow(imgBottom)
        ax2.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0) 

        if show:
            plt.show()
            plt.close()
 
        else:
            plt.savefig(path / f'wordCloud_{speaker}.png')
    
    return None


def wordCloudPerPeriod(show: bool = False) -> None:

    path = Path('analysis') / 'wordClouds'
    metadataDF = loadMetadata()


    # mask = getCircularMask(grid_size=600)
    stopwords = set(STOPWORDS) | {'cheers', 'applause', 'thank'}

    imagePath = Path('analysis/wordClouds/Politician_top.png')
    img = iio.imread(imagePath)
    imgBottom = iio.imread(Path('analysis/wordClouds/Politician_bottom.png'))

    threshold = 200
    img = np.where(img >= threshold, 255, img)


    for period in metadataDF['period'].unique():
        periodDF = metadataDF[metadataDF['period'] == period]

        fig, axes = plt.subplots(2, 2, figsize=(20,10))

        for i, speaker in enumerate(periodDF['speaker'].unique()):
            df = periodDF[periodDF['speaker'] == speaker]
            texts = concatTexts(df)


            wc = WordCloud(background_color='white', max_words=500, mask=img,
                        stopwords=stopwords, contour_width=5, contour_color='black',
                        )

            # generate word cloud
            wc.generate(texts)
            
            row, col = divmod(i, 2) 

            axes[row,col].imshow(wc)
            axes[row,col].axis('off')
            axes[row,col].set_title(f'Word Cloud {speaker}')
            axes[row+1,col].imshow(imgBottom)
            axes[row+1,col].axis('off')


        plt.tight_layout()
        plt.subplots_adjust(hspace=0) 

        if show:
            plt.show()
            plt.close()
            return None

        else:
            plt.savefig(path / f'wordCloud_period{period}.png')
        
    return None


def wordCloudPerPeriod2(show: bool = False) -> None:

    path = Path('analysis') / 'wordClouds'
    metadataDF = loadMetadata()

    silhouettePath = Path('analysis/wordClouds/silhouettes')

    # mask = getCircularMask(grid_size=600)
    stopwords = set(STOPWORDS) | {'cheers', 'applause', 'thank'}
    threshold = 200
    


    for period in metadataDF['period'].unique():
        periodDF = metadataDF[metadataDF['period'] == period]

        fig, axes = plt.subplots(1, 2, figsize=(20,10))

        for i, speaker in enumerate(periodDF['speaker'].unique()):
            df = periodDF[periodDF['speaker'] == speaker]
            texts = concatTexts(df)

            try:
                img = iio.imread(silhouettePath / f'Silhouette {speaker}.png')
            except FileNotFoundError:
                img = iio.imread(silhouettePath / f'Silhouette.png')

            if i == 0:
                img = np.flip(img, axis=1)
            img = np.where(img >= threshold, 255, img)

            wc = WordCloud(background_color='white', max_words=250, mask=img,
                        stopwords=stopwords, contour_width=5, contour_color='black',
                        )

            # generate word cloud
            wc.generate(texts)
            

            axes[i].imshow(wc)
            axes[i].set_title(f'Word Cloud {speaker}', fontsize=22)
            axes[i].axis('off')


        plt.tight_layout(pad=3)
        # plt.subplots_adjust(hspace=0) 

        if show:
            plt.show()
            plt.close()
            

        else:
            plt.savefig(path / f'wordCloud_period{period}.png')
        
    return None




if __name__ == '__main__':
    # wordCloudPerPeriod(show=False)
    wordCloudPerPeriod2(show=True)

