import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import imageio
from PIL import Image

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
# from eval_func.spice.spice import Spice
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def draw_from_dict(dicdata,RANGE, heng=0):
    #dicdata：字典的数据。
    #RANGE：截取显示的字典的长度。
    #heng=0，代表条状图的柱子是竖直向上的。heng=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴。
    # plt.subplot(2, 2, 1)
    # plt.figure(2,2,figsize=(14, 14), dpi=100)
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    # plt.figure(figsize=(26, 6.5))

    y = np.arange(RANGE)
    y0 = list(dicdata[0].values())#[6, 10, 4, 5, 1]
    y1 = list(dicdata[1].values())#[2, 6, 3, 8, 5]
    y2 = list(dicdata[2].values())

    bar_width = 0.2
    tick_label = list(dicdata[0].keys())

    plt.barh(y, y0, bar_width, align="center", color="r", label="train", alpha=1)
    plt.barh(y + bar_width, y1, bar_width, color="g", align="center", label="val", alpha=1)
    plt.barh(y + 2*bar_width, y2, bar_width, color="b", align="center", label="test", alpha=1)

    # plt.xlabel("词")
    # plt.ylabel("词频率")

    plt.yticks(y + bar_width / 0.8, tick_label,size = 10)

    plt.legend()

    plt.show()
    # for i in range(4):
    #     by_value = sorted(dicdata[i].items(),key = lambda item:item[1],reverse=True)
    #     x = []
    #     y = []
    #
    #     # plt.xlim((-2, 2))
    #     for d in by_value:
    #         x.append(d[0])
    #         y.append(d[1])
    #     if heng == 0:
    #         plt.subplot(2, 2, i + 1)
    #         plt.bar(x[0:RANGE], y[0:RANGE])
    #         # plt.show()
    #         # return
    #     elif heng == 1:
    #         plt.xlim((0, 2))
    #         plt.subplot(2, 2, i + 1)
    #         plt.barh(x[0:RANGE], y[0:RANGE])
    #         # plt.show()
    #         # return
    #     else:
    #         return "heng的值仅为0或1！"
    # plt.show()

def create_input_files2(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k','RSICD'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    test_5youxiao_image_paths=[]

    classlist = ['tree','building','airport','bareland','baseballfield','beach','bridge','center','church','commercial','denseresidential','desert',
                'farmland','forest','industrial','meadow','mediumresidential','mountain','park','school','square','parking',
                'playground','pond','viaduct','port','railwaystation','resort','river','sparseresidential','storagetanks','stadium']
    train_dict = {'tree':0,'building':0,'airport':0,'bareland':0,'baseballfield':0,'beach':0,'bridge':0,'center':0,'church':0,'commercial':0,'denseresidential':0,'desert':0,
                'farmland':0,'forest':0,'industrial':0,'meadow':0,'mediumresidential':0,'mountain':0,'park':0,'school':0,'square':0,'parking':0,
                'playground':0,'pond':0,'viaduct':0,'port':0,'railwaystation':0,'resort':0,'river':0,'sparseresidential':0,'storagetanks':0,'stadium':0}
    val_dict = {'tree':0,'building':0,'airport': 0, 'bareland': 0, 'baseballfield': 0, 'beach': 0, 'bridge': 0, 'center': 0, 'church': 0,
                  'commercial': 0, 'denseresidential': 0, 'desert': 0,
                  'farmland': 0, 'forest': 0, 'industrial': 0, 'meadow': 0, 'mediumresidential': 0, 'mountain': 0,
                  'park': 0, 'school': 0, 'square': 0, 'parking': 0,
                  'playground': 0, 'pond': 0, 'viaduct': 0, 'port': 0, 'railwaystation': 0, 'resort': 0, 'river': 0,
                  'sparseresidential': 0, 'storagetanks': 0, 'stadium': 0}
    test_dict = {'tree':0,'building':0,'airport': 0, 'bareland': 0, 'baseballfield': 0, 'beach': 0, 'bridge': 0, 'center': 0, 'church': 0,
                  'commercial': 0, 'denseresidential': 0, 'desert': 0,
                  'farmland': 0, 'forest': 0, 'industrial': 0, 'meadow': 0, 'mediumresidential': 0, 'mountain': 0,
                  'park': 0, 'school': 0, 'square': 0, 'parking': 0,
                  'playground': 0, 'pond': 0, 'viaduct': 0, 'port': 0, 'railwaystation': 0, 'resort': 0, 'river': 0,
                  'sparseresidential': 0, 'storagetanks': 0, 'stadium': 0}
    train_leng=0
    val_leng = 0
    test_leng = 0
    num=0
    sent_buffer=[]
    for img in data['images']:
        buffer = []
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])  # [[0], [1], [2], [3], [4]]

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)

            sentences = img['sentences']
            if sentences[0]['raw'] == sentences[1]['raw'] == sentences[2]['raw'] == sentences[3]['raw'] == sentences[4]['raw']:
                train_leng = train_leng + 1

        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)

            sentences = img['sentences']
            if sentences[0]['raw'] == sentences[1]['raw'] == sentences[2]['raw'] == sentences[3]['raw'] == sentences[4]['raw']:
            # if sentences[0]['raw'] != sentences[1]['raw']:
                val_leng = val_leng + 1

        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

            sentences = img['sentences']
            if sentences[0]['raw'] == sentences[1]['raw'] == sentences[2]['raw'] == sentences[3]['raw'] == sentences[4]['raw']:
            # if sentences[0]['raw'] != sentences[1]['raw']:
                test_leng = test_leng + 1
            for i in range(5):
                if sentences[i]['raw'] not in buffer:
                    buffer.append(sentences[i]['raw'])
            if len(buffer)==5:
                test_5youxiao_image_paths.append(img['filename'])
        sentences = img['sentences']
        for sent_id in range(5):
            if sentences[sent_id]['raw'] not in sent_buffer:
                sent_buffer.append(sentences[sent_id]['raw'])

        # total_dict = Counter(train_dict) + Counter(val_dict) +Counter(test_dict)
    # for imageclass in classlist:
    #     train_dict[imageclass] = train_dict[imageclass]/len(train_image_paths)
    #     val_dict[imageclass] = val_dict[imageclass] / len(val_image_paths)
    #     test_dict[imageclass] = test_dict[imageclass]/len(test_image_paths)
    #     total_dict[imageclass] = total_dict[imageclass] / (len(train_image_paths)+len(val_image_paths) + len(test_image_paths))

    print(train_leng,val_leng,test_leng)
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    print("find {} training data, {} val data, {} test data".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    print(test_5youxiao_image_paths)
    print(len(test_5youxiao_image_paths))
    print('RSICD中所有不同的语句:',len(sent_buffer))
    # # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}  # word2id
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0
    #
    # # Create a base/root name for all output files
    # base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    #
    # # Save word map to a JSON
    # with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
    #     json.dump(word_map, j)
    # print("{} words write into WORDMAP".format(len(word_map)))
    #
    # # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    # seed(123)
    # for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
    #                                (val_image_paths, val_image_captions, 'VAL'),
    #                                (test_image_paths, test_image_captions, 'TEST')]:
    #
    #     with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
    #         # Make a note of the number of captions we are sampling per image
    #         h.attrs['captions_per_image'] = captions_per_image
    #
    #         # Create dataset inside HDF5 file to store images
    #         images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
    #
    #         print("\nReading %s images and captions, storing to file...\n" % split)
    #
    #         enc_captions = []
    #         caplens = []
    #
    #         for i, path in enumerate(tqdm(impaths)):
    #
    #             # Sample captions
    #             if len(imcaps[i]) < captions_per_image:
    #                 captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
    #             else:
    #                 captions = sample(imcaps[i], k=captions_per_image)
    #
    #             # Sanity check
    #             assert len(captions) == captions_per_image
    #
    #             # Read images
    #             img = imageio.imread(impaths[i])
    #             # img = imread(impaths[i])
    #             if len(img.shape) == 2:
    #                 # gray-scale
    #                 img = img[:, :, np.newaxis]
    #                 img = np.concatenate([img, img, img], axis=2)  # [256, 256, 1+1+1]
    #             img = np.array(Image.fromarray(img).resize((256, 256)))
    #             # img = imresize(img, (256, 256))
    #             img = img.transpose(2, 0, 1)
    #             assert img.shape == (3, 256, 256)
    #             assert np.max(img) <= 255
    #
    #             # Save image to HDF5 file
    #             images[i] = img
    #
    #             for j, c in enumerate(captions):
    #                 # Encode captions
    #                 enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
    #                     word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
    #
    #                 # Find caption lengths
    #                 c_len = len(c) + 2
    #
    #                 enc_captions.append(enc_c)
    #                 caplens.append(c_len)
    #
    #         # Sanity check
    #         assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
    #
    #         # Save encoded captions and their lengths to JSON files
    #         with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(enc_captions, j)
    #
    #         with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(caplens, j)


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k','RSICD'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    classlist = ['tree','building','airport','land','field','beach','bridge','center','church','commercial','residential','desert',
                'farmland','forest','industrial','meadow',#'mediumresidential',
                 'mountain','park','school','square','parking',
                'playground','pond','viaduct','port','railway',
                 'resort','river',#'sparseresidential',
                 'tank','stadium']
    train_dict = {'tree':0,'building':0,'airport':0,'land':0,'field':0,'beach':0,'bridge':0,'center':0,'church':0,'commercial':0,'residential':0,'desert':0,
                'farmland':0,'forest':0,'industrial':0,'meadow':0,#'mediumresidential':0,
                  'mountain':0,'park':0,'school':0,'square':0,'parking':0,
                'playground':0,'pond':0,'viaduct':0,'port':0,'railway':0,
                  'resort':0,'river':0,#'sparseresidential':0,
                  'tank':0,'stadium':0}
    val_dict = {'tree':0,'building':0,'airport':0,'land':0,'field':0,'beach':0,'bridge':0,'center':0,'church':0,'commercial':0,'residential':0,'desert':0,
                'farmland':0,'forest':0,'industrial':0,'meadow':0,#'mediumresidential':0,
                  'mountain':0,'park':0,'school':0,'square':0,'parking':0,
                'playground':0,'pond':0,'viaduct':0,'port':0,'railway':0,
                  'resort':0,'river':0,#'sparseresidential':0,
                  'tank':0,'stadium':0}
    test_dict = {'tree':0,'building':0,'airport':0,'land':0,'field':0,'beach':0,'bridge':0,'center':0,'church':0,'commercial':0,'residential':0,'desert':0,
                'farmland':0,'forest':0,'industrial':0,'meadow':0,#'mediumresidential':0,
                  'mountain':0,'park':0,'school':0,'square':0,'parking':0,
                'playground':0,'pond':0,'viaduct':0,'port':0,'railway':0,
                  'resort':0,'river':0,#'sparseresidential':0,
                  'tank':0,'stadium':0}
    train_leng=0
    val_leng = 0
    test_leng = 0
    num=0
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])  # [[0], [1], [2], [3], [4]]

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval','val','test'}:
            num = num + 1
            if num % 10 == 0:
                val_image_paths.append(path)
                val_image_captions.append(captions)
                for imageclass in classlist:
                    for i in range(5):
                        sentences = img['sentences']
                        sentencesi = sentences[i]
                        # if imageclass in sentencesi['tokens']:
                        if (imageclass + ' ' in sentencesi['raw']) or (imageclass + 's ' in sentencesi['raw']):
                            val_dict[imageclass] = val_dict[imageclass] + 1
                            val_leng = val_leng + 1
            elif num % 10 == 1:
                test_image_paths.append(path)
                test_image_captions.append(captions)
                for imageclass in classlist:
                    for i in range(5):
                        sentences = img['sentences']
                        sentencesi = sentences[i]
                        # if imageclass in sentencesi['tokens']:
                        if (imageclass + ' ' in sentencesi['raw']) or (imageclass + 's ' in sentencesi['raw']):
                            test_dict[imageclass] = test_dict[imageclass] + 1
                            test_leng = test_leng + 1
            else:
                train_image_paths.append(path)
                train_image_captions.append(captions)
                for imageclass in classlist:
                    for i in range(5):
                        sentences = img['sentences']
                        sentencesi = sentences[i]
                        # if imageclass in img['filename']:
                        if (imageclass+' ' in sentencesi['raw']) or (imageclass +'s ' in sentencesi['raw']):
                            train_dict[imageclass] = train_dict[imageclass] +1
                            train_leng = train_leng + 1

        # if img['split'] in {'train', 'restval'}:
        #     train_image_paths.append(path)
        #     train_image_captions.append(captions)
        #     for imageclass in classlist:
        #         for i in range(5):
        #             sentences = img['sentences']
        #             sentencesi = sentences[i]
        #             # if imageclass in img['filename']:
        #             if (imageclass + ' ' in sentencesi['raw']) or (imageclass + 's ' in sentencesi['raw']):
        #                 train_dict[imageclass] = train_dict[imageclass] + 1
        #                 train_leng = train_leng + 1
        # elif img['split'] in {'val'}:
        #     val_image_paths.append(path)
        #     val_image_captions.append(captions)
        #     for imageclass in classlist:
        #         for i in range(5):
        #             sentences = img['sentences']
        #             sentencesi = sentences[i]
        #             # if imageclass in sentencesi['tokens']:
        #             if (imageclass + ' ' in sentencesi['raw']) or (imageclass + 's ' in sentencesi['raw']):
        #                 val_dict[imageclass] = val_dict[imageclass] +1
        #                 val_leng = val_leng + 1
        # elif img['split'] in {'test'}:
        #     test_image_paths.append(path)
        #     test_image_captions.append(captions)
        #     for imageclass in classlist:
        #         for i in range(5):
        #             sentences = img['sentences']
        #             sentencesi = sentences[i]
        #             # if imageclass in sentencesi['tokens']:
        #             if (imageclass + ' ' in sentencesi['raw']) or (imageclass + 's ' in sentencesi['raw']):
        #                 test_dict[imageclass] = test_dict[imageclass] +1
        #                 test_leng = test_leng + 1


        total_dict = Counter(train_dict) + Counter(val_dict) +Counter(test_dict)
    for imageclass in classlist:
        train_dict[imageclass] = train_dict[imageclass]/(5*len(train_image_paths))
        val_dict[imageclass] = val_dict[imageclass] / (5*len(val_image_paths))
        test_dict[imageclass] = test_dict[imageclass]/(5*len(test_image_paths))
        # total_dict[imageclass] = total_dict[imageclass] / (len(train_image_paths)+len(val_image_paths) + len(test_image_paths))

    draw_from_dict([train_dict,val_dict,test_dict], len(classlist), 1)
    print('train_dict:\n', train_dict)
    print('val_dict:\n', val_dict)
    print('test_dict:\n', test_dict)
    print('total_dict:\n', total_dict)
    print(train_leng,val_leng,test_leng)
    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    print("find {} training data, {} val data, {} test data".format(len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    # # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}  # word2id
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
    print("{} words write into WORDMAP".format(len(word_map)))

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imageio.imread(impaths[i])
                # img = imread(impaths[i])
                if len(img.shape) == 2:
                    # gray-scale
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)  # [256, 256, 1+1+1]
                img = np.array(Image.fromarray(img).resize((256, 256)))
                # img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(checkpoint_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset #FIXME:change data_name to decoder_mode
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch  #FIXME:change bleu4 to metrics
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'final_args': final_args}
    filename = 'checkpoint_' + checkpoint_name +'.pth.tar'

    filepath = os.path.join('./models_checkpoint_GRSL/', filename)  # 最终参数模型
    torch.save(state, filepath)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join('./models_checkpoint_GRSL/', 'BEST_' + filename))

    torch.save(state, os.path.join('./models_checkpoint_GRSL/', 'epoch_'+str(epoch) + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def convert2words(sequences, rev_word_map):
    for l1 in sequences:
        caption = ""
        for l2 in l1:
            caption += rev_word_map[l2]
            caption += " "
        print(caption)
