import torch
# tranforms


class Normalize(object):
    """Normalizes keypoints.
    """
    def __call__(self, sample, verbose=False):

        image, key_pts = sample['image'], sample['keypoints']

        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################
        image = image / 255
        key_pts = key_pts / 48
        key_pts = key_pts - 1
        if verbose:
            print("image after normalization: ")
            print(image)
            print("key_pts after notmalization: ")
            print(key_pts)
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}
