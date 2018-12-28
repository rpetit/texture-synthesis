import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms

import copy


IMAGENET_MEAN = [0.485, 0.456, 0.406]
LOSS_SCALING = 1e9


def pre_processing(image, image_size, device):
    pre_process = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    tensor = pre_process(image).unsqueeze(0)  # fake batch dimension to fit the network's input dimensions
    return tensor.to(device, torch.float)


def post_processing(tensor):
    post_process = transforms.Compose([transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
                                       transforms.ToPILImage()])
    tensor_copy = tensor.cpu().clone().squeeze(0)
    return post_process(tensor_copy)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # reshape the mean as : [C x 1 x 1] so that it is compatible with image tensors of shape [B x C x H x W]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def gram_matrix(input):
    a, b, c, d = input.size() # a: batch size (1), b: number of feature maps, (c,d): feature maps' dimensions
    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())  # compute the gram product

    # divide by the number of elements in each feature maps
    return G.div(a * b * c * d)


class TextureLoss(nn.Module):

    def __init__(self, target_feature):
        super(TextureLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = LOSS_SCALING * F.mse_loss(G, self.target)

        return input


def get_texture_model_and_losses(cnn, texture_img, texture_layers, device):
    cnn = copy.deepcopy(cnn)
    texture_losses = []

    cnn_normalization_mean = torch.tensor(IMAGENET_MEAN).to(device)
    cnn_normalization_std = torch.tensor([1.0, 1.0, 1.0]).to(device)

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    j = 0  # increment every time we see a pool

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # in-place version clashes with TextureLoss ?
        elif isinstance(layer, nn.MaxPool2d):
            j += 1
            # replace every max-pooling by an average-pooling
            layer = nn.AvgPool2d(layer.kernel_size, stride=layer.stride,
                                 padding=layer.padding, ceil_mode=layer.ceil_mode)
            name = 'pool_{}'.format(j)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        
        if name in texture_layers:
            target_feature = model(texture_img).detach()
            texture_loss = TextureLoss(target_feature)
            model.add_module("texture_loss_{}".format(i + j), texture_loss)
            texture_losses.append(texture_loss)

    # trim off the layers after the last texture loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], TextureLoss):
            break

    model = model[:(i + 1)]

    return model, texture_losses


def run_texture_synthesis(cnn, texture_image, image_size, num_steps, device, verbose=True):

    texture_img = pre_processing(texture_image, image_size, device)
    synthesized_img = torch.randn(texture_img.data.size(), device=device)
    rescale = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    synthesized_img = rescale(synthesized_img)
    
    if verbose:
        print('Building the texture model..\n')

    texture_layers = ['pool_4', 'pool_3', 'pool_2', 'pool_1', 'conv_1']

    model, texture_losses = get_texture_model_and_losses(cnn, texture_img, texture_layers, device)
    
    optimizer = optim.LBFGS([synthesized_img.requires_grad_()])

    if verbose:
        print('Optimizing..\n')
        
    run = [0]  # weird local variable behaviour
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            model(synthesized_img)
            texture_score = 0

            for tl in texture_losses:
                texture_score += tl.loss

            loss = texture_score
            loss.backward()

            run[0] += 1
            
            if run[0] % (500 if torch.cuda.is_available() else 100) == 0:
                if verbose:
                    print("run {}".format(run[0]))
                    print('loss : {:.2e}\n'.format(texture_score.item()))

            return texture_score

        optimizer.step(closure)

    return post_processing(synthesized_img)


import numpy as np
import scipy.interpolate


def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram

    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0] * n
    start = 0  # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i - 1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start + 1 + i) / 2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start + 1 + n) / 2.0;
    return np.asarray(Rx) / float(len(Rx))


def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges, bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r > cum_values.max()] = cum_values.max()
        matched_image = inv_cdf(r).reshape(org_image.shape)
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:, :, i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges, bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:, :, i].ravel()))
            r[r > cum_values.max()] = cum_values.max()
            matched_image[:, :, i] = inv_cdf(r).reshape(org_image[:, :, i].shape)

    return matched_image