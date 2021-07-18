import numpy as np
from numpy import inf
import librosa
import torch

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D


def real_imag_shrink(F,dim='new'):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim =='new':
        F_shrink = F[:,:,0] + F[:,:,1]*1j
    if dim =='same':
        F_shrink = F[:,::2] + F[:,1::2]*1j
    return F_shrink

def istft2librosa(S, n_fft, hop_length, win_length):
    s = real_imag_shrink(S)
    s = librosa.istft(s, hop_length, win_length)
    return s


def fast_stft(data, n_fft, hop_length, win_length):
    # directly transform the wav to the input
    return real_imag_expand(librosa.stft(data, n_fft, hop_length, win_length))

def generate_cRM(Y,S):
    '''
    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    '''
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:,:,0],S[:,:,0])+np.multiply(Y[:,:,1],S[:,:,1])
    square_real = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_real = np.divide(M_real,square_real+epsilon)
    M[:,:,0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:,:,0],S[:,:,1])-np.multiply(Y[:,:,1],S[:,:,0])
    square_img = np.square(Y[:,:,0])+np.square(Y[:,:,1])
    M_img = np.divide(M_img,square_img+epsilon)
    M[:,:,1] = M_img
    return M


def cRM_tanh_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = 1-np.exp(-C*M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1+np.exp(-C*M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator,denominator)

    return crm


def cRM_tanh_recover(O,K=10,C=0.1):
    '''
    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O + 1e-8
    denominator = K+O + 1e-8
    M = -np.multiply((1.0/C),np.log(np.divide(numerator,denominator)))

    return M


def cRM_sigmoid_compress(M, a=0.1, b=0):
    """sigmoid compression"""
    return 1. / (1. + np.exp(-a * M + b))


def cRM_sigmoid_recover(O, a=0.1, b=0):
    """inverse sigmoid"""
    return 1. / a * (np.log(O / (1 - O + 1e-8) + 1e-10) + b)


def fast_cRM(Fclean,Fmix,K=10,C=0.1):
    '''
    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_tanh_compress(M,K,C)
    return crm


def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_tanh_recover(crm,K,C)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S


def fast_cRM_sigmoid(Fclean,Fmix):
    '''
    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_sigmoid_compress(M)
    return crm


def fast_icRM_sigmoid(Y,crm):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_sigmoid_recover(crm)
    S = np.zeros(np.shape(M))
    S[:,:,0] = np.multiply(M[:,:,0],Y[:,:,0])-np.multiply(M[:,:,1],Y[:,:,1])
    S[:,:,1] = np.multiply(M[:,:,0],Y[:,:,1])+np.multiply(M[:,:,1],Y[:,:,0])
    return S


def batch_fast_icRM_sigmoid(Y, crm, a=0.1, b=0):
    """

    :param Y: (B, 2, F, T)
    :param crm: (B, 2, F, T)
    :param a:
    :param b:
    :return:
    """
    M = 1. / a * (torch.log(crm / (1 - crm + 1e-8) + 1e-10) + b)
    r = M[:, 0, :, :] * Y[:, 0, :, :] - M[:, 1, :, :] * Y[:, 1, :, :]
    i = M[:, 0, :, :] * Y[:, 1, :, :] + M[:, 1, :, :] * Y[:, 0, :, :]
    rec = torch.stack([r, i], dim=1)
    return rec

def extrapolate_audio(audio, n):
    orig_size = audio.shape[0]
    m = np.ceil(orig_size / n) 
    new_size = int(m*n)
    extra_values = int(n - (new_size - orig_size)) 
    new_value = np.mean(audio[extra_values:])
    new_audio = np.empty(new_size)
    new_audio[:orig_size] = audio
    new_audio[orig_size:] = np.full(new_size - orig_size, new_value)
    return new_audio