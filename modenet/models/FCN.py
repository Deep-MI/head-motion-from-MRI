import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN2D(nn.Module):
    def __init__(self, input_shape):
        super(FCN2D, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear((input_shape[-2]+2)*(input_shape[-1]+2),1),
            #nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        logits = self.fcn(x)
        out = self.linear(logits.flatten(1))
        return out

class FCN3D(nn.Module):
    def __init__(self, input_shape):
        super(FCN3D, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv3d(16, 8, 3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear((input_shape[-3]+2)*(input_shape[-2]+2)*(input_shape[-1]+2),1),
            #nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        logits = self.fcn(x)
        #import ipdb; ipdb.set_trace()
        out = self.linear(logits.flatten(1))
        return out

class custom_SFCN(nn.Module):
    def __init__(self, sequence='T1', channel_number=[1, 32, 64, 128, 256, 256, 64], output_dim=40, dropout=0.5):
        super(custom_SFCN, self).__init__()

        assert(len(channel_number) == 7), 'make sure this channel number is suitable for your MRI dimension'

        self.output_dim = output_dim
        self.feature_extractor = nn.Sequential()
        for i, in_channel in enumerate(channel_number):
            #print(channel_number[i], channel_number[i+1])
            if i+2 == len(channel_number): # last loop
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(channel_number[i],
                                                                  channel_number[i+1],
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
                print('last conv_%d' % i, channel_number[i], channel_number[i+1])
                break
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(channel_number[i],
                                                                  channel_number[i+1],
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
                print('conv_%d' % i, channel_number[i], channel_number[i+1])
            
        self.classifier = nn.Sequential()
        
        if sequence == 'T1' or sequence == 'T2' or sequence == 'background': # TODO: calc dynamically
            avg_shape = [7, 10, 10] 
        elif sequence == 'FLAIR':
            avg_shape = [5, 8, 8]
        else:
            raise ValueError('unknown sequence indetifier')

        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout > 0:
            self.classifier.add_module('dropout', nn.Dropout3d(dropout))
        self.classifier.add_module('conv_%d' % (i+1),
                                   nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1))

        self.classifier.apply(self.init_weights)
        self.feature_extractor.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            #m.bias.data.fill_(0.01)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        #print('input shape', x.shape)
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)#.squeeze()
        x = x.reshape(x.shape[:2])
        #print('output shape', x.shape)

        assert(len(x.shape) == 2), 'output dimension is not 2'
        
        if self.output_dim == 1:
            return x
        else:
            return F.log_softmax(x, dim=1)

# from Peng et al., Accurate brain age prediction with lightweight deep neural networks
class SFCN(nn.Module):
    def __init__(self, sequence='T1', channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)

        self.output_dim = output_dim

        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        
        if sequence == 'T1' or sequence == 'T2' or sequence == 'background': # TODO: calc dynamically
            avg_shape = [7, 10, 10] 
        elif sequence == 'FLAIR':
            avg_shape = [5, 8, 8]
        else:
            raise ValueError('unknown sequence indetifier')
        #avg_shape = [4,4,4]

        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):

        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        if self.output_dim != 1:
            x = F.log_softmax(x, dim=1)

        out = x.reshape(x.shape[:2])
        return out