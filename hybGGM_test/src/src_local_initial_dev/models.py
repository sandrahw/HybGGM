
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import data
import train
from torch.utils.tensorboard import SummaryWriter

class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of two Conv2D layers,
    each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # Conv -> BatchNorm -> ReLU
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a ConvBlock followed by MaxPooling.
    The output of the ConvBlock is stored for the skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)  # Apply the convolutional block
        x_pool = self.pool(x_conv) # Apply max pooling
        return x_conv, x_pool # Return both for the skip connection

class DecoderBlock(nn.Module):
    """
    Decoder block consisting of an upsampling (ConvTranspose2d) and a ConvBlock.
    It takes the skip connection from the corresponding encoder block.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)# Double input channels to account for skip connection

    def forward(self, x, skip):
        x = self.up(x)
        # Center crop the skip connection tensor to match the size of the upsampled tensor
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)  # Concatenate along channel axis
        x = self.conv(x)
        return x

class UNet6(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet6, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 1024)#new layer
        self.encoder6 = EncoderBlock(1024, 2048)#new layer

        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(2048, 4096)

        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(4096, 2048)
        self.decoder2 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder5 = DecoderBlock(256, 128)
        self.decoder6 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block
        x3, p3 = self.encoder3(p2) # Third block
        x4, p4 = self.encoder4(p3) # Fourth block
        x5, p5 = self.encoder5(p4) # Fifth block
        x6, p6 = self.encoder6(p5) # Sixth block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p6)

        # Decoder
        d1 = self.decoder1(bottleneck, x6)  # Upsample from bottleneck and add skip connection from encoder4
        d2 = self.decoder2(d1, x5)          # Continue with decoder and corresponding skip connection from encoder
        d3 = self.decoder3(d2, x4)          # Continue with decoder and corresponding skip connection from encoder
        d4 = self.decoder4(d3, x3)          # Continue with decoder and corresponding skip connection from encoder
        d5 = self.decoder5(d4, x2)          # Continue with decoder and corresponding skip connection from encoder
        d6 = self.decoder6(d5, x1)          # Continue with decoder and corresponding skip connection from encoder
       
        # Final output layer
        output = self.final_conv(d6)  
        return output
    
class UNet4(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet4, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)


        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(512, 1024)
 
        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block
        x3, p3 = self.encoder3(p2) # Third block
        x4, p4 = self.encoder4(p3) # Fourth block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p4)

        # Decoder
        d1 = self.decoder1(bottleneck, x4)  # Upsample from bottleneck and add skip connection from encoder4
        d2 = self.decoder2(d1, x3)          # Continue with decoder and corresponding skip connection from encoder3
        d3 = self.decoder3(d2, x2)          # Continue with decoder and corresponding skip connection from encoder2
        d4 = self.decoder4(d3, x1)          # Continue with decoder and corresponding skip connection from encoder1

        # Final output layer
        output = self.final_conv(d4)        # Reduce to the number of output channels (e.g., 1 for groundwater head)
        return output
    
class UNet2(nn.Module):
    """
    The complete U-Net architecture with an encoder-decoder structure.
    It uses skip connections from the encoder to the decoder.
    """
    def __init__(self, input_channels, output_channels):
        super(UNet2, self).__init__()
        
        # Encoder: Downsampling path
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)

        # Bottleneck layer (middle part of the U-Net)
        self.bottleneck = ConvBlock(128, 256)

        # Decoder: Upsampling path
        self.decoder1 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)

        # Final output layer to reduce to the number of desired output channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)  # First block
        x2, p2 = self.encoder2(p1) # Second block

        # Bottleneck (middle)
        bottleneck = self.bottleneck(p2)

        # Decoder  # Continue with decoder and corresponding skip connection from encoder3
        d1 = self.decoder1(bottleneck, x2)          # Continue with decoder and corresponding skip connection from encoder2
        d2 = self.decoder2(d1, x1)          # Continue with decoder and corresponding skip connection from encoder1

        # Final output layer
        output = self.final_conv(d2)        # Reduce to the number of output channels (e.g., 1 for groundwater head)
        return output

class ConvExample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvExample, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)  # First Conv Layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Second Conv Layer
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # Learnable transposed convolution for upscaling
        self.transconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Upscale by 2x
        self.transconv2 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1)  # Upscale by 2x again

        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Pass input through convolutional layers and pooling
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        # print('x1', x.shape)
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        # print('x2', x.shape)
        
        # Upscale using transposed convolutions
        x = self.relu(self.transconv1(x))  # First upscaling
        # print('x_transconv1', x.shape)
        x = self.transconv2(x)  # Second upscaling to original size
        # print('x_final', x.shape)

        return x

def hyperparam_tuning_training(X_norm, y_norm, testSize, trainSize, def_epochs, lr_rate, batchSize, mt, save_path):
    log_directory = 'results/testing/%s_%s_%s_%s' %(mt, def_epochs, lr_rate ,batchSize)
    log_dir_fig = 'results/testing/%s_%s_%s_%s/plots' %(mt, def_epochs, lr_rate ,batchSize)

    'check if testing prediction files are already there and if so stop the process'
    if os.path.exists(log_directory + '/y_pred_denorm.npy'):
        print('Hyperparameter tuning already done and prediction files are there')
        exit()
        return

    train_loader, validation_loader, test_loader = data.train_val_test_split(testSize, trainSize, X_norm, y_norm, batchSize, save_path)
    # print('Data preparation finished')
    #create folder in case not there yet
    if not os.path.exists(log_directory):
        os.makedirs(log_directory) 
    if not os.path.exists(log_dir_fig):
        os.makedirs(log_dir_fig)
    # print('starting training and saving logs')

    if mt == 'UNet6':
        writer = SummaryWriter(log_dir=log_directory)
        # print('UNet6, model loading')
        model = UNet6(input_channels=21, output_channels=1)
        # print('training')
        train.train_model(model, train_loader, validation_loader, lr_rate, def_epochs,  log_directory, writer)
        # print('testing')
        train.test_model(model, test_loader, writer)
        del model
    if mt == 'UNet4':
        writer = SummaryWriter(log_dir=log_directory)
        # print('UNet4, model loading')
        model = UNet4(input_channels=21, output_channels=1)
        # print('training')
        train.train_model(model, train_loader, validation_loader, lr_rate, def_epochs, log_directory, writer)
        # print('testing')
        train.test_model(model, test_loader, writer)
        del model
    if mt == 'UNet2':
        writer = SummaryWriter(log_dir=log_directory)
        # print('UNet2, model loading')
        model = UNet2(input_channels=21, output_channels=1)
        # print('training')
        train.train_model(model, train_loader, validation_loader, lr_rate, def_epochs, log_directory, writer)
        # print('testing')
        train.test_model(model, test_loader, writer)
        del model
    if mt == 'ConvExample':
        writer = SummaryWriter(log_dir=log_directory)
        # print('UNet2, model loading')
        model = ConvExample(input_channels=21, output_channels=1)
        # print('training')
        train.train_model(model, train_loader, validation_loader, lr_rate, def_epochs, log_directory, writer)
        # print('testing')
        train.test_model(model, test_loader, writer)
        del model

    print('done with hyperparameter tuning training')




def run_model_on_full_data(model, data_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inp, tar, mask in data_loader:
            inp = inp.float()
            tar = tar.float()  
            outputs = model(inp)
            all_outputs.append(outputs.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs


def hyperparam_tuning_prediction(epochs, learning_rate, batch_size, model_type, save_path):
    X_test = np.load('%s/X_test.npy' %save_path)
    mask_test = np.load('%s/mask_test.npy'%save_path)
    y_test = np.load('%s/y_test.npy'%save_path)
    # print(epochs, learning_rate, batch_size, model_type)
    # for batchSize in batch_size:
    #     print('Batch size: %s' %batchSize)
    #     for def_epochs in epochs:
    #         for lr_rate in learning_rate:
    #             for mt in model_type:
    print('Model type: %s, Epochs: %s, Learning rate: %s, Batch size: %s' %(model_type, epochs, learning_rate, batch_size))
    log_directory = 'results/testing/%s_%s_%s_%s' %(model_type, epochs, learning_rate ,batch_size)

    test_loader = data.DataLoader(data.CustomDataset(X_test, y_test, mask_test), batch_size=batch_size, shuffle=False)

    #load the model:
    if model_type == 'UNet2':
        model_reload = UNet2(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
        y_pred_raw = run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd
    if model_type == 'UNet4':
        model_reload = UNet4(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
        y_pred_raw= run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd
    if model_type == 'UNet6':   
        model_reload = UNet6(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
        y_pred_raw = run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd
    if model_type == 'ConvExample':   
        model_reload = ConvExample(input_channels=21, output_channels=1)
        model_reload.load_state_dict(torch.load(os.path.join(log_directory, 'best_model.pth')))
        y_pred_raw = run_model_on_full_data(model_reload, test_loader) #this is now the delta wtd

    out_var_mean = np.load('%s/out_var_mean.npy'%save_path)
    out_var_std = np.load('%s/out_var_std.npy'%save_path)
    y_pred_denorm = y_pred_raw*out_var_std[0] + out_var_mean[0] #denormalise the predicted delta wtd

    np.save('%s/y_pred_denorm.npy' %log_directory, y_pred_denorm)
    np.save('%s/y_pred_raw.npy' %log_directory, y_pred_raw)
    print('Hyperparameter tuning prediction finished')	