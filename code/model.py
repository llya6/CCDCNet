
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



from segmentation_models_pytorch.losses import DiceLoss
from config import *
class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, images):
        x = self.conv(images)
        p = self.pool(x)

        return x, p



# %%
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = conv(out_channels * 2, out_channels)

    def forward(self, images, prev):
        x = self.upconv(images)
        x = torch.cat([x, prev], axis=1)
        x = self.conv(x)

        return x

# %% [markdown]
# Burada kafa karıştıran bölüm fonksiyonlar arasında bağlantı olmamasına rağmen fonksiyonların bağlı olması olabilir. Bunu sağlayanın class'ın başlangıcında yazdığımız nn.Module'dür. 
# 
# nn.Module forward fonksiyonunu __init__ ile bağlayıp bir mimarı oluşturuyor...

# %%
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder(3, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        self.b = conv(512, 1024)

        self.d1 = decoder(1024, 512)
        self.d2 = decoder(512, 256)
        self.d3 = decoder(256, 128)
        self.d4 = decoder(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, images):
        x1, p1 = self.e1(images)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        b = self.b(p4)
        
        d1 = self.d1(b, x4)
        d2 = self.d2(d1, x3)
        d3 = self.d3(d2, x2)
        d4 = self.d4(d3, x1)

        output_mask = self.output(d4)

        return output_mask  
    

class SegmentationModel(nn.Module):  
  
    def __init__(self):    
        super(SegmentationModel,self).__init__()  

        self.arc = smp.Unet(  
            encoder_name=ENCODER,   
            encoder_weights=WEIGHTS,    
            in_channels=5,              
            classes=1,       
            encoder_depth=5,
            # upsampling = 8,
            activation=None         
            
        )  

    def forward(self, images, masks=None):   
        logits = self.arc(images)  
        if masks != None:   
            loss1 = DiceLoss(mode='binary')(logits, masks)  
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)  
  
            return logits, loss1, loss2  

        return logits


