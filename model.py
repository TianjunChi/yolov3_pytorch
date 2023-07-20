import torch
import torch.nn as nn

class ResBlock:
            def __init__(self,n_channel):
                super(ResBlock,self).__init__()
                self.conv1 = nn.Conv2d( #2n = 64 , n = 32
                    in_channels= 2*n_channel,
                    out_channels=n_channel,
                    kernel_size=(1,1),
                    padding=1
                )
                self.conv2 = nn.Conv2d(
                    in_channels=n_channel,
                    out_channels=2*n_channel,
                    kernel_size=(3,3),
                    stride=1,
                    padding = 1
                )

            def forward(self,x):
                # route 1
                x1 = x

                # route 2
                x2 = self.conv1(x)
                x2 = self.conv2(x2)
                x = x1 + x2
                return x

class ResOpt: # an operator that includes many ResBlock
            def __init__(self,n_ResBlock,n_channel):
                super(ResOpt,self).__init__()
                self.resBlocks = nn.ModuleList()
                self.n_ResBlock = n_ResBlock
                for i in range(0,n_ResBlock):
                    self.resBlocks.append(
                        ResBlock(n_channel=n_channel)
                    )
                
            def forward(self,x):
                for i in range(self.n_ResBlock):
                    x = self.n_ResBlocks[i](x)
                return x

class Darknet: # backbone: Darknet-53
    def __init__(self):
        super(Darknet, self).__init__()
        self.conv1 = nn.Conv2d( # 256*256*3 -> 256*256*32
            in_channels=3,
            out_channels=32,
            kernel_size=(3,3),
            stride = 1,
            padding=1
            )
        self.conv2 = nn.Conv2d( # 256*256*32 -> 128*128*64
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        

            
        self.res2 = ResOpt(1,32) # [1st ResOpt(1)] ResBlock num:1 n_channel(in ResBlock):32 
        
        
        self.conv3 = nn.Conv2d(
            in_channels= 64,
            out_channels=128,
            kernel_size=(3,3),
            stride=2,
            padding =1
        )
        self.res3 = ResOpt(2,64) # [2nd ResOpt(2)] ResBlock num:2 n_channel(in ResBlock):64 

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        self.res4 = ResOpt(8,128) # [3rd ResOpt(8)] ResBlock num:8 n_channel(in ResBlock):128

        self.conv5 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        self.res5 = ResOpt(8,256) # [4rd ResOpt(8)] ResBlock num:8 n_channel(in ResBlock):256

        self.conv6 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        self.res6 = ResOpt(4,512) # [4rd ResOpt(8)] ResBlock num:8 n_channel(in ResBlock):256



    def forward(self,x):
        # 3*3 CONV [256*256*3]
        x = self.conv1(x) 

        # ResOperator(1) [256*256*32]
        x = self.conv2(x)
        x = self.res2(x)
        
        # ResOperator(2) [128,128,64]
        x = self.conv3(x)
        x = self.res3(x)

        # ResOperator(8) [64,64,128]
        x = self.conv4(x)
        x = self.res4(x)

        out_large = x
        # ResOperator(8) [32,32,256]
        x = self.conv5(x)
        x = self.res5(x)

        out_middle = x
        # ResOperator(4) [16,16,512]
        x = self.conv6(x)
        x = self.res6(x)




class yoloBlock:
     def __init__(self):
        super(yoloBlock, self).__init__()
        self.conv1 = nn.Conv2d( # 256*256*3 -> 256*256*32
            in_channels=3,
            out_channels=32,
            kernel_size=(3,3),
            stride = 1,
            padding=1
            )
        self.conv2 = nn.Conv2d( # 256*256*32 -> 128*128*64
            in_channels=32,
            out_channels=64,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )