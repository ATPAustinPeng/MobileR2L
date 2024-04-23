import torch.nn as nn
from .channel_selection import channel_selection

class ResnetBlock(nn.Module):
    """
        Define a Resnet block.
        Refer to "https://github.com/ermongroup/ncsn/blob/master/models/pix2pix.py"
    """
    def __init__(
        self, 
        dim,
        cfg,
        kernel_size=1, 
        padding_type='zero',
        norm_layer=nn.BatchNorm2d, 
        use_dropout=False, 
        use_bias=False, #remove bias to help ease of pruning
        act=None
    ):
        
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, cfg, kernel_size, padding_type,
            norm_layer, use_dropout, use_bias, act
        )

    def build_conv_block(
        self, 
        dim,
        cfg,
        kernel_size, 
        padding_type, 
        norm_layer, 
        use_dropout, 
        use_bias, 
        act=nn.GELU()
    ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        conv_block = []

        ########## ADDING LAYERS FOR CHANNEL-WISE STRUCTURED PRUNING ##########
        conv_block += [norm_layer(dim, momentum=0.1)]
        conv_block += [channel_selection(dim)]
        #######################################################################

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(cfg[0], cfg[1], kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(cfg[1], momentum=0.1)]
        if act:
            conv_block += [act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(cfg[1], dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        # if norm_layer:
        #     conv_block += [norm_layer(dim, momentum=0.1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections

        return out
    
def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'gelu':
        func = nn.GELU()
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func

class R2LCFG(nn.Module):
    def __init__(
        self,
        args,
        input_dim, 
        output_dim,
        cfg=None
    ):
        super(R2LCFG, self).__init__()

        self.args = args
        self.input_dim = input_dim
        D, W = args.netdepth, args.netwidth
        Ws = [W] * (D-1) + [3]
        act = get_activation(args.activation_fn)
        self.head = nn.Sequential(
            *[nn.Conv2d(input_dim, Ws[0], 1), act]) if act else nn.Sequential(*[nn.Conv2d(input_dim, Ws[0], 1)]
        )

        n_block = (D - 2) // 2 # 2 layers per resblock
        n_conv = args.num_conv_layers # 2 conv layers in each sr
        n_up_block = args.num_sr_blocks # 3 sr blocls
        kernels = args.sr_kernel

        if cfg is None:
            # head doesn't have batchnorm2d
            # tail in MobileR2L usecase only has 3 sr blocks * 2 conv blocks
            # cfg = [[body], [tail]]
            cfg = [[256, 256] * n_block]
            cfg += [[kernels[n], kernels[n]] for n in range(n_up_block) for _ in range(n_conv)] # Note: must iterate from outer block -> inner block (or shapes don't match)
            cfg = [item for sub_list in cfg for item in sub_list]
        elif cfg is not None:
            tmp = [[kernels[n], kernels[n]] for n in range(n_up_block) for _ in range(n_conv)]
            tmp = [item for sub_list in tmp for item in sub_list]
            cfg += tmp
        cfg_count = 0

        body = []
        for _ in range(n_block):
            body += [ResnetBlock(W, cfg[2*cfg_count:2*(cfg_count+1)], act=act)]
            cfg_count += 1
        # body = [ResnetBlock(W, cfg[2*n:2*(n+1)] act=act) for n in range(n_block)]
        self.body = nn.Sequential(*body)
            
        if args.use_sr_module:
            # n_conv = args.num_conv_layers # 2 conv layers in each sr
            # n_up_block = args.num_sr_blocks # 3 sr blocls
            # kernels = args.sr_kernel

            up_blocks = []
            for i in range(n_up_block - 1):
                in_dim = Ws[-2] if not i else kernels[i]
                up_blocks += [nn.ConvTranspose2d(in_dim, kernels[i], 4, stride=2, padding=1)]

                for _ in range(n_conv):
                    up_blocks += [ResnetBlock(kernels[i], cfg[2*cfg_count:2*(cfg_count+1)], act=act)]
                    cfg_count += 1
                # up_blocks += [ResnetBlock(kernels[i], cfg[2*(n+n_block+i):2*(n+1+n_block+i)], act=act) for n in range(n_conv)]

            # hard-coded up-sampling factors
            # 12x for colmap
            if args.dataset_type == 'Colmap':
                k, s, p = 3, 3, 0
            elif args.dataset_type == 'Blender': # 8x for blender
                k, s, p = 4, 2, 1
            else:
                raise ValueError(f'Undefined dataset type: {args.dataset_type}.')            
        
            up_blocks += [nn.ConvTranspose2d(kernels[1], kernels[-1], k, stride=s, padding=p)]
            for _ in range(n_conv):
                up_blocks += [ResnetBlock(kernels[-1], cfg[2*cfg_count:2*(cfg_count+1)], act=act)]
                cfg_count += 1
            # up_blocks += [ResnetBlock(kernels[-1], act=act)  for _ in range(n_conv)]
            up_blocks += [nn.Conv2d(kernels[-1], output_dim, 1), nn.Sigmoid()]
            self.tail = nn.Sequential(*up_blocks)
        else:
            self.tail = nn.Sequential(*[nn.Conv2d(W, output_dim, 1), nn.Sigmoid()])
    
    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        return self.tail(x)