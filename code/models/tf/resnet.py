from .resnet_model import Model

class Resnet18(Model):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__(resnet_size=18,
                                      bottleneck=False,
                                      num_classes=num_classes,
                                      num_filters=64,
                                      kernel_size=7,
                                      conv_stride=2,
                                      first_pool_size=3,
                                      first_pool_stride=2,
                                      block_sizes=[2,2,2,2],
                                      block_strides=[1,2,2,2],
                                      final_size=512,
                                      resnet_version=1)