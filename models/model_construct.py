from models.resnet import resnet


def Model_Construct(args):
    if args.arch.find('resnet') != -1:
        model = resnet(args)
        return model
    else:
        raise ValueError('The required model does not exist!')
