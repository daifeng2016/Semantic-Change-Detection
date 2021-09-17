from .decoder import FCN,PSPNet
def get_model(model_head, backbone, pretrained, nclass, lightweight):
    if model_head == "fcn":
        model = FCN(backbone, pretrained, nclass, lightweight)
    elif model_head == "pspnet":
        model = PSPNet(backbone, pretrained, nclass, lightweight)
    # elif model == "deeplabv3plus":
    #     model = DeepLabV3Plus(backbone, pretrained, nclass, lightweight)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model_head)

    params_num = sum(p.numel() for p in model.parameters())
    print("\nParams: %.1fM" % (params_num / 1e6))

    return model