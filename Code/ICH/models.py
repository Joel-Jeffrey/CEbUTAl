def InceptionV3(num_classes):
    model = inception_v3(pretrained=False, aux_logits=False)
    
    model.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def DenseNet264(num_class):
    model = densenet201(pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_class)
    
    return model

def MobileNet(num_classes):
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    model.features[0] = torch.nn.Conv2d(1, 32, kernel_size=8, stride=1, bias=False)

    return model

def ResNet34(num_classes):
    model = resnet34(weights=None, num_classes=num_classes)  # Load ResNet18 without pre-trained weights
    
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    return model

def SqueezeNet(num_classes):
    model = squeezenet1_0(weights=None, num_classes=num_classes)
    model.features[0] = torch.nn.Conv2d(1, 96, kernel_size=7, stride=2)  # Adjust the first layer to accept single-channel input

    return model
