import torch
from src.banknote_classifier.models import BanknoteClassifierVGG


def test_vgg16_instantiation():
    """VGG16 應能成功建立，且為 torch.nn.Module 的子類別"""
    model = BanknoteClassifierVGG()
    assert isinstance(model, torch.nn.Module)


def test_vgg16_output_shape():
    """VGG16 在 CPU 上的 forward pass 輸出 shape 應符合預期"""
    model = BanknoteClassifierVGG()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[0] == 1  # batch size


def test_vgg16_output_is_tensor():
    """VGG16 的輸出應為 torch.Tensor"""
    model = BanknoteClassifierVGG()
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, torch.Tensor)