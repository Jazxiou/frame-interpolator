import io
import os
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import FrameIntUtils as fiutils


class TestFrameIntUtils(unittest.TestCase):
    def setUp(self):
        """
        Runs before every test. Creates a dummy image to test with.
        """
        self.test_img_name = "test_dummy.png"
        img = Image.new("RGB", (10, 10), color="red")  # A 10x10 red image
        img.save(self.test_img_name)

        self.df = pd.DataFrame({
            "Image 1 (Input)": [self.test_img_name],
            "Image 3 (Input)": [self.test_img_name],
            "Image 2 (Output)": [self.test_img_name],
        })

    def tearDown(self):
        """
        Runs after every test. Cleans up the dummy image.
        """
        if os.path.exists(self.test_img_name):
            os.remove(self.test_img_name)

    def test_get_batch_accuracy(self):
        """
        Test if the math for get_batch_accuracy is correct.
        """

        def metric(x, y):
            return torch.tensor(0.8)

        output = torch.randn(10, 3, 256, 256)  # 10 random images
        target = torch.randn(10, 3, 256, 256)
        N = 100

        result = fiutils.get_batch_accuracy(output, target, metric, N)
        self.assertAlmostEqual(result, 0.08)  # Expected (0.8 * 10) / 100 = 0.08

    def test_dataset_structure(self):
        """
        Test if the FrameInterpolation class returns tensors of the right shape.
        """
        dataset = fiutils.FrameInterpolation(self.df, augment=False)
        (t1, t3), t2 = dataset[0]

        # Check types
        self.assertIsInstance(t1, torch.Tensor)
        self.assertIsInstance(t2, torch.Tensor)
        self.assertIsInstance(t3, torch.Tensor)

        # Check structure (C, H, W)
        self.assertEqual(t1.shape, (3, 10, 10))

    def test_augmentation_enabled(self):
        """
        Test if turning on augmentation runs without crashing.
        """
        dataset = fiutils.FrameInterpolation(self.df, augment=True)
        (t1, t3), t2 = dataset[0]
        self.assertEqual(t1.shape, (3, 10, 10))

    def test_process_widget(self):
        """
        Test converting raw bytes to tensor.
        """
        img = Image.new("RGB", (100, 100), color="blue")  # Simulate file upload
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")

        class Widget:
            value = [{"content": img_byte_arr.getvalue()}]

        widget = Widget()
        device = torch.device("cpu")

        tensor = fiutils.process_widget(widget, target_size=(50, 50), device=device)
        self.assertEqual(tensor.shape, (1, 3, 50, 50))  # Batch dimension added

    def test_display_image(self):
        """
        Test if tensors are correctly converted to numpy for plotting.
        """
        tensor = torch.randn(3, 100, 100)
        result = fiutils.display_image(tensor)

        self.assertIsInstance(result, np.ndarray)  # Check type
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertTrue((result >= 0).all() and (result <= 1).all())  # Check value range

    def test_load_tensor_from_path(self):
        """
        Test loading an image file to a tensor.
        """
        tensor = fiutils.load_tensor_from_path(self.test_img_name, target_size=(32, 32))
        self.assertEqual(tensor.shape, (1, 3, 32, 32))

    def test_validate(self):
        """
        Test if the validate function runs without crashing.
        """

        class DummyModel(torch.nn.Module):
            def forward(self, img1, img3):
                return img1

        model = DummyModel()
        device = torch.device("cpu")

        dataset = fiutils.FrameInterpolation(self.df, target_size=(10, 10))
        dataloader = DataLoader(dataset, batch_size=1)

        loss_fn = torch.nn.L1Loss()

        def metric(x, y):
            return torch.tensor(0.5)

        loss, accuracy = fiutils.validate(model, dataloader, loss_fn, metric, device, N=1)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)

    def test_model_output_shape(self):
        """
        Verifies that the model takes two frames and ouputs a single frame of the same dimension.
        """
        img1 = torch.randn(1, 3, 256, 256)
        img3 = torch.randn(1, 3, 256, 256)

        class DummyModel(nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.y = nn.Conv2d(6, 3, kernel_size=3, padding=1)

            def forward(self, x1, x2):
                x = torch.cat((x1, x2), dim=1)
                return self.y(x)

        model = DummyModel()
        output = model(img1, img3)

        self.assertEqual(output.shape, img1.shape)


if __name__ == "__main__":
    unittest.main()
