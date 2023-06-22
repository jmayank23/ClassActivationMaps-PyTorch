# Class Activation Maps_ using PyTorch

This code provides an easy-to-use implementation for generating state-of-the-art Class Activation Maps (CAMs) using the `pytorch-grad-cam` library. CAMs are useful visualizations that highlight the regions of an image that contribute most to the prediction of a specific class by a convolutional neural network (CNN).

## Installation

Before running the code, make sure to install the `grad-cam` library by executing the following command:

```
pip install grad-cam
```

## Usage

1. Import the necessary modules:

```python
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import torchvision
import matplotlib.pyplot as plt
```

2. Define the CAM methods you want to use:

```python
cam_methods = {
    'GradCAM': GradCAM,
    # 'GradCAM++': GradCAMPlusPlus,
    # 'HiResCAM': HiResCAM,
    # 'ScoreCAM': ScoreCAM,
    # 'AblationCAM': AblationCAM,
    # 'XGradCAM': XGradCAM,
    # 'EigenCAM': EigenCAM,
    'FullGrad': FullGrad
}
```

Uncomment the desired methods based on your requirements.

3. Load the input image:

```python
rgb_img = cv2.imread('/path/to/image.jpg') / 255.0
```

Make sure to specify the correct path to your image.

4. Prepare the image for input to the CAM model:

```python
resize_to = (224, 224)
transform_norm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(size=resize_to, antialias=True),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform_norm(rgb_img).unsqueeze(0)
```

Adjust the `resize_to` dimensions according to your model's input size and preprocessing requirements.

5. Load your pre-trained model and specify the target layers:

```python
model = torchvision.models.resnet50(pretrained='imagenet').double()
target_layers = [model.layer4[-1]]
```

Replace `resnet50` with your desired model architecture, and modify `target_layers` based on the layers you want to visualize.

6. Create CAM objects for each method:

```python
cams = {}
for cam_name, cam_method in cam_methods.items():
    cams[cam_name] = cam_method(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
```

Ensure that the `use_cuda` flag is set to `True` if you have CUDA-enabled GPU(s) available.

7. Specify the targets (optional):

```python
targets = None
# targets = [ClassifierOutputTarget(281)]
```

If you want to generate CAMs for specific targets, uncomment the second line and modify the target index accordingly.

8. Generate visualizations for each CAM method:

```python
visualizations = {}
for cam_name, cam in cams.items():
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(cv2.resize(rgb_img, resize_to), grayscale_cam, use_rgb=True)
    visualizations[cam_name] = visualization
```

9. Plot the original image and the visualizations:

```python
num_visualizations = len(visualizations)
fig, axs = plt.subplots(1,

 num_visualizations + 1, figsize=(4 * (num_visualizations + 1), 4))
axs[0].imshow(cv2.resize(rgb_img, resize_to))
axs[0].set_title('Original Image')

for i, (cam_name, visualization) in enumerate(visualizations.items()):
    axs[i + 1].imshow(visualization)
    axs[i + 1].set_title(cam_name)

plt.tight_layout()
plt.show()
```

Adjust the plot settings according to your preferences.

## Acknowledgments

This code utilizes the `pytorch-grad-cam` library developed by [Jacob Gil](https://github.com/jacobgil/pytorch-grad-cam). Make sure to consult the library's documentation for more advanced usage and additional features.

## License

The code is provided under the [MIT License](https://opensource.org/licenses/MIT). Feel free to modify and use it according to your needs.

Please note that this readme assumes some familiarity with deep learning concepts and Python programming. If you have any further questions or issues, don't hesitate to consult the library's documentation or seek additional support.
