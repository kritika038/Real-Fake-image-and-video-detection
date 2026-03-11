import cv2
import numpy as np
import torch


def generate_heatmap(model, image_tensor):

    image_tensor.requires_grad = True

    output = model(image_tensor)

    class_idx = torch.argmax(output)

    model.zero_grad()

    output[0, class_idx].backward()

    gradients = image_tensor.grad[0].cpu().numpy()

    gradients = np.mean(gradients, axis=0)

    heatmap = np.maximum(gradients, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(original_img, heatmap):

    original = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay