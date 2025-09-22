from typing import Any, Dict, List

from PIL import Image


def resize_images(
    images: List[Image.Image],
    config: Dict[str, Any]
):
    images_resized = [image.resize((config["IMG_SIZE"], config["IMG_SIZE"])) for image in images]
    if images_resized[0].mode != 'RGB':
        images_resized = [image.convert('RGB') for image in images_resized]
    return images_resized