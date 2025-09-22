import base64
from io import BytesIO
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

def image_to_base64(image: Image.Image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_image(base64_string: str):
    """Convert base64 string to PIL Image"""
    return Image.open(BytesIO(base64.b64decode(base64_string)))
