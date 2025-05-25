import torch
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

# load BLIP model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        # wrap the Python function
        result = tf.py_function(self.process_image,
                                inp=[image_path, task],
                                Tout=tf.string)
        # ensure TensorFlow knows this is a scalar string
        result.set_shape(())
        return result

    def process_image(self, image_path, task):
        """
        image_path: a tf.string tensor containing the file path
        task:       a tf.string tensor, either "caption" or "summary"
        """
        try:
            # decode tensors to Python strings
            path = image_path.numpy().decode("utf-8")
            mode = task.numpy().decode("utf-8")

            # load & preprocess
            image = Image.open(path).convert("RGB")
            prompt = (
                "This is a picture of"
                if mode == "caption"
                else "This is a detailed photo showing"
            )
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            # generate & decode
            output = self.model.generate(**inputs)
            text = self.processor.decode(output[0], skip_special_tokens=True)
            return text

        except Exception as e:
            print(f"Error processing {path!r}: {e}")
            return "Error processing image"


def generate_text(image_path: str, task: str) -> str:
    """
    image_path: Path to image file
    task:       "caption" or "summary"
    """
    layer = BlipCaptionSummaryLayer(processor, model)
    img_tensor = tf.constant(image_path)
    task_tensor = tf.constant(task)
    out_tensor = layer(img_tensor, task_tensor)
    return out_tensor.numpy().decode("utf-8")


if __name__ == "__main__":
    # example usage
    img_file = "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"

    caption = generate_text(img_file, "caption")
    print("Generated Caption:", caption)

    summary = generate_text(img_file, "summary")
    print("Generated Summary:", summary)

    # display the image
    img = plt.imread(img_file)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
