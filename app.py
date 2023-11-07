import gradio as gr
import tensorflow as tf

# Load MNIST model
model = tf.keras.models.load_model('mnist.h5')

def recognizer(image):
    if image is not None:
        image = image.reshape((1,28,28,1)).astype('float32') / 255

        preds = model.predict(image)

        return {str(i): float(preds[0][i]) for i in range(10)}
    else:
        return ' '

ui = gr.Interface(
  fn=recognizer,
  inputs=gr.Image(shape=(28,28), image_mode='L', invert_colors=True, sources=['canvas']),
  outputs=gr.Label(num_top_classes=3),
  title='MNIST Recognizer',
  live=True
)

ui.launch()
