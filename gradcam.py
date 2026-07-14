import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model):

    _ = model(img_array, training=False)

    
    base_model = model.layers[0]

   
    last_conv_layer = base_model.get_layer("out_relu")

   
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_array)

       
        x = model.layers[1](base_outputs)  
        x = model.layers[2](x)             
        x = model.layers[3](x, training=False)  
        predictions = model.layers[4](x)   

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return heatmap.numpy()

    heatmap = heatmap / max_value

    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(
        heatmap,
        (image.size[0], image.size[1])
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    image = np.array(image)

    superimposed_img = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img