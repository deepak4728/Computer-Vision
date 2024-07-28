from django.shortcuts import render

# Create your views here.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image

from django.shortcuts import render, redirect
from django import forms
from forms import ImageUploadForm
from models import ImageUpload

# def load_and_process_img(img_path):
#     img = Image.open(img_path)
#     img = img.resize((400, 400))
#     img = kp_image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = vgg19.preprocess_input(img)
#     return img
def load_and_process_img(img_path):
    img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((400, 400))
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in ['block1_conv1',
                                                             'block2_conv1',
                                                             'block3_conv1',
                                                             'block4_conv1',
                                                             'block5_conv1']]
    content_outputs = [vgg.get_layer('block5_conv2').output]
    model_outputs = style_outputs + content_outputs
    return Model(vgg.input, model_outputs)

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:5]
    content_output_features = model_outputs[5:]
    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(5)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * tf.reduce_mean(tf.square(comb_style - target_style))

    weight_per_content_layer = 1.0 / float(1)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * tf.reduce_mean(tf.square(comb_content - target_content))

    style_score *= style_weight
    content_score *= content_weight
    loss = style_score + content_score
    return loss

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    style_features = [style_layer[0] for style_layer in style_outputs[:5]]
    content_features = [content_layer[0] for content_layer in content_outputs[5:]]
    return style_features, content_features

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def perform_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    opt = Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    iter_count = 1
    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -103.939, 255.0 - 103.939)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

    return best_img

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_upload = form.save()
            content_img_path = img_upload.content_image.path
            style_img_path = img_upload.style_image.path
            output = perform_style_transfer(content_img_path, style_img_path)
            
            output_img = Image.fromarray(output)
            output_img_path = f'media/output_images/output_{img_upload.id}.jpg'
            output_img.save(output_img_path)
            img_upload.output_image = output_img_path
            img_upload.save()
            return redirect('result', img_upload.id)
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})

def result(request, pk):
    img_upload = ImageUpload.objects.get(pk=pk)
    return render(request, 'result.html', {'img_upload': img_upload})
