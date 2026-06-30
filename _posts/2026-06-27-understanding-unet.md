---
layout: post
title: Understanding U-Net as a “what → where” machine
date: 2026-06-26 10:00:00+0200
description: A visual walkthrough of U-Net for image segmentation, explaining how CNNs learn “what” is in an image but lose precise “where” information. The post breaks down the U shaped architecture in down-sample blocks, up-sample blocks, skip connections, transposed convolutions, and supervised training with segmentation masks.
tags: u-net network-architecture segmentation deep-learning
categories: paper-notes
thumbnail: assets/img/blog/unet/unet_u_shape.png
toc:
  beginning: true
related_posts: false
mermaid:
  enabled: true
  zoomable: false
---

CNN architectures are great at recognizing **_what_** is in an image. They are good at extracting abstract features using CNN layers, and these features can then be used by simple feedforward networks to generate task outputs such as classification labels.

While learning _what_ is in an image is useful, knowing **_where_** something is is also very relevant and important. Learning the localization, i.e. the pixel locations corresponding to an identified label, is the **segmentation** task.

> A classification problem with CNN and MLP layers can predict that there is probably a robot arm, a cell boundary, a gripper, or a foreground object somewhere in the image. But a segmentation model has to answer a denser question: does pixel `(px, py)` belong to the robot arm or to the background?

This is where Ronneberger, Fischer, and Brox introduced U-Net {% cite ronneberger2015u --file blogs_unet %}. Their motivation was to solve segmentation tasks on light microscopy images from the ISBI Cell Tracking Challenge 2015.

Although there are other ways to perform image segmentation, U-Net is reminiscent of encoder-decoder architectures and uses interesting tricks to improve segmentation efficiency. Moreover, U-Net-inspired architectures are frequently used in modern foundation models.

This post looks at U-Net and its components to better understand this architecture.

## CNNs understand, but they blur location, making them difficult to use directly for segmentation

To give a quick overview of CNNs, let us assume we have an image and want to identify which animal is in it.

This can be done with CNN architectures, where there are what I would call CNN layers and MLP layers. CNN layers take input images, stored as tensors, or processed tensors from previous CNN layers, and apply three common operations:

```text
Convolution
Non-linear activation, i.e. ReLU
Pooling or sub-sampling
```

A convolutional block contains at least one instance of these operations. Each time such a block is applied, the image size is compressed and the number of channels progressively increases.

The channels of the tensor accumulate abstract information as the tensor passes through these operations. A CNN architecture typically consists of many convolutional blocks followed by an MLP layer, which converts the abstract information stored in the final channels into outputs.

These outputs are generally classification labels, for instance animal classes in the example above. Some variations can also include bounding box information.
See the blog post by Karn for more details on CNN architecture {% cite karn2016convnets --file blogs_unet %}.

{% include figure.liquid loading="eager" path="assets/img/blog/unet/unet_down_block.png" class="img-fluid rounded z-depth-1" zoomable=true %}
_Note that the **Down Block** in the diagram uses the same basic operations as a CNN layer, with repeated convolution and ReLU operations. U-Net uses the same design principle of compressed spatial size and increased channels in each block to capture semantic information in the network._

Pooling operations in CNNs are important because, during pooling, we lose some image detail in order to build image context. By compressing the image, the same-sized convolution operator in the next CNN layer effectively looks at a larger region of the original image. Thus, **pooling allows the network to look at larger parts of the image, but it does so by shrinking the spatial grid**.

By the time the network reaches deeper layers, it may be able to understand that there is a cat-like thing in the image. However, the exact boundary has become blurry because of repeated shrinking of the image.

For classification, this is fine, because the MLP only needs to predict the label and its confidence. However, segmentation becomes difficult with a compressed abstract representation of the image, because pixel-level details are lost.

## Overall concept of a U-Net

U-Net solves the problem of losing spatial detail in convolutional blocks, or down-sample blocks, by using up-convolutions and skip connections in the up-sample blocks.

The architecture effectively uses two kinds of information:

```text
uncompressed information from the corresponding down-sample block
+
recovered information from previous up-sample layers
```

Together, these help recover the spatial detail needed for the segmentation task.

#### Contracting path: down-sample blocks help zoom out to understand context

The left side of U-Net is the contracting path. It looks like a standard CNN block repeated several times.

At each level, the spatial resolution shrinks while the number of channels increases.

When the image becomes spatially smaller, the network can afford to store more feature channels. These channels are no longer just local edge-like features. They can represent more abstract patterns such as object parts, shapes, context, and higher-level structure.

{% include figure.liquid loading="eager" path="assets/img/blog/unet/unet_u_shape.png" class="img-fluid rounded z-depth-1" zoomable=true %}

\_The down-sample blocks contract the image. The bottleneck contains high-level meaning. The up-sample blocks expand the representation back toward an output map. The skip connections carry high-resolution information from the down-sample blocks directly to the corresponding up-sample blocks.

#### Expanding path: up-sample blocks help recover where things are

The up-sample blocks progressively increase spatial resolution.

However, upsampling alone is not enough. If we only upsample the previous tensor features, we get a larger map, but with the same limited information about the original image structure.

Thus, to recover lost spatial detail, U-Net uses skip connections from the contracting path. These skip features come from tensors saved before pooling operations were performed.

## up-sample block, step by step

One up-sample block contains three conceptually separate operations: upsampling, concatenation with the skip feature, and convolutional refinement.
In short, an up-sample block does three things:

```text
upsample
concatenate skip features
refine using convolution + ReLU
```

Suppose the bottleneck tensor has size `28 × 28 × 1024`. After an up-convolution, the tensor becomes `56 × 56 × 512`.

This is, in a way, the opposite of a CNN/down-sample block. In the down-sample block, the number of channels increases while the spatial size decreases. In the up-sample block, upsampling increases the spatial size, while the number of channels is reduced.

The output is now larger, but it has still not recovered the lost spatial information. To recover that information, U-Net retrieves the skip feature from the corresponding block on the left side of the down-sample path.

We use the layer with the matching spatial size from before max-pooling. In this example, the skip feature would have size `56 × 56 × 512`.

Note that the height and width may vary depending on padding and stride. In the original paper, the authors adjusted this discrepancy by cropping the tensors before concatenation.

Then the two tensors are simply _concatenated_. Concatenation happens across channels, not across space.

```python
# cat(56 × 56 × 512, 56 × 56 × 512) -> (56 × 56 × 1024)
x = torch.cat([upsampled, skip], dim=1)

# data is shaped as (Batch, Channel, Height, Width)
```

Thus, the block now contains both semantic information from the previous up-sample block or bottleneck, and spatial information from the skip connection. Concatenation puts both kinds of evidence in the same tensor.

Then the next convolution and non-linear activation functions learn how to combine this information and prepare it for the next block.

{% include figure.liquid loading="eager" path="assets/img/blog/unet/unet_up_block.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### Skip connections are an important trick

It is important to note that skip features are saved **before pooling**.

The tensor outputs at each down-sample block are saved. Later, when the up-sample path comes back to the same spatial scale, the saved tensor is retrieved and combined with the upsampled feature.

Why is this useful?

Because the saved tensor still contains relatively precise spatial information, such as edges, corners, boundaries, textures, and local structure. This information is more spatially detailed than what is available in the upsampled tensor alone.

### Transposed convolution for the upsampling operation

The up-convolution, or upsampling operation, is often performed using a transposed convolution.

```python
nn.ConvTranspose2d(
    in_channels=1024,
    out_channels=512,
    kernel_size=2,
    stride=2,
)
```

In this example, the operation takes a tensor with `1024` channels and produces a tensor with `512` channels. With `kernel_size=2` and `stride=2`, the spatial size is doubled.

So, for example:

```text
28 × 28 × 1024
↓ transposed convolution
56 × 56 × 512
```

A useful way to think about transposed convolution is that it does not simply copy or stretch pixels. Instead, it learns how to place information from a smaller feature map into a larger feature map.

The stride controls how far apart the input locations are spread in the larger output grid. The kernel then learns how each input location contributes to nearby output locations. If several contributions overlap, they are summed.

{% include figure.liquid loading="eager" path="assets/img/blog/unet/transposed_conv_stride2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

_Figure 3: Transposed convolutions spread input locations apart according to stride, stamp a learned kernel at each location, and sum the overlapping contributions._

In the U-Net up-sample block, this operation increases the spatial size of the tensor. This makes it possible to concatenate the upsampled tensor with the corresponding tensor from the skip connection.

### Conv + ReLU in the up-sample block combine meaning and detail

After concatenation, the tensor is passed through convolution and a non-linear activation function. These operations can be repeated within the block.

We can think of this part as functioning like a decoder. It tries to move the tensor back toward an image-like representation using two sources of information:

```text
semantic & low resolution information from the previous layer
+
spatial detail from the skip connection
```

The previous layer provides low-resolution, abstract information about _what_ is present. The skip connection provides more local, image-like information that helps fill in the missing details.

Hence, the up-sample block takes semantic information from previous layers and augments it with less abstract, more spatially detailed information received through skip connections. This helps the network get closer to the final segmentation map.

## How U-Net is trained

{% include figure.liquid loading="eager" path="assets/img/blog/unet/unet_training_loop.png" class="img-fluid rounded z-depth-1" zoomable=true %}

_Figure 4: U-Net is trained end-to-end. The model maps an image tile to dense pixel-wise logits, which are compared against a segmentation mask._

U-Net is trained in a supervised way. The input is an image tile, and the target is a ground-truth segmentation mask.

The model outputs a dense pixel-wise prediction map. Each output pixel contains class scores, for example foreground/background or cell/boundary/background. These predictions are compared with the corresponding labels in the segmentation mask, and the loss is computed over pixels.

Interestingly, the original paper also used a weight map so that difficult pixels, especially thin borders between touching cells, contributed more strongly to the loss.

## Retrieval summary

The whole architecture can be reduced to a compact picture:

```text
down-sample block:
    reduce spatial size
    compute and update abstract meaning in channels

Bottleneck:
    store compressed meaning in the most compressed H × W format

up-sample block:
    recover spatial size
    combine semantic meaning with spatial detail

Skip connection:
    restore detail
```

The key insight is that U-Net has a U-shape because it first compresses the image with down-sample blocks to extract meaning in the channels. Then it uses up-convolutions and skip connections to move back toward the image size while keeping useful semantic information.

A plain CNN gives us semantic understanding, but tends to lose precise localization through pooling. U-Net recovers localization by saving high-resolution features from the down-sample blocks and reusing them during the up-sample blocks.

---

<h2>References</h2>

---

{% bibliography --file blogs_unet --cited_in_order --template bib_blog --group_by none %}
