# InFluxEdit
# Flux Fill LoRA Editing

This project explores image editing with **FLUX.1 Fill [dev]** by leveraging (a) its powerful outpainting and inpainting capabilities, and (b) using its additional channels in a slightly modified way. Our method enables high-fidelity image editing while avoiding full model finetuning, building on prior efforts such as **ACE++**.

## 🧠 Key Idea

We investigate two methods for using **Flux Fill [dev]** to generate training data for image editing LoRAs:

### a) **Spatial Concatenation**

Concatenate the **input image** and **generated output image** side-by-side and use a mask to supervise the right (output) half. This approach was previously used by the AI community with LoRAs.

![Spatial Concatenation](https://github.com/andjoer/InFluxEdit/raw/main/graphics/concat_spacial.png)

### b) **Extra Channel Masking (New Approach)**

Feed only the **input image** to Flux Fill and use a white mask, but do not apply the mask on the input image itself (i.e., do not zero out masked pixel values as is standard with Flux Fill). This retains reference context and boosts output fidelity. Unlike ACE++, we do not fully finetune the model when changing the purpose of the channels.

![Extra Channel Masking](https://github.com/andjoer/InFluxEdit/raw/main/graphics/concat_channels.png)

---

## 💡 Method Summary

- The model uses **Flux Fill [dev]** inpainting mode with an all-white mask in the extra channel.
- LoRA adapters are trained with this setup—**no full model finetune is needed**.
- Optionally, the **`x_embedder`** module (which encodes extra input channels) can be set as trainable.

---

## 🔬 Findings (Preliminary)

1. **LoRA training alone is sufficient** for high-quality editing using extra channel input.
2. **Training `x_embedder` is optional** – it may help in some tasks but doesn’t harm performance.
3. **Spatial concatenation leads to lower fidelity** results compared to processing the unmasked input.

---

## 🎜️ License

This work uses FLUX.1 Fill [dev], which is distributed under the [FLUX.1 [dev] Non-Commercial License](https://bfl.ml).

---

## 📊 Acknowledgments

- **FLUX.1 Fill [dev]**: for cutting-edge image completion and inpainting.
- **ACE++**: for introducing the spatial concatenation approach for training.
- **Sebastian-Zok**: to our knowledge, the first to release a training script for Flux Fill LoRAs. This project borrows from his work.


