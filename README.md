# Custom Vit Implementation for Cifar10 classification

This repo contains the basic components for a simple ViT classification. 
- components: standard components used for a ViT such as LayerNorm, MHA, MLP, TransformerBlock etc.
- model: image patchification, transformer model using the modules from "components"
- main.py: a basic training loop for CIFAR10. 

The implementation avoids using existing modules from PyTorch (such as torch.nn.LayerNorm) for educational purposes.
In most cases I stick with nn.Parameter and handle the forward passes on my own.

I will incrementally enrich this repo with DINO implementation
