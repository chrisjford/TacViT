from peft import LoraConfig, get_peft_model

def apply_lora(model):
    """
    Applies LoRA to the ViT model to reduce the number of trainable parameters.
    """
    config = LoraConfig(
        r=8,  # Low-rank update size
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]  # LoRA on attention layers
    )

    # Apply LoRA only if enabled
    if model.use_lora:
        model = get_peft_model(model, config)

    return model
