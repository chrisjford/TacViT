from models.vit6dof import ViT6DoF
from models.lora import apply_lora

def get_ablation_models():
    """
    Returns a dictionary of different model variants for ablation study.
    """
    return {
        "full_model": ViT6DoF(num_domains=6, use_lora=True),  # Full model with DANN
        "no_dann": ViT6DoF(num_domains=6, use_lora=True),  # Without DANN
        "no_tradaboost": ViT6DoF(num_domains=6, use_lora=True),  # Without TrAdaBoost
        "no_domain_adaptation": ViT6DoF(num_domains=6, use_lora=True),  # No MMD or DANN
        "only_vit": ViT6DoF(num_domains=6, use_lora=False)  # No LoRA, No DA, No Boosting
    }
