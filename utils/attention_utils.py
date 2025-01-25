import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
from modules.transformer_temporal import MatchingBasicTransformerBlock


def set_matching_attention(unet): 
    for name, module in unet.named_children():
        if isinstance(module, BasicTransformerBlock):
            new_module = MatchingBasicTransformerBlock(
                dim = module.dim,
                num_attention_heads = module.num_attention_heads,
                attention_head_dim = module.attention_head_dim,
                dropout = module.dropout,
                cross_attention_dim = module.cross_attention_dim,
                activation_fn = module.activation_fn,
                num_embeds_ada_norm = module.num_embeds_ada_norm,
                attention_bias = module.attention_bias,
                only_cross_attention = module.only_cross_attention,
                double_self_attention = module.double_self_attention,
                norm_elementwise_affine = module.norm_elementwise_affine,
                norm_type = module.norm_type,
                positional_embeddings = module.positional_embeddings,
                num_positional_embeddings = module.num_positional_embeddings,
            )
            new_module.load_state_dict(module.state_dict(),strict=False)
            setattr(unet, name, new_module)
        else:
            set_matching_attention(module)
    return unet

def set_matching_attention_processor(unet, attn_processor):
    for block in unet.down_blocks:
        if hasattr(block, "attentions"):
            for attn in block.attentions:
                if isinstance(attn, TransformerSpatioTemporalModel):
                    for a_block in attn.transformer_blocks:
                        if isinstance(a_block, MatchingBasicTransformerBlock):
                            a_block.sparse_attention.processor = attn_processor

    for attn in unet.mid_block.attentions:
        if isinstance(attn, TransformerSpatioTemporalModel):
            for a_block in attn.transformer_blocks:
                    if isinstance(a_block, MatchingBasicTransformerBlock):
                            a_block.sparse_attention.processor = attn_processor
            
    for block in unet.up_blocks:
        if hasattr(block, "attentions"):
            for attn in block.attentions:
                if isinstance(attn, TransformerSpatioTemporalModel):
                    for a_block in attn.transformer_blocks:
                        if isinstance(a_block, MatchingBasicTransformerBlock):
                            a_block.sparse_attention.processor = attn_processor
    return unet
