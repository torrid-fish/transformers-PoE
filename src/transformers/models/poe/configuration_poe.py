from ...configuration_utils import PretrainedConfig
from ...utils import logging
from typing import List

logger = logging.get_logger(__name__)

class PoEModelConfig(PretrainedConfig):
    r"""
    This configuration is mostly based on the configuration from mixtral.
    """

    model_type = "poe" # [2024/10/19 torridfish] Change back to poe (poe -> mixtral -> poe)

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1e6,
        sliding_window: int = None,
        attention_dropout: float = 0.0,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8, # Set to 1 to reproduce the normal FFN
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        ## PoE specific
        pool_layer: List[list] = [[-3, -2, -1]],
        pool_routing_type: str = "Top2",
        pool_routing_config: dict = {},
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # PoE specific
        self.pool_layer = pool_layer
        self.pool_routing_type = pool_routing_type
        self.pool_routing_config = pool_routing_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
