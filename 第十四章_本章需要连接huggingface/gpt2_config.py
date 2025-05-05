class GPT2Config(object):
    def __init__(
            self,
            vocab_size=1024,						#字符个数
            n_positions=1024,						#位置embedding的维度
            n_ctx=1024,							#注意力中的embedding的维度
            n_embd=768,						#GPT模型维度
            n_layer=12,							#GPT中Block的层数
            n_head=12,							#GPT中的注意力头数
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
