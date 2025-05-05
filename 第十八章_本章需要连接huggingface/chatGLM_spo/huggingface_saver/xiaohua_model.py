import torch
from 第十八章_本章需要连接huggingface.chatGLM_spo.huggingface_saver import configuration_chatglm,modeling_chatglm

class XiaohuaModel(torch.nn.Module):
    def __init__(self,model_path = "./chatglm6b.pth",config = None,strict = True):

        super().__init__()
        self.glm_model = modeling_chatglm.ChatGLMForConditionalGeneration(config)
        model_dict = torch.load(model_path)

        self.glm_model.load_state_dict(model_dict,strict = strict)

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self,input_ids,labels = None,position_ids = None,attention_mask = None):
        logits,hidden_states = self.glm_model.forward(input_ids=input_ids,position_ids = None,attention_mask = None)

        loss = None
        if labels != None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Flatten the tokens
            logits_1 = shift_logits.view(-1, shift_logits.size(-1))
            logits_2 = shift_labels.view(-1)

            loss = self.loss_fct(logits_1, logits_2)

        return logits,hidden_states,loss

    def generate(self,start_question_text="抗原呈递的原理是什么？",continue_seq_length = 128,tokenizer = None,temperature = 0.95, top_p = 0.95):
        """
        Args:
            start_question_text:这里指的是起始的问题 ，需要用中文进行展示
            continue_seq_length: 这里是在question后面需要添加的字符
            temperature:
            top_p:
        Returns:
        --------------------------------------------------------------------------------------------
        记录：这个tokenizer可能会在开始encode的时候，在最开始加上一个空格20005
        --------------------------------------------------------------------------------------------
        下面是做多轮问答的时候用的peompt，现在我还没实现，不想用
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        """

        #这里就是我写的一个简单例子，用来判定问答还是做其他工作。
        if "：" not in start_question_text:
            inputs_text_ori = start_question_text
            inputs_text = f"[Round 0]\n问：{inputs_text_ori}\n答："
        else:
            inputs_text = start_question_text

        input_ids = tokenizer.encode(inputs_text)

        for _ in range(continue_seq_length):
            input_ids_tensor = torch.tensor([input_ids]).to("cuda")
            logits,_,_ = self.forward(input_ids_tensor)
            logits = logits[:,-3]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = self.sample_top_p(probs, top_p)  # 预设的top_p = 0.95
            #next_token = next_token.reshape(-1)

            # next_token = result_token[-3:-2]
            input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
            if next_token.item() == 130005:
                print("break")
                break
        result = tokenizer.decode(input_ids)
        return result

    def sample_top_p(self,probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import tokenization_chatglm

    config = configuration_chatglm.ChatGLMConfig()
    model = XiaohuaModel(config=config).half().cuda()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,cache_dir = "./huggingface_saver")
    inputs_text_ori = "抗原呈递的原理是什么？"
    result = model.generate(inputs_text_ori, continue_seq_length=256,tokenizer=tokenizer)
    print(result)

    while True:
        print("请输入:")
        ques = input()
        inputs_text_ori = ques
        result = model.generate(inputs_text_ori, continue_seq_length=256, tokenizer=tokenizer)
        print(result)
