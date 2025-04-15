import os 
import litellm
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = self.eos_tokens 

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
        
        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
        )
        
        responses = [output["choices"][0]["message"].content for output in outputs]

        return responses

# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    



class LocalvLLM(LanguageModel):
    """适配AttackLM和TargetLM的本地vLLM模型类，使用DeepSeek R1模型格式"""
    
    def __init__(self, model_name: str):
        """初始化LocalvLLM模型"""
        super().__init__(model_name)
        from vllm import LLM, SamplingParams
        import os
        from config import MODEL_NAMES
        
        # 验证模型名称
        if "DeepSeek" in model_name:
            # 根据模型名称选择相应的模型路径
            if "DeepSeek-R1-Distill-Qwen-7B" in model_name:
                self.hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            elif "DeepSeek-R1-Distill-Llama-8B" in model_name:
                self.hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            elif "DeepSeek-R1-Distill-Qwen-1.5B" in model_name:
                self.hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            elif "DeepSeek-R1-Distill-Qwen-32B" in model_name:
                self.hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            else:
                raise ValueError(f"未知的DeepSeek模型: {model_name}")
        elif model_name in MODEL_NAMES:
            from config import HF_MODEL_NAMES
            self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
        else:
            raise ValueError(f"无效的模型名称: {model_name}")
        
        # 设置vLLM模型参数
        from config import TARGET_TEMP, TARGET_TOP_P, FASTCHAT_TEMPLATE_NAMES
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P
        self.max_n_tokens = 512  # 默认值，实际使用时会被覆盖
        
        # 初始化vLLM模型
        try:
            from utils import destroy_model_parallel
            destroy_model_parallel()
        except (ImportError, ModuleNotFoundError):
            # 如果没有destroy_model_parallel函数，就跳过
            pass
        
        # 初始化模型
        self.model = LLM(
            model=self.hf_model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            max_num_seqs=256,
            max_model_len=32768
        )
        
        # 属性设置，使其与AttackLM和TargetLM兼容
        self.use_open_source_model = True
        self.post_message = ""
        self.template = FASTCHAT_TEMPLATE_NAMES.get(model_name, "vicuna_v1.1")
    
    def _get_sampling_params(self, max_n_tokens, temperature, top_p, extra_eos_tokens=None):
        """获取采样参数"""
        from vllm import SamplingParams
        
        # 为DeepSeek模型添加特定的停止标记
        stop_tokens = ["<|im_end|>"]
        if extra_eos_tokens:
            stop_tokens.extend(extra_eos_tokens)
        
        if temperature > 0:
            return SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_n_tokens,
                stop=stop_tokens
            )
        else:
            return SamplingParams(
                temperature=0,
                max_tokens=max_n_tokens,
                stop=stop_tokens
            )
    
    def _format_prompt_deepseek(self, messages):
        """
        使用DeepSeek R1模型格式格式化OpenAI风格的消息
        
        DeepSeek R1格式:
        <|im_start|>system
        系统指令
        <|im_end|>
        <|im_start|>user
        用户消息
        <|im_end|>
        <|im_start|>assistant
        助手回复
        <|im_end|>
        """
        formatted_prompt = ""
        
        # 添加系统消息(如果有)
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
                break
        
        if system_message != '':
            formatted_prompt += f"<|im_start|>system\n{system_message}\n<|im_end|>\n"
        
        # 添加对话消息
        for msg in messages:
            if msg["role"] == "user":
                formatted_prompt += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "assistant" and msg["content"]:
                formatted_prompt += f"<|im_start|>assistant\n{msg['content']}"
                # 对于最后一条assistant消息，不要添加结束标记，以便让模型继续生成
                if msg != messages[-1]:
                    formatted_prompt += "\n<|im_end|>\n"
        
        # 如果最后一个消息不是assistant，添加assistant标记以开始生成
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"]:
            formatted_prompt += "<|im_start|>assistant\n"
            
        return formatted_prompt
    
    def batched_generate(self, messages_list, max_n_tokens=None, temperature=None, top_p=None, extra_eos_tokens=None):
        """
        为AttackLM和TargetLM提供的批量生成接口
        
        参数:
        - messages_list: OpenAI格式的消息列表的列表
        - max_n_tokens: 最大生成token数
        - temperature: 温度参数
        - top_p: top-p采样参数
        - extra_eos_tokens: 额外的EOS标记列表
        
        返回:
        - 生成的文本响应列表
        """
        # 使用提供的参数，如果没有则使用实例默认值
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        max_tokens = max_n_tokens if max_n_tokens is not None else self.max_n_tokens
        
        # 设置采样参数
        sampling_params = self._get_sampling_params(max_tokens, temp, tp, extra_eos_tokens)
        
        # 将OpenAI格式的消息转换为DeepSeek R1格式的提示词
        formatted_prompts = [self._format_prompt_deepseek(messages) for messages in messages_list]
        
        # 生成响应
        outputs = self.model.generate(formatted_prompts, sampling_params)
        
        # 提取生成的文本，并处理可能的<|im_end|>标记
        responses = []
        for output in outputs:
            text = output.outputs[0].text
            # 移除结尾的<|im_end|>（如果有）
            if text.endswith("<|im_end|>"):
                text = text[:-10].strip()
            responses.append(text)
        
        return responses
    
    def query(self, prompts, behavior=None, phase=None, max_new_tokens=None):
        """
        为与TargetLM类兼容添加的查询方法
        
        参数:
        - prompts: 字符串或字符串列表
        - behavior: 行为类别（可选）
        - phase: 阶段信息（可选）
        - max_new_tokens: 最大生成token数（可选）
        
        返回:
        - 带有responses属性的响应对象
        """
        class ResponseObject:
            def __init__(self, responses):
                self.responses = responses
        
        # 将单个提示转为列表
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # 构建消息列表
        messages_list = []
        for prompt in prompts:
            # 如果提供了behavior，添加系统消息
            if behavior:
                messages = [
                    {"role": "system", "content": f"You are an AI assistant that provides helpful and harmless responses. Category: {behavior}"},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            messages_list.append(messages)
        
        # 调用batched_generate
        responses = self.batched_generate(
            messages_list,
            max_n_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return ResponseObject(responses)


