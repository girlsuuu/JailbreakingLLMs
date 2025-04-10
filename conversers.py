from common import get_api_key, conv_template, extract_json
from language_models import APILiteLLM
from config import FASTCHAT_TEMPLATE_NAMES, Model


def load_attack_and_target_models(args):
    # 将AttackLM加载到GPU 0,1
    attackLM = AttackLM(model_name=args.attack_model, 
                      max_n_tokens=args.attack_max_n_tokens, 
                      max_n_attack_attempts=args.max_n_attack_attempts, 
                      category=args.category,
                      evaluate_locally=args.evaluate_locally,
                      cuda_device="0,1"  # 指定使用的GPU
                      )
    
    # 将TargetLM加载到GPU 2,3
    targetLM = TargetLM(model_name=args.target_model,
                       category=args.category,
                       max_n_tokens=args.target_max_n_tokens,
                       evaluate_locally=args.evaluate_locally,
                       phase=args.jailbreakbench_phase,
                       cuda_device="2,3"  # 指定使用的GPU
                       )
    
    return attackLM, targetLM

def load_indiv_model(model_name, local=False, use_jailbreakbench=True, cuda_device=None):
    if cuda_device is not None:
        # 设置当前函数作用域内的CUDA可见设备
        import os
        old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        
    if use_jailbreakbench: 
        if local:
            from jailbreakbench import LLMvLLM
            lm = LLMvLLM(model_name=model_name)
        else:
            from jailbreakbench import LLMLiteLLM
            api_key = get_api_key(Model(model_name))
            lm = LLMLiteLLM(model_name= model_name, api_key = api_key)
    else:
        if local:
            # 实现本地运行DeepSeek模型的逻辑
            from vllm import LLM, SamplingParams
            
            # 根据模型名称选择相应的模型路径
            if "DeepSeek-R1-Distill-Qwen-7B" in model_name:
                model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 替换为实际模型路径
            elif "DeepSeek-R1-Distill-Llama-8B" in model_name:
                model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 替换为实际模型路径
            elif "DeepSeek-R1-Distill-Qwen-1.5B" in model_name:
                model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 替换为实际模型路径
            else:
                raise ValueError(f"未知的本地模型: {model_name}")
            
            # 初始化VLLM模型
            vllm_model = LLM(
                model=model_path,
                dtype="bfloat16",
                gpu_memory_utilization=0.5,  
                tensor_parallel_size=1,     
                max_num_seqs=256,            
                max_model_len=4096          
            )
            
            # 创建一个包装类，使其接口与其他LM保持一致
            class LocalDeepSeekModel:
                def __init__(self, vllm_model, model_name):
                    self.model = vllm_model
                    self.model_name = model_name
                    self.use_open_source_model = True  # 与AttackLM的initialize_output兼容
                    self.post_message = ""  # 用于在AttackLM._generate_attack中拼接输出
                    self.template = FASTCHAT_TEMPLATE_NAMES.get(model_name, "vicuna_v1.1")  # 为了兼容可能的模板引用
                
                def batched_generate(self, messages_list, max_n_tokens=None, temperature=0.7, top_p=0.9, extra_eos_tokens=None):
                    """
                    实现与AttackLM和TargetLM兼容的batched_generate方法
                    
                    参数:
                    - messages_list: OpenAI格式的消息列表列表
                    - max_n_tokens: 最大生成token数量
                    - temperature: 温度参数
                    - top_p: top-p采样参数
                    - extra_eos_tokens: 额外的EOS标记列表
                    
                    返回:
                    - 生成的文本响应列表
                    """
                    responses = []
                    
                    # 设置采样参数
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_n_tokens if max_n_tokens else 512,
                        stop=extra_eos_tokens if extra_eos_tokens else None
                    )
                    
                    # 处理每一条消息
                    for messages in messages_list:
                        # 构造完整的prompt
                        full_prompt = ""
                        for i, msg in enumerate(messages):
                            role = msg["role"]
                            content = msg["content"]
                            
                            # 简单的提示词工程，可以根据具体模型调整
                            if role == "system":
                                full_prompt += f"<system>\n{content}\n</system>\n"
                            elif role == "user":
                                full_prompt += f"<human>\n{content}\n</human>\n"
                            elif role == "assistant":
                                # 如果是最后一条消息且来自assistant，这可能是AttackLM的初始化提示
                                if i == len(messages) - 1:
                                    full_prompt += f"<answer>\n{content}"  # 不加结束标记，让模型继续生成
                                else:
                                    full_prompt += f"<answer>\n{content}\n</answer>\n"
                        
                        if not full_prompt.endswith("<answer>\n"):
                            full_prompt += "<answer>\n"  # 添加最后的答案前缀
                        
                        # 生成回复
                        outputs = self.model.generate(full_prompt, sampling_params)
                        
                        # 获取生成的文本
                        generated_text = outputs[0].outputs[0].text
                        responses.append(generated_text)
                    
                    return responses
                
                def query(self, prompts, behavior=None, phase=None, max_new_tokens=None):
                    """
                    为与TargetLM类兼容添加的方法，虽然在use_jailbreakbench=False时不会调用
                    但为了代码的完整性，我们也实现它
                    
                    返回一个带有responses属性的对象
                    """
                    # 创建一个简单的对象来模拟jailbreakbench的响应格式
                    class ResponseObject:
                        def __init__(self, responses):
                            self.responses = responses
                    
                    # 将单个提示转为列表
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    
                    # 构建消息列表
                    messages_list = []
                    for prompt in prompts:
                        # 为每个提示创建一个简单的消息列表
                        messages = [{"role": "user", "content": prompt}]
                        messages_list.append(messages)
                    
                    # 调用batched_generate
                    responses = self.batched_generate(
                        messages_list,
                        max_n_tokens=max_new_tokens,
                        temperature=0.7,  # 使用默认温度
                        top_p=0.9         # 使用默认top_p
                    )
                    
                    return ResponseObject(responses)
            
            lm = LocalDeepSeekModel(vllm_model, model_name)
        else:
            lm = APILiteLLM(model_name)
            
        # 恢复原来的CUDA_VISIBLE_DEVICES设置
        if cuda_device is not None:
            if old_cuda_visible_devices:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
            else:
                del os.environ['CUDA_VISIBLE_DEVICES']
    return lm

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                category: str,
                evaluate_locally: bool,
                cuda_device=None):
        
        # self.model_name = Model(model_name)
        self.model_name = model_name
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts

        from config import ATTACK_TEMP, ATTACK_TOP_P
        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P

        self.category = category
        self.evaluate_locally = evaluate_locally
        self.model = load_indiv_model(model_name, 
                                      local = evaluate_locally, 
                                      use_jailbreakbench=False, # Cannot use JBB as attacker
                                      cuda_device=cuda_device
                                      )
        self.initialize_output = self.model.use_open_source_model
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        # For open source models, we can seed the generation with proper JSON
        init_message = ""
        if self.initialize_output:
            
            # Initalize the attack model's generated output to match format
            if len(convs_list[0].messages) == 0:# If is first message, don't need improvement
                init_message = '{"improvement": "","prompt": "'
            else:
                init_message = '{"improvement": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message
        
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize
        
        # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            # Generate outputs 
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        extra_eos_tokens=["}"]
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                full_output = init_message + full_output + "}" # Add end brace since we terminate generation on end braces
                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    new_adv_prompts[orig_index] = json_str
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            raise ValueError(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        # Convert conv_list to openai format and add the initial message
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            # For open source models, we can seed the generation with proper JSON and omit the post message
            # We add it back here
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)
        
        return valid_outputs

class TargetLM():
    """
        JailbreakBench class for target language models.
    """
    def __init__(self, 
            model_name: str, 
            category: str,
            max_n_tokens : int,
            phase: str,
            evaluate_locally: bool = False,
            use_jailbreakbench: bool = True,
            cuda_device=None
            ):
        
        self.model_name = model_name
        self.max_n_tokens = max_n_tokens
        self.phase = phase
        self.use_jailbreakbench = use_jailbreakbench
        self.evaluate_locally = evaluate_locally

        from config import TARGET_TEMP,  TARGET_TOP_P   
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P

        # self.model = load_indiv_model(model_name, evaluate_locally, use_jailbreakbench)
        self.model = load_indiv_model(model_name, evaluate_locally, use_jailbreakbench=False, cuda_device=cuda_device)              
        self.category = category
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def get_response(self, prompts_list):
        if self.use_jailbreakbench:
            llm_response = self.model.query(prompts = prompts_list, 
                                behavior = self.category, 
                                phase = self.phase,
                                max_new_tokens=self.max_n_tokens)
            responses = llm_response.responses
        else:
            batchsize = len(prompts_list)
            convs_list = [conv_template(self.template) for _ in range(batchsize)]
            full_prompts = []
            for conv, prompt in zip(convs_list, prompts_list):
                conv.append_message(conv.roles[0], prompt)
                full_prompts.append(conv.to_openai_api_messages())

            responses = self.model.batched_generate(full_prompts, 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
        
        # llm_response = self.model.query(prompts = prompts_list, 
        #                         behavior = self.category, 
        #                         phase = self.phase,
        #                         max_new_tokens=self.max_n_tokens)
        # responses = llm_response.responses
           
        # return responses