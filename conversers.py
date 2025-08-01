from common import get_api_key, conv_template, extract_json
from language_models import APILiteLLM, LocalvLLM
from config import FASTCHAT_TEMPLATE_NAMES, Model
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from loggers import logger
import time


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
                       use_jailbreakbench = args.use_jailbreakbench,
                       cuda_device="2,3"  # 指定使用的GPU
                       )
    
    return attackLM, targetLM

def load_indiv_model(model_name, local=False, use_jailbreakbench=True, cuda_device=None):
    if cuda_device is not None:
        # 设置当前函数作用域内的CUDA可见设备
        import os
        old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        
    try:
        if use_jailbreakbench: 
            if local:
                from jailbreakbench import LLMvLLM
                lm = LLMvLLM(model_name=model_name)
            else:
                from jailbreakbench import LLMLiteLLM
                api_key = get_api_key(Model(model_name))
                lm = LLMLiteLLM(model_name=model_name, api_key=api_key)
        else:
            if local:
                # 使用LocalvLLM类加载本地模型
                lm = LocalvLLM(model_name)
            else:
                lm = APILiteLLM(model_name)
    finally:
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
        
        self.model_name = Model(model_name)
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
            # force appending '<think>\n'
            init_message = '<think>\n'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message
        
    # def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
    #     batchsize = len(openai_conv_list)
    #     indices_to_regenerate = list(range(batchsize))
    #     valid_outputs = [None] * batchsize
    #     new_adv_prompts = [None] * batchsize
        
    #     # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
    #     for attempt in range(self.max_n_attack_attempts):
    #         # Subset conversations based on indices to regenerate
    #         convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
    #         # Generate outputs 
    #         outputs_list = self.model.batched_generate(convs_subset,
    #                                                     max_n_tokens = self.max_n_tokens,  
    #                                                     temperature = self.temperature,
    #                                                     top_p = self.top_p,
    #                                                     extra_eos_tokens=["}"]
    #                                                 )
            
    #         # Check for valid outputs and update the list
    #         new_indices_to_regenerate = []
    #         for i, full_output in enumerate(outputs_list):
    #             orig_index = indices_to_regenerate[i]
    #             full_output = init_message + full_output + "}" # Add end brace since we terminate generation on end braces
    #             attack_dict, json_str = extract_json(full_output)
    #             if attack_dict is not None:
    #                 valid_outputs[orig_index] = attack_dict
    #                 new_adv_prompts[orig_index] = json_str
    #             else:
    #                 new_indices_to_regenerate.append(orig_index)
            
    #         # Update indices to regenerate for the next iteration
    #         indices_to_regenerate = new_indices_to_regenerate
    #         # If all outputs are valid, break
    #         if not indices_to_regenerate:
    #             break

    #     if any([output is None for output in valid_outputs]):
    #         raise ValueError(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.")
    #     return valid_outputs, new_adv_prompts

    # def get_attack(self, convs_list, prompts_list):
    #     """
    #     Generates responses for a batch of conversations and prompts using a language model. 
    #     Only valid outputs in proper JSON format are returned. If an output isn't generated 
    #     successfully after max_n_attack_attempts, it's returned as None.
        
    #     Parameters:
    #     - convs_list: List of conversation objects.
    #     - prompts_list: List of prompts corresponding to each conversation.
        
    #     Returns:
    #     - List of generated outputs (dictionaries) or None for failed generations.
    #     """
    #     assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
    #     # Convert conv_list to openai format and add the initial message
    #     processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
    #     valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

    #     for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
    #         # For open source models, we can seed the generation with proper JSON and omit the post message
    #         # We add it back here
    #         if self.initialize_output:
    #             jailbreak_prompt += self.model.post_message
    #         conv.update_last_message(jailbreak_prompt)
        
    #     return valid_outputs
    
    
    
    
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        full_outputs = [None] * batchsize  # 存储完整的输出文本，而不仅仅是json_str
        
        attack_safety_settings = []
        if "gemini" in self.model.litellm_model_name:
            attack_safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]
        
        # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            # Generate outputs 
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        # extra_eos_tokens=["}"],
                                                        safety_settings=attack_safety_settings
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                
                if "ERROR:" in output:
                    # 如果API调用失败，直接将其标记为需要重新生成
                    new_indices_to_regenerate.append(orig_index)
                    continue
                
                if not output.strip().endswith('}'):
                    full_output = init_message + output + "}"
                else:
                    full_output = init_message + output
                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    full_outputs[orig_index] = full_output  # 存储完整的输出文本
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            # 改进错误信息，使其更有用
            failed_indices = [i for i, out in enumerate(valid_outputs) if out is None]
            error_msg = f"Failed to generate valid output after {self.max_n_attack_attempts} attempts for indices {failed_indices}. Terminating."
            raise ValueError(error_msg)
        
        return valid_outputs, full_outputs  # 返回完整的输出文本，而不是json_str

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
        valid_outputs, full_outputs = self._generate_attack(processed_convs_list, init_message)

        for full_output, conv in zip(full_outputs, convs_list):
            # 使用完整的输出文本更新最后一条消息
            # 对于开源模型，我们可以做一些后处理
            if self.initialize_output:
                full_output += self.model.post_message
            conv.update_last_message(full_output)
        
        return valid_outputs

class TargetLM():
    """
        JailbreakBench class for target language models.
    """
    def __init__(self,
            model_name: str,
            category: str,
            max_n_tokens : int,
            phase: str = None,
            evaluate_locally: bool = False,
            use_jailbreakbench: bool = True,
            cuda_device=None
            ):

        self.model_name = Model(model_name)
        # self.max_n_tokens = max_n_tokens # <- 不再直接使用这行
        self.phase = phase
        self.use_jailbreakbench = use_jailbreakbench
        self.evaluate_locally = evaluate_locally

        from config import TARGET_TEMP,  TARGET_TOP_P
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P
        
        # --- 开始修改 ---
        
        # 1. 根据模型动态设置 max_n_tokens
        MODEL_TOKEN_LIMITS = {
            # "gpt-o4-mini": 100000,
            "gpt-o4-mini": 10000,
            "gemini-2.5-pro": 64000,
            # "claude-4-sonnet": 64000,
            "claude-4-sonnet": 8000,
            "grok-3": 128000,
        }
        # 如果模型在我们的字典里，就用字典里的值，否则使用命令行传入的值作为后备
        self.max_n_tokens = MODEL_TOKEN_LIMITS.get(self.model_name.value, max_n_tokens)
        logger.info(f"为模型 {self.model_name.value} 设置 max_n_tokens 为: {self.max_n_tokens}")

        # 2. 根据模型动态设置 thinking 或 reasoning_effort 等特殊参数
        self.extra_params = {} # 创建一个字典来存储额外参数
        model_value = self.model_name.value

        if "gemini" in model_value or "claude" in model_value:
            # Gemini 和 Claude 使用 'thinking' 参数
            # thinking_budget = 30000 if "claude" in model_value else 32768
            thinking_budget = 4000 if "claude" in model_value else 32768
            self.extra_params['thinking'] = {"type": "enabled", "budget_tokens": thinking_budget}
            logger.info(f"为模型 {model_value} 启用 'thinking' 参数，预算为: {thinking_budget}")

        elif "grok" in model_value in model_value:
            # Grok 和 GPT-o4-mini 使用 'reasoning_effort' 参数
            self.extra_params['reasoning_effort'] = "high"
            logger.info(f"为模型 {model_value} 设置 'reasoning_effort' 参数为: 'high'")
            
        elif "gpt" in model_value in model_value:
            # Grok 和 GPT-o4-mini 使用 'reasoning_effort' 参数
            self.extra_params['reasoning_effort'] = "high"
            logger.info(f"为模型 {model_value} 设置 'reasoning_effort' 参数为: 'high'")

        # --- 修改结束 ---

        self.model = load_indiv_model(model_name, evaluate_locally, use_jailbreakbench=use_jailbreakbench, cuda_device=cuda_device)
        self.category = category
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def get_response(self, prompts_list: list[str]) -> list[str]:
        if self.use_jailbreakbench:
            # JailbreakBench 的逻辑保持不变
            llm_response = self.model.query(prompts=prompts_list,
                                             behavior=self.category,
                                             phase=self.phase,
                                             max_new_tokens=self.max_n_tokens)
            return llm_response.responses
        else:
            # --- API 模型的重试逻辑 ---
            
            MAX_ATTEMPTS = 1
            batch_size = len(prompts_list)
            
            # 初始化最终响应列表，用 None 作为未完成的标记
            final_responses = [None] * batch_size
            indices_to_process = list(range(batch_size))

            for attempt in range(MAX_ATTEMPTS):
                if not indices_to_process:
                    break # 所有 prompt 都已成功

                logger.info(f"TargetLM generation attempt {attempt + 1}/{MAX_ATTEMPTS}. Processing {len(indices_to_process)} prompts.")

                # 1. 准备本轮需要处理的 prompts
                prompts_this_iteration = [prompts_list[i] for i in indices_to_process]
                
                # 2. 将 prompts 转换为 API 需要的对话格式
                convs_list = [conv_template(self.template) for _ in range(len(prompts_this_iteration))]
                for conv in convs_list:
                    conv.system_message = "You are a helpful assistant."
                
                full_prompts = []
                for conv, prompt in zip(convs_list, prompts_this_iteration):
                    conv.append_message(conv.roles[0], prompt)
                    full_prompts.append(conv.to_openai_api_messages())

                # 3. 调用底层的 batched_generate 方法
                generated_responses = self.model.batched_generate(
                    full_prompts,
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    **self.extra_params
                )
                
                # 4. 处理返回结果，决定哪些需要重试
                indices_for_next_iteration = []
                for i, response in enumerate(generated_responses):
                    original_index = indices_to_process[i]

                    # 检查响应是否有效 (非空、非错误占位符)
                    if response and response.strip() and "ERROR:" not in response:
                        final_responses[original_index] = response
                    else:
                        logger.info(f"Received empty or error response for prompt at index {original_index}. Scheduling for re-generation.")
                        indices_for_next_iteration.append(original_index)

                # 5. 更新下一轮要处理的索引列表
                indices_to_process = indices_for_next_iteration
                
                if indices_to_process and attempt < MAX_ATTEMPTS - 1:
                    time.sleep(2) # 重试前短暂等待

            # 循环结束后，处理仍然失败的 prompts
            for i in range(batch_size):
                if final_responses[i] is None:
                    # final_responses[i] = "ERROR: Failed to generate a valid response after multiple attempts."
                    final_responses[i] = ""
                    logger.error(f"Prompt at index {i} failed after {MAX_ATTEMPTS} attempts.")

            return final_responses