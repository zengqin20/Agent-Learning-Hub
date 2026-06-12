import requests
import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class LLMClient:
    """LLM API 调用客户端"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API 密钥，优先级：参数 > 环境变量 > .env 文件
            base_url: API 基础 URL
            model: 默认使用的模型名称
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.default_model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            raise ValueError("请提供 API 密钥或设置 OPENAI_API_KEY 环境变量/在 .env 文件中配置")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def chat(self, 
             messages: list, 
             model: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表，格式如 [{"role": "user", "content": "你好"}]
            model: 使用的模型名称，默认使用初始化时指定的模型
            temperature: 温度参数，控制随机性 (0-1)
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
            
        Returns:
            API 响应字典
        """
        url = f"{self.base_url}/chat/completions"
        
        # 使用传入的 model 或默认 model
        use_model = model or self.default_model
        
        # 从环境变量读取 temperature
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        # 处理 max_tokens
        if max_tokens is None:
            max_tokens_env = os.getenv("MAX_TOKENS")
            if max_tokens_env:
                max_tokens = int(max_tokens_env)
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"错误详情: {e.response.text}")
            raise
    
    def simple_chat(self, 
                   user_input: str, 
                   system_prompt: Optional[str] = None,
                   model: Optional[str] = None,
                   **kwargs) -> str:
        """
        简化版聊天接口
        
        Args:
            user_input: 用户输入
            system_prompt: 系统提示（可选）
            **kwargs: 传递给 chat 方法的其他参数
            
        Returns:
            AI 回复的文本内容
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_input})
        
        response = self.chat(messages, model=model, **kwargs)
        
        # 提取回复内容
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError("API 响应格式异常")
    
    def stream_chat(self,
                   messages: list,
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   **kwargs):
        """
        流式聊天
        
        Args:
            messages: 消息列表
            model: 使用的模型
            temperature: 温度参数
            **kwargs: 其他参数
            
        Yields:
            逐块生成的内容
        """
        url = f"{self.base_url}/chat/completions"
        
        # 使用传入的 model 或默认 model
        use_model = model or self.default_model
        
        # 从环境变量读取 temperature
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        try:
            # with 表达式 as 变量:上下文管理器（Context Manager）语法 用于自动管理资源的分配和释放。
            with self.session.post(url, json=payload, stream=True) as response:
                #检查 HTTP 状态码
                response.raise_for_status()

                # 逐行读取
                for line in response.iter_lines():
                    # 空行跳过
                    if line:
                        #字节流转为字符串
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            # 检测结束信号
                            if data_str.strip() == '[DONE]':
                                break
                            
                            # 解析 JSON
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                     # 生成器返回内容片段
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except requests.exceptions.RequestException as e:
            print(f"流式请求失败: {e}")
            raise


def main():
    """主函数 - 交互式对话示例"""
    print("=== LLM API 对话客户端 ===")
    print("输入 'quit' 或 'exit' 退出\n")
    
    # 初始化客户端（自动从 .env 文件和环境变量读取配置）
    try:
        client = LLMClient()
        print(f"✓ 已加载配置:")
        print(f"  - 模型: {client.default_model}")
        print(f"  - API: {client.base_url}")
        print()
    except ValueError as e:
        print(f"初始化失败: {e}")
        print("\n请配置环境变量：")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 在 .env 文件中填入你的 API 密钥和配置")
        return
    
    # 对话循环
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"}
    ]
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见!")
            break
        
        if not user_input:
            continue
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 获取 AI 回复 flush=True 强制刷新立马输出 、end="" 避免换行
            print("AI: ", end="", flush=True)
            
            # 使用流式输出
            full_response = ""
            for chunk in client.stream_chat(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # 换行
            
            # 添加 AI 回复到历史
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            print(f"\n对话出错: {e}")


if __name__ == "__main__":
    main()