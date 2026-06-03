import sys
import ollama

class LocalTranslator:
    def __init__(self, model_name: str = "gemma3:1b"):
        """
        初始化翻譯引擎
        :param model_name: 本地 Ollama 中的模型名稱，例如 qwen2.5:7b, deepseek-r1:7b 等
        """
        self.model_name = model_name
        
        # 核心：強力的系統提示詞，杜絕模型說廢話
        self.system_prompt = (
            "You are a professional, high-quality translation engine.\n"
            "Task: Translate English to Chinese.\n"
            "Rules:\n"
            "1. ONLY output the direct translation results. Do NOT include any explanations, "
            "introductory words, notes, or markdown formatting blocks.\n"
            "2. Maintain the original professional tone and formatting."
        )

    def translate(self, text: str, stream: bool = True) -> str:
        """
        執行翻譯
        :param text: 待翻譯的文本
        :param stream: 是否開啓流式輸出（打字機效果）
        """
        if not text.strip():
            return ""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Translate the following text:\n{text}"}
        ]

        try:
            if stream:
                full_response = ""
                # 使用 ollama.chat 的流式傳輸
                response_stream = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.3} # 較低的隨機性保證翻譯穩定
                )
                
                for chunk in response_stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    full_response += content
                print() # 換行
                return full_response
            else:
                # 單次直接返回
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options={"temperature": 0.3}
                )
                return response['message']['content']

        except ollama.ResponseError as e:
            print(f"\n[錯誤] Ollama 響應異常，請檢查模型名稱是否正確: {e}", file=sys.stderr)
        except Exception as e:
            print(f"\n[錯誤] 無法連接到 Ollama 服務，請確保 Ollama 已在後台運行: {e}", file=sys.stderr)
        return ""

if __name__ == "__main__":
    # 換成你本地已下載的模型名稱，如 "qwen2.5:7b" 或 "deepseek-r1:7b"
    MODEL = "gemma3:1b" 
    
    print(f"正在加載本地翻譯引擎 (模型: {MODEL})...")
    translator = LocalTranslator(model_name=MODEL)
    print("引擎加載成功！請輸入需要翻譯的文本（輸入 'exit' 或 'quit' 退出）：\n" + "-"*50)

    while True:
        try:
            user_input = input("\n請輸入文本 > ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("已退出翻譯引擎。")
                break
                
            if not user_input:
                continue

            print("翻譯結果 > ", end="")
            translator.translate(user_input, stream=True)
            
        except (KeyboardInterrupt, EOFError):
            print("\n已退出翻譯引擎。")
            break
