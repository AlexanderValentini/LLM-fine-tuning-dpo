# LLM-fine-tuning-dpo
Tutor chatbots are projects under active development: even nowadays, finding such an artificial intelligence to answer our science-related problems is a challenging task. Large language models (LLMs) such as ChatGPT or Gemini help in answering our questions, however the accuracy of the generated answers can still be improved. This is expected, as these LLMs were not trained to answer science-related questions specifically.
To deal with this, I have undertaken a project to build an artificial intelligence specifically tailored for this task. This was done by incorporating supervised fine-tuning on Multiple Choice Question Answer data and doing DPO alignment of existing LLM's. Lastly Activation-aware Weight Quantization was used to compress model size and speed up inference. The base model used was the "stablelm-zephyr-3b" model by stabilityai, which was sourced from Huggingface.


