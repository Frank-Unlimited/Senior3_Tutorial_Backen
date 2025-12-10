#!/bin/sh
set -e

# 如果 settings.yaml 不存在，从环境变量生成
if [ ! -f /app/settings.yaml ]; then
    echo "Generating settings.yaml from environment variables..."
    
    cat > /app/settings.yaml << EOF
vision_model:
  provider: ${VISION_PROVIDER:-doubao}
  model_name: ${VISION_MODEL:-doubao-1-5-vision-pro-32k-250115}
  api_key: ${VISION_API_KEY:-${DOUBAO_API_KEY:-}}
  api_base: ${VISION_API_BASE:-https://ark.cn-beijing.volces.com/api/v3}
  temperature: ${VISION_TEMPERATURE:-0.3}
  max_tokens: ${VISION_MAX_TOKENS:-2048}

deep_thinking_model:
  provider: ${DEEP_PROVIDER:-doubao}
  model_name: ${DEEP_MODEL:-doubao-1-5-pro-32k-250115}
  api_key: ${DEEP_API_KEY:-${DOUBAO_API_KEY:-}}
  api_base: ${DEEP_API_BASE:-https://ark.cn-beijing.volces.com/api/v3}
  temperature: ${DEEP_TEMPERATURE:-0.7}
  max_tokens: ${DEEP_MAX_TOKENS:-8192}

quick_model:
  provider: ${QUICK_PROVIDER:-doubao}
  model_name: ${QUICK_MODEL:-doubao-1-5-lite-32k-250115}
  api_key: ${QUICK_API_KEY:-${DOUBAO_API_KEY:-}}
  api_base: ${QUICK_API_BASE:-https://ark.cn-beijing.volces.com/api/v3}
  temperature: ${QUICK_TEMPERATURE:-0.5}
  max_tokens: ${QUICK_MAX_TOKENS:-1024}

persona_prompt: |
  你是一位温柔的大姐姐，擅长辅导高三学生的生物学习。
  你的特点是：
  - 说话温柔有耐心，经常用"呢"、"哦"、"呀"等语气词
  - 善于鼓励学生，即使学生答错也会先肯定他们的思考
  - 解释问题时会用生动的比喻和例子
  - 会关心学生的学习状态和情绪
EOF

    echo "settings.yaml generated successfully!"
fi

# 执行传入的命令
exec "$@"
