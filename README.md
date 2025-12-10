# Senior3 Tutorial Backend - 高中生物智能辅导系统（后端）

基于 FastAPI + LangChain 的高中生物错题辅导系统后端，采用"温柔大姐姐"人设。

## 功能特点

- 📸 **视觉理解**: 使用视觉模型提取错题图片中的题目内容
- 🧠 **深度解答**: 深度思考模型生成详细解答过程
- 📊 **考察点分析**: 快速总结题目考察的知识点
- 🔗 **逻辑链梳理**: 整理解题思路和逻辑链
- 💬 **个性化辅导**: 支持引导式和直接解答两种辅导方式
- 🌸 **温柔人设**: 温柔大姐姐风格的交互体验
- ⚡ **异步并行**: 使用 LangChain RunnableParallel 实现并行处理
- 📡 **实时推送**: SSE 实时推送任务状态

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

```bash
cp settings.example.yaml settings.yaml
# 编辑 settings.yaml 配置你的 API Key
```

### 运行

```bash
python main.py
```

访问 http://localhost:8000

API 文档: http://localhost:8000/docs

## Docker 部署

### 构建镜像

```bash
docker build -t biotutor-backend .
```

### 运行容器

```bash
docker run -d -p 8000:8000 \
  -v ./settings.yaml:/app/settings.yaml \
  biotutor-backend
```

## 配置说明

`settings.yaml` 示例：

```yaml
vision_model:
  provider: doubao
  model_name: doubao-1-5-vision-pro-32k-250115
  api_key: your-api-key
  api_base: https://ark.cn-beijing.volces.com/api/v3
  temperature: 0.3
  max_tokens: 2048

deep_thinking_model:
  provider: doubao
  model_name: doubao-1-5-pro-32k-250115
  api_key: your-api-key
  api_base: https://ark.cn-beijing.volces.com/api/v3
  temperature: 0.7
  max_tokens: 8192

quick_model:
  provider: doubao
  model_name: doubao-1-5-lite-32k-250115
  api_key: your-api-key
  api_base: https://ark.cn-beijing.volces.com/api/v3
  temperature: 0.5
  max_tokens: 1024

persona_prompt: |
  你是一位温柔的大姐姐，擅长辅导高三学生的生物学习。
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/session | 创建会话 |
| POST | /api/session/{id}/image | 上传图片 |
| POST | /api/session/{id}/message | 发送消息 |
| POST | /api/session/{id}/chat | 普通聊天 |
| GET | /api/session/{id}/events | SSE 事件订阅 |
| GET | /api/session/{id}/status | 获取状态 |

## 项目结构

```
├── api/           # API 路由
├── config/        # 配置管理
├── models/        # 数据模型
├── prompts/       # Prompt 模板
├── session/       # 会话管理
├── sse/           # SSE 发布
├── utils/         # 工具函数
├── workflow/      # LangChain 工作流
│   └── chains/    # 各种 Chain
├── main.py        # 入口文件
└── settings.yaml  # 配置文件
```

## 技术栈

- FastAPI
- LangChain
- Python 3.10+
- Pydantic
- SSE (Server-Sent Events)

## 相关仓库

- 前端仓库: [Senior3_Tutorial](https://github.com/Frank-Unlimited/Senior3_Tutorial)

## License

MIT
