# 游戏系列RAG聊天机器人

这是一个专注于《杀手》游戏系列的RAG（检索增强生成）聊天机器人。它结合了语义检索和词法检索，并通过章节感知的重排序技术，提供附带相关原文片段引用的答案。

## 功能特点

- **混合检索器**：使用RRF（逆序融合）结合FAISS语义搜索和BM25词法搜索
- **章节感知重排序**：优先考虑“游戏评价”、“争议事件”、“游戏玩法”等关键章节，以提高答案相关性
- **大语言模型集成**：使用Google Gemini API基于检索到的文档生成自然语言回答
- **答案附带引用**：回答中清晰引用检索到的文档
- **Streamlit聊天界面**：用于测试和查询RAG系统的交互式网页界面

## 安装步骤

1.  克隆此仓库
2.  创建Python虚拟环境并激活它
3.  使用 `requirements.txt` 安装依赖包
4.  将您的Gemini API密钥设置为环境变量：
    - Linux/macOS: `export GEMINI_API_KEY="your_api_key"`
    - Windows: `set GEMINI_API_KEY=your_api_key`

## 使用方法

在 `backend` 目录下运行命令 `streamlit run app.py` 启动Streamlit聊天机器人，即可提出任何与《杀手》游戏系列相关的问题。
