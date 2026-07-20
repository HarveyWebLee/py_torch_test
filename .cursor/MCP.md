# MCP 配置说明

本项目的 MCP 配置位于 [mcp.json](mcp.json)。

当前默认为空配置。可按需在 Cursor Settings → MCP 中添加服务器，例如：

| 用途       | 示例                                       |
| ---------- | ------------------------------------------ |
| 文档检索   | `user-docs-langchain`（LangChain 文档）    |
| 浏览器调试 | `cursor-ide-browser`                       |
| Git 操作   | `user-eamodio.gitlens-extension-GitKraken` |

添加后同步更新此文件，便于团队共享。MCP 密钥与 token 不要提交到仓库。
