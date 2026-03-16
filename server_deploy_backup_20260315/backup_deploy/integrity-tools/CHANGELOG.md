# Integrity Tools 更新日志

## [1.0.1] - 2026-03-14

### Bug 修复

- **修复 debate.html JavaScript 语法错误**
  - 删除第 168-173 行残留的 `catch` 代码块
  - 解决 `startDebate is not defined` 错误

- **修复 PDF 模块 404 错误**
  - 在 `app/__init__.py` 中添加 PDF 蓝图注册
  - `/api/pdf/*` 端点现已可用

### API 验证

- ✅ `/api/tokens/calc` - Token 计算正常
- ✅ `/api/tokens/compare` - 多模型对比正常
- ✅ `/api/compare/providers` - 提供商列表正常
- ✅ `/api/compare` - AI 对比正常
- ✅ `/api/debate/config` - 辩论配置正常
- ✅ `/api/debate/start` - 辩论开始正常
- ✅ `/api/pdf/info` - PDF 信息正常
- ✅ `/api/pdf/images_to_pdf` - 图片转 PDF 正常
- ✅ `/api/pdf/merge` - PDF 合并正常
- ✅ `/api/pdf/remove_pages` - 删除页面正常
- ✅ `/api/lines/*` - 台词学习正常

---

## [1.0.0] - 2026-03-14

### 新增功能

- **AI 辩论赛** - 多模型驱动的AI辩论竞技场
  - 支持 Qwen、Doubao、Kimi、DeepSeek 四大模型
  - SSE 流式输出辩论过程
  - Edge TTS 语音合成
  - 完整的辩论流程（开篇立论、攻辩质询、自由辩论、总结陈词、裁判点评）

- **Token 计算** - 不同模型的 Token 消耗和成本计算
  - 支持 Qwen、GPT-4o、DeepSeek 等模型
  - 多模型成本对比

- **AI 模型对比** - 同一问题横向对比多个大模型
  - 并发调用多个 API
  - 实时显示响应时间和结果

- **台词学习** - 扇贝单词 PDF 学习工具
  - PDF 上传和学习数据管理

- **PDF 工具集** - 在线 PDF 处理工具
  - 图片转 PDF
  - PDF 合并
  - 删除页面
  - 页面重排（竖版在前）

- **认证系统**
  - 用户注册/登录
  - JWT Token 认证
  - 邀请码机制

### 架构设计

- GitHub Pages (HTTPS) + 云服务器 (HTTP) 混合架构
- Flask + Gunicorn + SQLite 技术栈
- WireGuard VPN 保留

### Bug 修复

- 修复 Gunicorn 多 worker 模式下全局变量不共享问题（改为单 worker）
- 修复 SSE 时序问题（先连接再启动辩论）
- 修复辩论状态卡在 running 问题（添加超时自动重置）
- 修复前端错误提示不显示问题
- 删除废弃的 docs/server 目录

### 文档

- 新增 DEPLOY.md 部署文档
- 更新源码跳转链接

---

## 开发历史

### 2026-03-14

- 完成从零部署 integrity-tools 项目
- 整合 5AI辩论赛、18tokens、2台词、17ai-compare 四个工具
- 新增 PDF 工具集
- 修复辩论赛 SSE 相关问题
- 完善部署文档