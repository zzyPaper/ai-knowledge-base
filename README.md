# AI 知识库 — 驾驭 AI 开发

> **一句话**：AI 已经很聪明了，问题不是让它更聪明，而是怎么让它稳定地把事做对。

---

## 这个大图是怎么来的

读了一堆腾讯云开发者和字节的 Harness Engineering 文章后，发现所有观点都指向同一个方向：

```
怎么问 AI（Prompt Engineering）
  → 给 AI 什么信息（Context Engineering）
    → 怎么管住 AI 让它把事做对（Harness Engineering）
```

现在卡在第 3 步。模型能力已经够用，缺的是**管住 AI 的制度**。

### 快速类比

| 阶段 | 你在做什么 | 类比 |
|------|-----------|------|
| Prompt Engineering | 把需求说清楚 | 跟服务员说清要点什么菜 |
| Context Engineering | 背景资料准备好 | 给新员工看项目文档 |
| Harness Engineering | 建制度、加检查、防跑偏 | 定流程、设 Code Review、跑 CI |

---

## 六个模块一览

### 1. Harness — 驾驭工程
**AI 是野马，Harness 是缰绳 + 马鞍 + 赛道 + 裁判。**
模型之外的整套基础设施：规则（Rule）、标准流程（Skill）、验收脚本（Scripts）、角色分工（Sub Agent）、工作流（Workflow）、外部接口（MCP）。核心：**AI 说做完不算，脚本判通过才算。**
→ [01-Harness.md](01-Harness.md)

### 2. Agent — 智能体
**别把 AI 当助手，要当一支必须制度化管理的团队。**
单 Agent 有角色冲突问题，复杂的任务需要拆成：PM、需求分析、方案设计、闸门总控、开发实现、代码审查、测试验证。每个角色各司其职，流程交接清楚。
→ [02-Agent.md](02-Agent.md)

### 3. Context — 上下文工程
**与其祈祷 AI 自己想明白，不如主动把信息给到位。**
AI 窗口有限，项目信息无限。关键方法：先给目录索引（AGENTS.md 约 100 行），需要时再展开细节（渐进式披露）。规则必须沉淀到仓库，规范要可验证。
→ [03-Context.md](03-Context.md)

### 4. Agentic Engineering — 人机协作方法论
**工程师掌舵，AI 划船。**
信息每传一步就损耗一次，所以要分步验证。大任务拆小步，错误经验沉淀成 Rule/Skill。知识当代码管理，跟仓库一起版本化。
→ [04-Agentic-Engineering.md](04-Agentic-Engineering.md)

### 5. AI 辅助软件 — 产品化形态
**Harness/Agent/Context 的理念在这些产品里已经落地了。**
Claude Code 内置 8 种 Harness 模式，Managed Agents 解耦大脑-双手-工作台，Hermes 能从错误中进化。Cursor 验证了多 Agent 实践。MCP 是对接外部系统的标准接口。
→ [05-AI辅助软件.md](05-AI辅助软件.md)

### 6. 大模型 — 推理底座
**LLM 路线 vs 世界模型路线的深层分歧，以及对整个知识库的根本性挑战。**

核心问题：当前 Harness/Agent/Context 工程是否是在沙地上盖楼？

关键洞察：
- LLM 擅长符号操作，不擅长物理世界建模
- Context Engineering 优化的是有限游戏
- Harness 解决执行问题，不解决认知问题
- 架构解耦与开源透明是两种路线的共同指向

→ [06-大模型.md](06-大模型.md)

---

## 提炼出的规则体系

从这些文章里提取的核心约束，整理成了一套 **AI 约束规范**，专门给我（Claude）用的：

```
rules/
├── 00-rules-for-claude.md    ← 总纲：所有约束的索引和执行优先级
├── 01-harness-rules.md        ← Harness：软约束+硬门禁+反馈闭环
├── 02-agent-rules.md          ← Agent：角色分工+流程交接+质量门禁
├── 03-context-rules.md        ← 上下文：信息供给+渐进式披露+记忆管理
├── 04-iteration-rules.md      ← 迭代准则：不做多余假设+最小改动+尊重原有逻辑
├── 05-coding-rules.md         ← 编码准则：简单+命名+错误处理+重构规范
└── 06-ecc-guide.md            ← ECC 集成：如何与 ECC 规则配合使用
```

每一条规则都是可直接执行的，不是空话。详见 `rules/` 目录。

---

## 已收录文章来源

| # | 标题 | 来源 | 覆盖模块 |
|---|------|------|---------|
| 1 | 来自字节跳动TRAE的HARNESS ENGINEERING指南 | AI技术立文 | Harness, Agent, Context |
| 2 | Harness Engineering 即控制论 | 邬俊杰/腾讯云开发者 | Harness, Agent, Context |
| 3 | 万字干货！Harness Engineering 如何工程化落地 | 白家杰/腾讯云开发者 | Harness, Agent, Context, 产品 |
| 4 | 深入浅出Harness Engineering之核心模式与理念 | 张碧泉/腾讯云开发者 | Harness, Agent, Context, 产品 |
| 5 | 从第一性原理思考 Agentic Engineering | 魏依承/腾讯云开发者 | Agentic Engineering |
| 6 | RAG已死？不，是GREP回归了！ | 何理扬/腾讯云开发者 | Context, 产品 |
| 7 | 程序员越早想通这些越好 | 吴正伟/腾讯云开发者 | 编码准则 |
| 8 | AI Infra 其实没有多少新东西 | 腾讯云开发者 | AI基金/硬件栈 |
| 9 | 大模型的Agent Skill功能，在LLM HTTP底层交互流中是怎么承载的？ | 张敏/腾讯云开发者 | 大模型 |
