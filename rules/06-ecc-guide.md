# ECC 集成指南

> 把 AI 知识库的约束规则和已有的 ECC（Engineering Coding Convention）规则体系对齐。

---

## ECC 已有的规则（已安装）

当前在你的 `~/.claude/rules/ecc/` 中已安装：

### common/（通用，始终加载）

| 文件 | 覆盖内容 |
|------|---------|
| coding-style.md | 不可变性、KISS/DRY/YAGNI、文件组织、错误处理、命名规范 |
| git-workflow.md | 提交格式、PR 流程 |
| testing.md | 80% 覆盖、TDD、AAA 模式 |
| performance.md | 模型选择、上下文管理、扩展思考 |
| patterns.md | Repository 模式、API 响应格式 |
| hooks.md | PreToolUse/PostToolUse、TodoWrite |
| agents.md | 可用 Agent 列表、并行执行 |
| security.md | 安全检查清单、密钥管理 |
| development-workflow.md | 完整开发流程、研究复用、代码审查 |
| code-review.md | 审查流程、严重等级、常见问题 |

### csharp/
| coding-style.md | hooks.md | patterns.md | security.md | testing.md |

### python/
| coding-style.md | hooks.md | patterns.md | security.md | testing.md |

### lua/
| coding-style.md | hooks.md | patterns.md | security.md | testing.md |

---

## 知识库规则 vs ECC 规则对照

### 重复/重叠的规则

以下规则在两边都有，**以 ECC 为准**（因为 ECC 是系统级加载）：

| 知识库规则          | ECC 对应                                  | 处理         |
| -------------- | --------------------------------------- | ---------- |
| 80% 覆盖测试       | `common/testing.md`                     | ❌ 重复，用 ECC |
| 先写测试（TDD）      | `common/testing.md` + `tdd-guide` agent | ❌ 重复，用 ECC |
| KISS/DRY/YAGNI | `common/coding-style.md`                | ❌ 重复，用 ECC |
| 错误处理           | `common/coding-style.md`                | ❌ 重复，用 ECC |

### 互补的规则

知识库规则补充 ECC 没有覆盖的部分：

| 知识库规则 | 覆盖 ECC 空白 | 建议 |
|-----------|-------------|------|
| 🔥 Scripts 硬门禁 | ECC 没提到「AI 说做完不算」 | **补充到 ECC** |
| 🔥 不做多余假设 | ECC 没提到迭代中的假设问题 | **补充到 ECC** |
| 🔥 最小化改动 | ECC 没提到新增字段成本 | **补充到 ECC** |
| 🔥 尊重原有时序 | ECC 没提到重构时序风险 | **补充到 ECC** |
| Agent 角色分工 | ECC 的 agents.md 只有 agent 列表 | **补充到 ECC** |
| 上下文渐进式披露 | ECC 没提到上下文管理 | **补充到 ECC** |
| 规则必须可验证 | ECC 的规则偏原则性 | **补充到 ECC** |
| 代码是负债 | `common/coding-style.md` 可扩展 | **补充到 ECC** |

---

## 建议补充到 ECC 的规则

如果你想把这些新规则正式加入 ECC，建议以下文件新增内容：

### common/coding-style.md 增加

```markdown
## AI 迭代约束

### 不做多余假设
- 只处理需求明确说到的场景，不自行脑补边界
- 不确定就问，不假设业务上下文
- 不了解时倾向于少做

### 最小化改动
- 新增字段 = 增加复杂度，非必要不加
- 改接口是最昂贵的操作
- 改动范围集中可控

### 尊重原有时序
- 能跑通就不重构时序
- 新增逻辑插在已有流程的缝隙中
- 一条路走到底，不开辟平行路径
```

### common/agents.md 增加

```markdown
## Agent 协作规则

- 下游 Agent 不能改上游输出，只能提阻塞项
- 完成必须有基线验证
- PM 只做路由不做专业判断
- 结构化调度优于自由讨论
- 凭证不进沙盒
```

### common/testing.md 增加

```markdown
## 验收标准

- AI 说做完不算，脚本判通过才算
- 开发前后各跑一次基线验证脚本
- 可判定的约束优先写成 Scripts
```

---

## 执行优先级

当规则冲突时：

```
用户明确指令 > ECC 系统规则 > 知识库自定义规则 > 默认行为
```

| 场景 | 优先级谁高 |
|------|-----------|
| ECC 编码规范 vs 编码准则 | ECC 高（因为是系统级） |
| ECC 测试规范 vs Scripts 规则 | ECC 高 |
| ECC 说「函数 < 50 行」vs 编码准则说「函数 < 40 行」 | ECC 高 |
| Harness 规则（知识库独有） vs AI 默认行为 | 知识库规则高 |
| Agent 角色分工（知识库独有） vs AI 默认行为 | 知识库规则高 |

---

## 总结

**ECC 是基础规范**（编码、测试、安全、Git 等通用工程实践）
**知识库规则是 AI 开发专项约束**（Harness、Agent、Context、迭代方式）

两者互补，不冲突。当两者都覆盖同一议题时，以 ECC 为准。
