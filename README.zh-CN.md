# Task-Scoped Physical Undo

这个仓库围绕一个问题：**撤销过去的一项任务，同时保护后来发生的有效改变。**

这是适合放到 GitHub 的精简版本。分层源码、任务示例、测试、运行说明和论文工作稿都在这里；开发过程中的大量脚本、日志、模型权重和仿真资产留在原开发仓库。

## 建议的阅读和运行顺序

1. 先按 [英文 README](README.md) 安装本地包。
2. 运行 `python examples/history_contrast.py`，理解“当前状态相同，历史不同，目标不同”。它使用手写事实，不调用模型，也不运行仿真。
3. 看 [任务规范](docs/TASK_SPEC.md)，再读 `src/task_undo/domain/policies.py`。
4. 连接模型服务，使用 `task-undo-predict` 比较直接目标与事实＋规则。后者仍经过两次语言模型调用。
5. 按 [仿真说明](docs/SIMULATION.md) 生成实际物理历史，再运行撤销与独立评分。

任务输入、运行输出和模型提示词全部使用英文。代码注释可以使用中文。README 中的示例不能计作新的实验成绩；当前源码也不能替代论文中各次实验冻结的源码和原始记录。

## 核心目录

| 目录 | 作用 |
| --- | --- |
| `src/task_undo/domain` | 根据历史定义恢复与保护边界 |
| `src/task_undo/agents` | 模型推断与事实读出 |
| `src/task_undo/application` | 组织命令、执行循环与源代码快照 |
| `src/task_undo/simulation` | 真正执行导航、抓取、放置、释放和退离 |
| `src/task_undo/evaluation` | 用独立参考目标检查实际轨迹 |
| `tests` | 不需要 GPU 的软件回归 |
| `paper/manuscript.pdf` | 当前六页英文工作稿 |
