使用 harmonyos-control MCP 工具完成 HarmonyOS 应用的编译、打包、安装和启动。

按以下顺序执行：

1. **构建**：调用 `mcp__harmonyos-control__hv_build`，参数：
   - `project`: 当前项目根目录路径
   - `module`: "entry"

2. **安装**：构建成功后，调用 `mcp__harmonyos-control__hv_install`，参数：
   - `hapPath`: 从构建输出中获取 HAP 文件路径（通常在 `entry/build/default/outputs/default/` 下）

3. **启动**：安装成功后，调用 `mcp__harmonyos-control__hv_start`，参数：
   - `project`: 当前项目根目录路径

每一步都要检查返回结果，如果失败则停止并报告错误。全部成功后输出简洁的完成状态。

如果构建失败，先分析错误信息，尝试修复代码问题后重新构建（最多重试 2 次）。
