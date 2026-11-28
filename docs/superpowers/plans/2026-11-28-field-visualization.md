# 引斥场可视化实现计划

**Goal:** 为现有控制台提供场方向箭头、顶部菜单、白色/深色主题、自动保存演示录制与实验目录隔离，在不中断当前算法的前提下推送 GitHub。

**Architecture:** `visual/field.js` 计算真实场梯度和屏幕抽样；`map.js` 负责 Canvas，`theme.js` 提供语义配色与持久化，`app.js` 连接菜单、图层、主题和录制开关。`recording.js` 捕获实际帧并交给 VisualServer 转码保存。main.py 在初始化输出和日志之前应用实验名。沿用现有数据与自动录制协议。

**Tech Stack:** 原生 JavaScript ES modules、Canvas 2D、CSS、pytest/Node、Playwright CLI。

- [x] 添加并运行梯度、掩膜、边缘、采样、物理间距与绘图验证，确认缺失实现导致失败后实现。
- [x] 添加 `visual/field.js`，实现梯度、有效性检查和防窄边界遗漏的屏幕抽样。
- [x] 接入独立方向开关、当前帧渲染、近景相机、固定场值色标、半径图例与缺场提示。
- [x] 将数据、视图、播放和录制操作移入顶部菜单，支持互斥展开、键盘和移动端。
- [x] 完成默认白色与深海蓝主题、语义配色、选择持久化和录制主题同步。
- [x] 接入录制开关、定时/手动停止自动保存、失败清理与重试、保存路径和下载状态。
- [x] 添加 experiment-name，隔离输出、TensorBoard 与缓存，验证目录名并保持旧调用兼容。
- [x] 使用独立服务在 Playwright 检查真实 coverage/tracking 回放、双主题、菜单、开关、回放进退、缺场数据与手机布局；查看截图并修复问题。
- [x] 通过真实浏览器录制开关保存并读取 H.264 MP4，未停止现有算法或 TensorBoard。
- [x] 完成独立代码审查，修复窄边界方向抽样与小数 CSS 尺寸画布清理问题，更新文档。
- [x] 完成组合回归：117 通过、1 项 Windows 符号链接权限检查跳过；7 个 JavaScript 模块语法通过，差异无空白错误。

交付：按用户要求提交并推送当前 GitHub main 分支，核对远端提交及训练进程状态；不提交用户原有的 `docs/formal-training-validation.md` 修改。
