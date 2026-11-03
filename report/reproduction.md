# 工具版本、构建与复现

## 1. 版本选择

本提交沿用 **Vivado 2017.4、XC7Z100-2FFG900、DNNDK 2.05、ARMv7 hard-float**。采用旧版的原因是既有板厂显示工程、DPU IP、32位 ARM 运行库与板端 Linux 4.14.0-xilinx / Debian 9 的配套关系。保持版本用于复现既有位流和模型接口；没有验证直接升级工具后的 IP、ABI 与时序等价性。

竞赛指南允许报告说明后使用其他版本，因此本包不依靠静默升级规避版本说明。复现时记录 Vivado、交叉编译器、Python、训练库与实际板卡版本。

## 2. 获取代码和依赖

1. 从公开仓库获取源码，或解压本地完整 RAR，使用英文且无空格的工作目录。
2. 使用支持该器件的 Vivado 2017.4，以及合法取得的 DPU IP、DNNDK ARM 头文件和运行库。
3. 本地包在 `local_dependencies/` 保存现有工程依赖；公开仓库不分发授权 IP。公开仓库用户需从板厂配套资料及 AMD 对应工具渠道取得同版本依赖。
4. 保留环境路径映射。历史配置中的 E/F 盘是原机器的记录，不是新机器默认路径；以下入口通过参数指定源码和输出位置。

## 3. 主机离线检查

仅运行新增通用技能需要 Python 3 标准库；既有启动和数据选择测试另需 NumPy。模型训练依赖见 `config/soundjie_d4_host_requirements.txt`，该文件中的 NumPy、SciPy、PyTorch 版本应使用相容的 Python 环境安装。训练环境与 ARM 编译环境可以分开准备。

```sh
python -m unittest discover -s skill/tests -v
python -m unittest tests.test_soundjie_autostart tests.test_soundjie_alert5bg_probes -v
```

完整哈希校验和时序报告检查的命令见 [技能包](../skill/README.md)。报告存在、脚本通过与硬件运行通过属于不同阶段，检查结果分别保存。

## 4. ARM 软件构建

Linux 环境需有 `arm-linux-gnueabihf-g++`、`arm-linux-gnueabihf-readelf`、bash、GNU coreutils 与匹配的 DNNDK 库。使用仍不存在的新输出目录，避免混入上次构建结果。具体调用见 [docs/BUILD.md](../docs/BUILD.md)。四个构建入口为：

- `scripts/guest/build_soundjie_d4_runtime.sh`
- `scripts/guest/build_soundjie_d4_classifier.sh`
- `scripts/guest/build_soundjie_hdmi_ui.sh`
- `scripts/guest/build_xvf3800_doa_tool.sh`

编译脚本检查 ARMv7、硬浮点 ABI、动态加载器和依赖信息。六输出模型使用 `models/release/dpu_soundjie_alert5bg.elf`，分类器参数为 physical/storage outputs 均 6、kernel `soundjie_alert5bg`。不要以脚本历史默认值代替本版本参数。

## 5. PL 与现有板端包

本地完整包的 `LOCAL_BUILD.md` 给出 PL 源码落位和项目路径。恢复完整板厂工程结构后，先只读核对器件、IP 路径、BD、约束和外部引用。联合工程已包含 DPU，应使用既有 Tcl 的 `resume/report/export` 路线而非再次插入 IP。保存综合和实现报告，逐项核对同步路径与未约束 I/O。

`local_dependencies/deployment/SJ6V1.TGZ` 与 `boot/` 保持原版本；它们不是整卡镜像。启动还依赖板厂配套的 BOOT.bin、内核和根文件系统。先校验哈希、硬件型号和内核 ABI，再按板厂操作流程部署；不要把历史磁盘号当作当前 SD 卡号。

## 6. 验收顺序和停止条件

按“文件校验 → 主机测试 → ARM 构建与 ABI 检查 → PL 实现检查 → 板端设备初始化 → 输入/推理/显示联合检查”逐级推进。出现模型哈希、器件型号、ABI、关键时序或存储错误时停止该阶段，不跳过错误继续写卡。

历史测试与结果入口见 [verification.md](verification.md)。未包含原始音频的验证需要依据数据来源和划分配置重新取得数据；源码包不意味着自带所有第三方训练材料。本次只对新增材料执行主机检查，没有重新进行全链路从零综合或板测。
