# 构建与运行说明

## 平台

| 部分 | 环境 |
|---|---|
| 器件 | MZ7100FA / XC7Z100-2FFG900 |
| PL 工具 | Vivado 2017.4 |
| 板端 | ARMv7 hard-float，Linux 4.14.0-xilinx，Debian 9 |
| AI 部署 | DNNDK 2.05，DPU v1.4.0 / B1152F / 150 MHz |
| ARM 编译 | arm-linux-gnueabihf-g++ 与对应运行库 |
| 模型训练 | Python、NumPy、PyTorch 及配置列出的音频依赖 |

## 应用构建

在配置好 ARM 工具链的 Linux 环境中，以项目根目录作为工作目录：

```sh
REPO="$PWD"
mkdir -p "$REPO/build"
sh scripts/guest/build_soundjie_d4_runtime.sh --repo-root "$REPO" --output-dir "$REPO/build/runtime"
sh scripts/guest/build_soundjie_hdmi_ui.sh --repo-root "$REPO" --output-dir "$REPO/build/ui"
bash scripts/guest/build_soundjie_d4_classifier.sh --repo-root "$REPO" \
  --dnndk-root /your/licensed/dnndk-arm \
  --model-elf "$REPO/models/release/dpu_soundjie_alert5bg.elf" \
  --physical-outputs 6 --storage-outputs 6 --kernel-name soundjie_alert5bg \
  --output-dir "$REPO/build/classifier"
```

方向接口工具的构建入口为 `scripts/guest/build_xvf3800_doa_tool.sh`，字库入口为 `scripts/host/build_soundjie_glyph_atlas.py`。训练、量化与部署入口位于 `scripts/host/`，类别顺序与参数以 `config/soundjie_alert5bg_*.json` 为准。

## PL 集成

`scripts/vivado/` 提供工程集成与报告导出工具，`hardware/emmc_candidate_bd.tcl` 保存 BD 导出。综合需要独立取得板厂工程、DPU IP 和对应工具许可。原工程已经包含 DPU、显示与 eMMC 集成，恢复构建应使用脚本中的 `resume/report/export` 路线，避免重复添加 IP。

## 目录与复现

历史脚本中的 E/F 盘和 Linux 共享目录需要替换为本机路径。不得直接照抄磁盘编号运行写卡工具。公开源码不是整张 SD 镜像；板端运行仍需匹配的 BOOT.bin、内核与根文件系统。

本仓库整理应用与构建源码，未附开发阶段的操作日志及全部历史文档。部分历史回归测试依赖原开发机、私有板厂包或本地数据，不能把全量测试收集结果作为当前电脑的开箱验证。

可先运行不依赖板卡的启动脚本与数据选择检查：

```sh
python -m unittest tests.test_soundjie_autostart tests.test_soundjie_alert5bg_probes -v
```

维护关机应等待板端 `System halted` 后再切断电源。
