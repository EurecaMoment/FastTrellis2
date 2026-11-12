# FastTrellis2

FastTrellis2 基于 [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2)，提供常驻的单图到 **512³ Mesh + 4K PBR GLB** 推理服务。完整采样仍是稀疏结构、形状、材质各 12 步。发布入口固定使用已验证的 4096 面连通预分块 xatlas UV 路线，并整合 node1 v7 的通用推理优化。原项目说明保存在 [README.upstream.md](README.upstream.md)。

本仓库不包含模型权重或已生成资产。**4096 是单个 UV 预分组的面数上限，不限制最终网格的总面数。**全部分组仍由 xatlas 展开并统一打包；预分组可能改变切缝与纹素分布。发布版没有旧 UV 路线的切换开关。

## 已验证的范围

2026-11-12 在 node1 A800-SXM4-80GB GPU6 上，对六张此前未用于调优的图片分别用发布路线和历史 v7 配置完整生成资产。A/B 使用同一隔离常驻模型、512³、各 12 步、4K PBR、1024×1024 已整理输入和 alpha 掩码；每次重新生成资产，启动时只用其他图片预热。计时从读图至 GLB 写完。

发布路线 `whole_bounded` 的六物体平均完整耗时为 **8.953 秒**，最慢 **10.693 秒**，**5/6** 低于 10 秒；历史 v7 UV 配置的同批平均值为 17.863 秒。旧配置只作为 A/B 对照，发布代码不提供运行入口。

同一清理几何和解码 PBR 材质的八视角对照，六个物体的平均 PSNR 为 49.46–54.11 dB，48 个视角轮廓 IoU 均为 1，未新增岛内翻折；局部放大复核未见明显观感退化。纹素密度变化为 −1.12% 至 +1.49%，菠萝样本的像素中心 UV 重叠率轻微增加。启动和测试期间显存按 0.1 秒采样，峰值 26.634 GB，未触发 50 GB 保护。**这些是六个样本的结果，不保证任意新图都在 10 秒内，也不证明 UV 或外观逐像素完全相同；3 秒目标未达成。**

2026-11-12 已将 node1 GPU6 常驻服务切换到本仓库的快速 UV 路线。切换后的独立验收使用不同于预热图的 `reference.png` 新生成一份 4K PBR GLB：969,593 个三角面、166,651,972 字节，从读图到资产写完耗时 10.20 秒；服务随后返回 `ready`。这一个新样本不计入上述六样本平均值。

## 环境

已验证的组合是 Linux x86_64、Python 3.10、PyTorch 2.6.0 + CUDA 12.4、A800 80GB，以及支持 AVX-512F/DQ/VL 的主机 CPU。xatlas 优化源码使用 AVX-512 指令；其他 CPU 需要另做移植与验证。运行时需 CUDA Toolkit、C++ 编译器、`libjpeg-dev` 和 `libdeflate`。A100 80GB 是合适的目标硬件，但上面的延迟数据来自 A800，不能当作 A100 实测。

在仓库根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.0
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
sudo apt-get install -y libjpeg-dev libdeflate0
. ./setup.sh --basic --nvdiffrast --o-voxel
python -m pip install xformers==0.0.29.post3 --no-deps timm==1.0.31
git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git third_party/FlexGEMM
git -C third_party/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e
python -m pip install ./third_party/FlexGEMM --no-build-isolation
git clone --recursive https://github.com/JeffreyXiang/CuMesh.git third_party/CuMesh
git -C third_party/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C third_party/CuMesh submodule update --init --recursive
python -m pip install ./third_party/CuMesh --no-build-isolation
python fasttrellis2/build_extensions.py --cumesh-source third_party/CuMesh
```

`setup.sh` 是上游安装脚本，负责基础库、nvdiffrast 与仓库自带的 O-Voxel。启动时必须能从当前 Python 环境导入已编译的 `o_voxel._C`。`fasttrellis2/build_extensions.py` 从本仓库的 C++/CUDA 源码构建一个有界分组 xatlas 模块、两个 Telea 模块、异步 BVH 和有界 CUDA scratch arena。产物位于 `fasttrellis2/build/`，不会提交到 Git。

## 准备模型

```bash
python fasttrellis2/prepare_models.py --output-dir models/trellis2
```

脚本下载 `microsoft/TRELLIS.2-4B` 的 512 路线权重、外部稀疏解码器、公开的 timm DINOv3 与 `ZhengPeng7/BiRefNet`，在本机生成 `models/trellis2/pipeline.json`。node1 的验证基线使用这套公开条件模型配置；它与上游 `pipeline.json` 原配的 Facebook DINOv3、RMBG-2.0 配置不同。模型版本或条件模型改变后应重新做质量对照。模型下载需遵守各模型仓库的许可和访问要求。

## 启动常驻服务与 OpenAI 风格 API

给 `--warmup-image` 一张**不同于正式测试输入**的 1024×1024 图片。该参数可省略，但首个正式请求会承担预热开销。GPU 可以用序号或 UUID；示例中的路径请换成实际绝对路径。

```bash
bash fasttrellis2/start.sh \
  --model-dir "$PWD/models/trellis2" \
  --gpu 6 --port 3956 \
  --max-requests 64 \
  --memory-budget-gib 32 \
  --warmup-image /absolute/path/to/warmup-1024.png
```

服务默认只绑定 `127.0.0.1`。`GET /ready` 返回 `state=ready` 后即可请求。模型权重、固定时间条件、CUDA Graph 和核缓存常驻；每个新物体的图像 K/V、稀疏元数据和材质等输入相关内容会重新计算或在同物体采样步间复用，并在请求结束清理。不会复用已生成资产。50 GB 进程显存门槛由 NVML 请求准入检查保护；`--memory-budget-gib 32` 是 PyTorch 分配预算，另有原生 CUDA arena 和预留空间。

推理接口统一为 **OpenAI 风格 3D 扩展**，旧 `/infer` 已移除。使用 `GET /v1/models` 获取模型，向 `POST /v1/objects/generations` 提交 `model="fasttrellis2"` 与 base64/data URL 图片，获取任务 ID，再查询同一任务并通过 `/v1/files/{file_id}/content` 下载 GLB。图片必须为 1024×1024；可传同尺寸掩码，透明输入会保留 alpha，无掩码或 alpha 时使用 BiRefNet。无需向客户端暴露服务器文件路径。

默认最多受理 **64 个未结束任务**；额外提交返回 429。请求之间采用有界流水线：下一请求的 GPU 生成可与前一请求的 CPU 校验、无损 PNG 编码和 GLB 写出重叠；默认两个 CPU 导出工作线程、最多三个请求持有阶段数据。同一物体内部仍保留 CPU UV 与 GPU 材质的重叠。GPU 阶段使用一份常驻模型，按物体隔离输入缓存。任务有持久化回执，支持异步状态、超时、协作式取消和服务重启后的完整结果查询；只有完整产物才会标记完成。

设置环境变量 `FASTTRELLIS2_API_KEY` 可启用 Bearer 鉴权；非回环监听必须设置。OpenAI 没有官方图生 3D 接口，本服务不实现 Chat Completions、Responses 或 Images。完整字段、状态、SDK 示例与并发语义见 [API 文档](fasttrellis2/OPENAI_API.md)。

如需按配置文件启动，可使用 `bash fasttrellis2/start.sh --config /path/to/config.json`。可配置路径、GPU、端口、内存预算及预热图片；UV 分组上限固定为 4096，稀疏结构并行 CFG、零负条件缓存和融合算子保持启用。相对路径以配置文件目录为基准。运行环境需要通过 `fasttrellis2/start.sh` 提供扩展目录与 `LD_PRELOAD`。

## 优化内容与限制

当前代码包含每物体图像 cross-attention K/V 和时间条件复用、零负条件缓存、结构阶段 CUDA Graph、保持原舍入边界的 Triton 融合、双流 CFG、CPU UV 与 GPU 材质流水执行、异步 BVH、3 GiB 有界临时显存池、xatlas C++ 批量与内核优化、原算法 Telea C++/CUDA 流式补边、并行无损 PNG 编码及直接 GLB 写出。UV 路线按当前清理网格做每组最多 4096 面的连通分组，然后仍由 xatlas 完成分岛、展开和打包；它不会跳过 12 步采样或降低 4K 贴图分辨率，也不把整个资产限制在 4096 面。

此版本依赖已编译的 Linux 扩展和指定硬件指令集；**源码整合不等于在新机器上已完成端到端验收**。部署到另一台机器应先核对扩展导入、显存和完整 GLB，并用本机样本复测质量及延迟。

## 许可与来源

主仓库延续上游 [MIT 许可](LICENSE)。xatlas、OpenCV Telea 的修改源码及 CuMesh cuBVH 绑定保留在 `fasttrellis2/native/`，对应的第三方归属见 [THIRD_PARTY_NOTICES.md](fasttrellis2/THIRD_PARTY_NOTICES.md)。模型权重不在仓库内，许可由各模型发布方决定。
