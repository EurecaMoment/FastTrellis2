# OpenAI 风格的图生 3D API

这是 FastTrellis2 的 **OpenAI 风格扩展**。OpenAI 没有官方图生 3D 接口；本服务遵循其模型列表、Bearer 鉴权、JSON 错误和请求 ID 约定，提供专用 `/v1/objects/generations`，不冒充 Chat Completions、Responses 或 Images API。旧 `/infer` 已移除。

接口约定参考 [OpenAI API overview](https://developers.openai.com/api/reference/overview)、[Models list](https://developers.openai.com/api/reference/resources/models/methods/list) 和 [Error codes](https://developers.openai.com/api/docs/guides/error-codes)。本服务不会向 OpenAI 发送数据或调用其模型。

## 请求

所有接口在配置 `FASTTRELLIS2_API_KEY` 后要求 `Authorization: Bearer <key>`；监听非回环地址时必须配置密钥。通过 `--api-key-env` 可指定另一环境变量名，密钥不写进启动参数。以下均以 `http://127.0.0.1:3956` 为服务地址。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/v1/models` | OpenAI 格式的模型列表 |
| GET | `/v1/models/fasttrellis2` | 模型信息 |
| GET | `/openapi.json` | 可供客户端使用的 OpenAPI 3.1 定义 |
| POST | `/v1/objects/generations` | 提交图生 3D 任务 |
| GET | `/v1/objects/generations/{id}` | 查询状态与结果 |
| DELETE | `/v1/objects/generations/{id}` | 请求协作式取消 |
| POST | `/v1/objects/generations/{id}/cancel` | 同一取消操作 |
| GET | `/v1/files/{file_id}` | GLB 文件元信息 |
| GET | `/v1/files/{file_id}/content` | 下载 GLB |
| GET | `/ready`、`/health`、`/metrics` | 运维探测，使用相同鉴权 |

提交 JSON 字段：

| 字段 | 类型、默认值 | 约束 |
| --- | --- | --- |
| `model` | 必填字符串 | `fasttrellis2` |
| `image` | 必填字符串 | PNG/JPEG/WebP 的 base64，或 `data:image/png;base64,...` 等 data URL；严格 1024×1024 |
| `mask` | 可选字符串 | 同尺寸 base64/data URL 掩码，亮值为前景；省略时优先保留输入 alpha，否则用 BiRefNet |
| `seed` | 整数，42 | 0–4294967295 |
| `wait` | 布尔值，false | true 最多等待 30 秒；未结束时仍返回 202 和同一任务 ID |
| `timeout_seconds` | 数字，3600 | 0.01–3600，包含队列等待和生成；HTTP 断开不会重提任务 |
| `response_format` | 字符串，url | `url` 或 `b64_json`，后者把 GLB 编码进结果 |

不接受服务器文件路径、远程图片 URL、客户端输出目录和其他字段。客户端读取文件并编码后提交，因此无需共享服务器磁盘。JSON 上限 16 MiB，每张解码图片上限 6 MiB。返回的文件 URL 为同源相对路径，下载时同样发送 Bearer 密钥。

## Python / OpenAI SDK

客户端安装 `openai`，服务端不依赖它。SDK 的 `models.list()` 可直接使用；3D 为扩展路径，使用 SDK 通用 HTTP 方法，不能用 `images.generate()` 替代。

```python
import base64
import os
from pathlib import Path
import time
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:3956/v1", max_retries=0,
                api_key=os.environ["FASTTRELLIS2_API_KEY"])
print(client.models.list())
image = "data:image/png;base64," + base64.b64encode(Path("object.png").read_bytes()).decode()
job = client.post("/objects/generations", cast_to=dict, body={
    "model": "fasttrellis2", "image": image, "seed": 42, "wait": False,
})
print(job["id"])  # 保存 ID；后续只查询，勿重复提交。
while job["status"] in {"queued", "in_progress", "cancelling"}:
    time.sleep(1)
    job = client.get("/objects/generations/" + job["id"], cast_to=dict)
if job["status"] != "completed":
    raise RuntimeError(job["error"])
file_id = job["data"][0]["file_id"]
with client.files.with_streaming_response.content(file_id) as response:
    response.stream_to_file("object.glb")
```

`response_format=b64_json` 的结果可用 `base64.b64decode(job["data"][0]["b64_json"])` 写成 GLB。取消可调用 `client.delete("/objects/generations/" + job_id, cast_to=dict)`；`cancelling` 表示已收到取消请求，只有 `cancelled` 才表示工作线程已确认终止。取消检查发生在模型和导出检查点，不能保证立刻停止 GPU 内核。

## 状态、容量与恢复

默认同一服务最多受理 **64 个未结束任务**；图片准备、排队和运行都占用容量。第 65 个请求返回 HTTP 429、`generation_capacity_exceeded` 与 `Retry-After: 1`，不会额外派发 GPU 工作。仅在空出容量后重试被拒绝的提交。已受理的任务必须使用原 ID 查询；64 个请求不等于 GPU 同时执行 64 次推理。

跨请求采用 **GPU → CPU 导出流水线**：CPU 图片准备可与 GPU 工作重叠；GPU 阶段返回完全位于 CPU 的网格和材质后，交给独立工作线程进行校验、无损 PNG 编码与 GLB 写出，同时 GPU 开始下一请求。同一物体内部原有的 CPU UV / GPU 材质流水线也保留。跨请求的 GPU 阶段按物体隔离，避免共享 CUDA Graph、条件缓存和原生临时池被不同请求覆盖。

默认 `--export-workers 2`，`--pipeline-depth 3`。在进入 GPU 前预留阶段容量，最多同时持有三个请求的阶段数据，避免 64 份 4K 材质堆积在内存。导出线程堵塞时对 GPU 施加背压；`--pipeline-depth 1` 可用于顺序执行对照。GPU 使用一份权重和同一内存预算，仍是 512³、三个阶段各 12 步、4K PBR。排队较长的任务应有足够的 `timeout_seconds`。

`stage` 显示 `queued`、`gpu`、`waiting_for_cpu_export`、`cpu_export` 或终态。`usage.pipeline` 给出 GPU、CPU 导出阶段的起止时间，便于验证不同请求的重叠。`/metrics` 提供 `gpu_active`、`cpu_exports_active`、`active_requests`、`pipeline_depth`；请求完成必须等到导出结束，GPU 阶段完成不会提前释放受理容量。

任务状态为 `queued`、`in_progress`、`cancelling`、`completed`、`failed`、`cancelled`。只有 `completed` 返回可下载产物。每个任务在 `outputs/api/{id}/` 保存独立输入、状态回执和输出，可通过 `--api-output-dir` 改目录。服务重启后，完整结果可继续查询和下载；未结束任务标为 `failed` / `service_restarted`，不会自动重跑。运维方负责目录容量管理和备份，不自动删除生成文件。

错误格式为 `{"error":{"message":"...","type":"invalid_request_error","param":"image","code":"invalid_request"}}`。所有响应带 `x-request-id`。常见状态：400 参数错误、401 鉴权失败、404 模型/任务/接口不存在、409 文件尚未完成、413 请求过大、429 容量已满、503 模型未就绪。任务自身的生成错误在查询结果的 `error` 字段内保留；请求成功不等于模型生成完成。

## 验证

无需权重或 GPU 的协议、64 并发准入、跨请求阶段重叠与容量边界、取消、超时和重启回归：

```bash
python -m unittest discover -s fasttrellis2/tests -p 'test_*.py' -v
```

这些测试使用明确的假模型，验证服务协议和调度边界；真实模型吞吐量及输出仍需在目标 GPU 上单独验证。
