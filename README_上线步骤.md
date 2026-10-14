# GFENet Streamlit 在线部署包

这是基于现有本地 webapp 改成的独立 Streamlit 版本，使用同一套真实训练权重。当前交付是部署包，尚未发布到互联网。

## 功能

- 上传分子 CSV 和四类配套指纹 CSV。
- 选择一个或多个呼吸毒性任务；模型计算全部任务，界面和导出展示所选任务。
- 一键真实示例预测、失败原因、CSV 下载及既有留出集指标。
- CPU 推理；无需让原来的 Windows 电脑开机，也不需要将 CUDA 环境上传。
- 模型全局只加载一次，上传文件和结果不写入公共目录；每个会话单独保存结果，结果不跨用户共享。
- 同一时刻只允许一个预测计算；忙碌时提示稍后重试，没有后台长队列。
- 云端原型每次最多 50 个分子、每个分子最多 80 个原子、每个上传文件最多 5 MB。超限记录有明确提示。

## 第一步：注册账号

1. 注册/登录 https://github.com 。
2. 访问 https://share.streamlit.io ，使用 GitHub 账号登录并按页面提示授权。

不需要把密码交给任何人。本包不含登录凭证。

## 第二步：上传到 GitHub

1. 新建仓库，例如 gfenet-respiratory-predictor。
2. 将部署包解压，把本目录中的文件和文件夹上传到仓库根目录，不是只上传 ZIP 文件。
3. 确认仓库能直接看到 streamlit_app.py、requirements.txt、engine.py、model.py、features.py、assets、example、.streamlit。
4. assets/gfenet_best.pt 必须完整上传，不能只有文件名或下载占位文件。

可以使用公开仓库；如果使用私人仓库，需要授权 Streamlit 访问它，并在之后将应用本身设置为公开。仓库是否公开与网站是否公开是两个设置。
本包仅带推理必要的模型、查表、测试指标及一个示例，没有打包完整训练数据与图缓存。模型权重会随仓库一起提供。

## 第三步：部署

在 Streamlit Community Cloud 中点击 Create app / Deploy an app：

- Repository：选择刚才建立的仓库。
- Branch：main，或仓库实际使用的分支。
- Main file path：streamlit_app.py。
- Advanced settings / Python version：选择 **Python 3.11**。不要使用平台默认的更新版本；本包依赖按 3.11 配置。
- App URL：可填写一个尚未被使用的名称。

点击 Deploy，等待依赖安装和模型加载。成功后得到 https://你的名称.streamlit.app/ 形式的网址。
在应用 Settings → Sharing 中设置为公开可访问（public）。用未登录的浏览器窗口打开网址，并运行一次示例预测，确认访客也能使用。

## 第四步：给别人使用

分享上述 streamlit.app 网址即可。网站在云端运行，与你本机 127.0.0.1:8000 的旧网站相互独立。
首次访问、冷启动或应用休眠后唤醒可能较慢；免费共享资源有计算和内存限制，不保证高并发或始终立即响应。遇到资源不足时先减小批量，再考虑更高资源的部署方式。

## 使用者需要准备什么

分子 CSV 要有 smiles 列；配套 pubchem.csv / graph.csv / apc2d.csv / fp1024.csv 分别包含 881 / 1024 / 780 / 1024 个数值特征列，外加可选 Name 列。
如果带 Name，必须是 AUTOGEN_smi_1 起的连续编号。四类指纹与分子必须严格同顺序，并使用与训练数据相同的算法和参数生成。
本包仍没有从 SMILES 自动生成四类指纹的功能。不能拿示例指纹搭配新的分子列表。
上传文件在当前服务进程的会话内存中处理，没有持久化上传库。结果请及时下载；刷新连接、清除结果、应用休眠或重启可能丢失会话内容。

## 本地启动（可选）

在本目录打开终端，使用独立 Python 3.11 环境：

    python -m pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Windows 本地测试可以复用现有依赖，但请在独立环境安装 Streamlit；不要直接覆盖训练环境的软件版本。
Linux 部署时 requirements.txt 会选择 PyTorch 2.7.1 CPU 包。

## 文件说明

- streamlit_app.py：Streamlit 页面入口。
- engine.py：从原 webapp 复用的预测与校验逻辑，移除了 HTTP 服务和全局用户结果存储。
- model.py：原 GFENet 结构，内嵌所需辅助函数，参数名保持不变。
- features.py：从原图预处理代码抽取的推理函数，无桌面路径或训练数据依赖。
- assets/gfenet_best.pt：原训练权重，内容未改变。
- assets/qm9_lookup.json：与训练配套的参考特征查表。
- assets/test_metrics.csv：既有留出集指标，不是网站上线后的新验证。
- example：一条真实测试分子及配套指纹。
- .streamlit/config.toml：上传限制和主题。
- requirements.txt：部署依赖。
- VERIFICATION.json：已做的验证与未验证范围。

## 当前验证范围

已在本机独立 Streamlit 测试环境验证：页面初始化、真实示例推理、任务选择与导出、上传模式的输入要求、不同会话结果隔离、指纹 Name 错位拒绝、CPU 预测与原权重参考结果一致。
还没有在 Streamlit Community Cloud 的 Linux 容器中实际构建/运行，也没有生成公网网址。以上不构成云端部署已成功的声明。云端部署日志是最终安装结果的依据。

## 常见部署错误

- 找不到 model.py / 权重：确认整个包上传到正确层级，入口路径与仓库一致。
- NumPy/torch 安装失败：确认 Python 3.11，requirements.txt 未被旧 rt.yml 或其他环境文件覆盖。
- 应用资源超限：先用一条示例预测，减少批量；本地 GPU 的速度不代表共享云端 CPU 的速度。
- 网址需要登录：到应用 Sharing 中改为公开，使用未登录窗口检查。
- Name 编号错误：核对并按输入顺序整理指纹，不要只改编号却不核对分子对应关系。

模型面向研究用途，尚未经过外部数据验证或概率校准；预测为 0 不能证明安全。网页只是这套模型的使用入口，不会增加模型准确率。

## 官方参考

- 部署：https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- 依赖：https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- 公开访问：https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- 资源与休眠：https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
