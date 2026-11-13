# 来源与第三方组件

- `assets/cotton_cloth.obj`、`assets/cotton_weave.png`、`assets/soy_puddle.obj` 来自用户明确指定的 [原素材目录](https://github.com/Chloe-Jinlei/genesis-dual-franka-cotton-wiping/tree/main/soy_cloth_scene/assets)。原始下载地址与 SHA-256 见 `assets/provenance.json`。本仓库没有读取或复制原仓库的仿真控制程序。
- 仿真依赖 [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)。本机使用的上游提交记录在 `environment_provenance.json`，未将 Genesis 安装目录复制到本仓库。
- Franka Panda 几何、惯量和关节定义来自所安装 Genesis 提供的 `xml/franka_emika_panda/panda.xml`。运行时由 Genesis 加载，适用其资产来源的许可条款。
- 本仓库不为上述第三方组件重新授予许可；使用、再分发时应遵守各自来源的条款。
