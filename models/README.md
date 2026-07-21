# 模型目录说明

## 核心原则

- **本地只保留当前最佳模型** — 旧模型通过 git tag 永久保存到 GitHub
- **每次发布打 tag** — 历史版本可随时从 GitHub 检出恢复
- **训练脚本指定输出目录** — 覆盖还是新建由你决定

## 当前模型

| 目录 | 说明 | T_RMSE | RH_RMSE |
|------|------|--------|---------|
| `models_mp3000a_v2/brnn_*.pt` | **当前最佳** — MP-3000A Obs_BT + QC | 1.45 K | 9.0% |

## 历史版本（GitHub Tags）

[GitHub Releases/Tags](https://github.com/3035596206/mwr-retrieval/tags)

| Tag | 说明 | 恢复命令 |
|-----|------|---------|
| `v1.0` | POC 管线跑通，6/6 模型训练 | `git checkout v1.0 -- models/` |
| `v2.0` | MP-3000A Obs_BT + BT订正 + QC | `git checkout v2.0 -- models_mp3000a_v2/` |

## 工作流：发布新模型

```bash
# 1. 训练产出新的模型目录，例如 models_mp3000a_v3/
python train_mp3000a_v3.py

# 2. 提交 + 打 tag
git add models_mp3000a_v3/ train_mp3000a_v3.py results/
git commit -m "v3: (简述改进)"
git tag v3.0 -m "v3.0: T_RMSE=X.XXK RH_RMSE=X.XX% — (一句话说明)"
git push origin main v3.0

# 3. 本地可以删除旧模型目录（GitHub 上永久保留）
rm -rf models_mp3000a_v2/
git add -A && git commit -m "清理旧模型v2(已用tag v2.0保留)"
git push
```

## 恢复旧模型

```bash
# 查看所有历史版本
git tag -ln
# 恢复某个版本的全部模型
git checkout v1.0 -- models/
# 恢复单个文件
git show v2.0:models_mp3000a_v2/brnn_T_0-2km.pt > brnn_T_0-2km.pt
```
