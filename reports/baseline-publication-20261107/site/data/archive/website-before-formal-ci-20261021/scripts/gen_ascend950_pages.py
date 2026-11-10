"""Render 950 correctness and historical performance with separate provenance."""

import argparse
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

from ascend950_data import FORMAL_PROTOCOL, key, validate_ci, validate_performance

GH = "https://github.com/yyttt6/TileOPs"
SITE_GH = "https://github.com/yyttt6/TileOPs.github.io"


def cell(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ascend950"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/benchmarks"))
    args = parser.parse_args()
    perf = validate_performance(json.loads((args.data_dir / "development-v2.json").read_text()))
    ci = validate_ci(json.loads((args.data_dir / "current-ci.json").read_text()))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    for name in ("development-v2.json", "current-ci.json"):
        shutil.copyfile(args.data_dir / name, out / "data" / name)
    groups, ci_groups = defaultdict(list), defaultdict(list)
    for row in perf["cases"]:
        groups[row["op"]].append(row)
    for row in ci["cases"]:
        ci_groups[row["op"]].append(row)
    dates = sorted(r["tileops"]["measured_at"] for r in perf["cases"])
    passed = sum(r["passes_80pct"] for r in perf["cases"])
    run, env = ci["run"], ci["environment"]

    for zh in (False, True):
        def t(en, cn, zh=zh):
            return cn if zh else en

        def write(name, lines, zh=zh):
            (out / (name + (".zh.md" if zh else ".md"))).write_text("\n".join(lines) + "\n")

        snapshot = [f"**{t('Hardware', '设备')}** {cell(env['device'])} · **TileLang** {env['tilelang']}", "",
                    f"**{t('CI commit', 'CI 提交')}** [{run['commit'][:12]}]({GH}/commit/{run['commit']}) · "
                    f"[CI]({run['url']}) · {cell(run['completed_at'])}", ""]
        notice = [f"!!! warning \"{t('Different version scopes', '性能与正确性对应不同版本')}\"", "",
                  "    " + t("Release CI verifies correctness only. Latencies below were measured on October 18–20, 2026 with an earlier TileLang development revision and v2 protocol, without verified clock locking. They are not TileLang 0.1.15 performance results. Locked release measurements and expanded baseline comparisons remain pending.",
                             "Release CI 仅验证正确性。现有耗时采集于 2026 年 10 月 18–20 日，使用较早的 TileLang 开发版本与 v2 协议，未验证锁频；不能作为 TileLang 0.1.15 的性能成绩。锁频 release 测量和扩充后的 baseline 对比仍待完成。"), ""]
        write("index", ["# " + t("Ascend 950 results", "Ascend 950 结果"), "", *snapshot, *notice,
              t("| Dataset | Operators | Cases | Meaning |", "| 数据集 | 算子 | Case | 含义 |"), "| --- | ---: | ---: | --- |",
              f"| [{t('Release correctness', 'Release 正确性')}](correctness.md) | {len(ci_groups)} | {ci['passed']} / {len(ci['cases'])} | TileLang 0.1.15 · {t('all outputs verified', '全部输出已验证')} |",
              f"| [{t('Development performance', '开发性能记录')}](performance.md) | {len(groups)} | {len(perf['cases'])} | v2 · {t('measured revisions, unlocked', '对应已测版本，未锁频')} |", "",
              t(f"The historical development threshold is baseline / TileOPs ≥ 0.8: **{passed}/{len(perf['cases'])} cases** meet it. This count does not establish final operator acceptance or release performance.",
                f"历史开发门槛为 baseline / TileOPs ≥ 0.8：**{passed}/{len(perf['cases'])} 个 case** 达到门槛。此计数不代表算子最终验收，也不代表 release 性能。"), "",
              "- [" + t("Measurement method", "测量方法") + "](reading.md)",
              "- [" + t("Download CI evidence", "下载 CI 数据") + "](data/current-ci.json)",
              "- [" + t("Download performance samples and source hashes", "下载原始性能样本及来源哈希") + "](data/development-v2.json)",
              "- [" + t("Ascend 910B1 backup", "Ascend 910B1 备份") + "](archive.md)", "",
              "## " + t("Updates", "更新方式"), "",
              t("Deployments and scheduled refreshes fetch the latest successful trusted ascend950 CI artifact. Its actual date and commit are shown above. Development timings change only when a newly reviewed snapshot is imported; refreshing the page does not rerun benchmarks.",
                "部署与定时刷新获取最新成功的 ascend950 CI 产物，上方显示其实际日期和提交。开发性能仅随新审查快照更新；刷新页面不会重新执行 benchmark。")])
        lines = ["# " + t("Development performance — Ascend 950", "开发性能记录 — Ascend 950"), "", *notice,
                 f"**{t('Measured UTC', '采样 UTC')}** {dates[0]} — {dates[-1]}", "",
                 t("One row per case. Ratio = selected baseline time / TileOPs time, in milliseconds. Both sides use v2, 3 warmups, 15 samples and 1 GiB eviction. The baseline is the fastest eligible candidate in the frozen v2 pool, not a claim about every library. No roofline/SOL value is inferred.",
                   "每行一个 case。比值 = 选中 baseline 耗时 / TileOPs 耗时，单位为毫秒。两侧均用 v2、3 次 warmup、15 次采样及 1 GiB 缓存驱逐。baseline 是冻结 v2 候选池中最快的有效记录，不代表所有库的最优结果。未推算 roofline 或 SOL。"), "",
                 "[" + t("Method", "测量方法") + "](reading.md) · [JSON](data/development-v2.json)", ""]
        for op, rows in sorted(groups.items()):
            lines += [f"## {op}", "", '<div class="datatable" markdown="1">', "",
                      t("| Workload | Ratio | TileOPs ms | Baseline ms | Selected baseline | ≥ 0.8 | Measured UTC |",
                        "| Workload | 比值 | TileOPs ms | Baseline ms | 选中 baseline | ≥ 0.8 | 采样 UTC |"),
                      "| --- | ---: | ---: | ---: | --- | --- | --- |"]
            for row in sorted(rows, key=key):
                ratio = row["ratio"]
                color = "perf-ahead" if ratio > 1.05 else "perf-behind" if ratio < .95 else "perf-par"
                verdict = t("yes", "是") if row["passes_80pct"] else t("no", "否")
                lines.append(f"| `{cell(row['case_id'])}` | <span class=\"{color}\">{ratio:.4f}×</span> | "
                             f"{row['tileops']['latency_ms']:.9g} | {row['baseline']['latency_ms']:.9g} | "
                             f"`{cell(row['baseline']['candidate'])}` | {verdict} | {row['tileops']['measured_at']} |")
            lines += ["", "</div>", "", "??? info \"" + t("Recorded shapes and dtypes", "实际记录的形状与 dtype") + "\"", ""]
            for row in sorted(rows, key=key):
                lines += [f"    **{cell(row['case_id'])}**", "", "    ```json",
                          "    " + json.dumps(row["params"], ensure_ascii=False, sort_keys=True), "    ```", ""]
        write("performance", lines)
        lines = ["# " + t("Release correctness — Ascend 950", "Release 正确性 — Ascend 950"), "", *snapshot,
                 t("All manifest cases for every registered Ascend operator are checked. The receipt records verify_only=true, successful guard exit and unchanged source fingerprints. This is not a latency measurement or a performance acceptance result.",
                   "检查覆盖所有已注册 Ascend 算子的全部 manifest case。回执记录 verify_only=true、设备锁正常退出及源码指纹未变。此结果不是耗时测量或性能验收。"), "",
                 "[JSON](data/current-ci.json)", "", t("| Runtime | Version |", "| 运行环境 | 版本 |"), "| --- | --- |",
                 f"| TileLang | {env['tilelang']} |", f"| PyTorch | {env['torch']} |", f"| torch-npu | {env['torch_npu']} |"]
        lines += [f"| {cell(k)} | {cell(v)} |" for k, v in env["cann_components"].items()]
        for op, rows in sorted(ci_groups.items()):
            lines += ["", f"## {op}", "", t("| Case | Outputs checked | Result |", "| Case | 已检查输出 | 结果 |"), "| --- | ---: | --- |"]
            for row in sorted(rows, key=key):
                c = row["implementations"]["tileops"]["correctness"]
                lines.append(f"| `{cell(row['case_id'])}` | {c['checked_outputs']} / {c['total_outputs']} | {t('passed', '通过')} |")
        write("correctness", lines)
        write("reading", ["# " + t("Measurement scope", "测量口径"), "", *notice,
              f"## {t('Historical development protocol', '历史开发协议')}", "", f"`{perf['protocol']}` · {perf['hardware']}", "",
              t("3 warmups; 15 independent profiler windows; one call per window. Reset, 1 GiB cache eviction and synchronization are outside the window. The score is the arithmetic mean of summed device kernel durations, excluding host launch gaps. Every sample is retained. Formal timing uses torch_npu.profiler.profile, not Event timing. Clock locking was not established.",
                "3 次 warmup；15 个独立 profiler 窗口，每窗调用一次。reset、1 GiB 缓存驱逐及同步在计时窗外。分数是每次设备 kernel 耗时之和的算术平均值，不计 host 发射间隙，保留全部样本。正式计时使用 torch_npu.profiler.profile，不使用 Event；当时未建立锁频证据。"), "",
              t("manifest_reference denotes the spec reference callable, not an independent external library. Additional baseline integration is ongoing. Reaching the 0.8 development threshold differs from beating the baseline (ratio > 1). Recorded commits, toolchain fingerprints, timestamps and source-file hashes are in the JSON; the measured source/compiler revisions may differ from release CI.",
                "manifest_reference 表示 spec 参考 callable，不表示独立外部库。额外 baseline 接入仍在进行。达到 0.8 门槛与超过 baseline（比值大于 1）不同。JSON 保留记录提交、工具链指纹、日期和来源文件哈希；已测源码及编译器版本可能与 release CI 不同。"), "",
              f"## {t('Shared release benchmark', '共享 release benchmark')}", "", f"`{FORMAL_PROTOCOL}` · TileLang **0.1.15**", "",
              t("The shared entry requires verified clock control and measures the mean union of device intervals, counting overlapping kernels once. It reserves the host device before runtime imports, checks every output, preserves traces and requires a final successful receipt. All developers must use the cooperative guard. Missing clock control prevents formal measurement. Results cannot be mixed with the historical v2 baseline.",
                "共享入口要求已验证的锁频控制，测量设备执行区间并集的平均值，重叠 kernel 只计一次。导入运行时前获取主机设备锁，检查全部输出、保留 trace，并要求最终回执成功。所有开发者都须遵守协作锁。缺少锁频控制时禁止正式测量，其结果不能与历史 v2 baseline 混比。"), "",
              f"[{t('Shared benchmark guide', '共享 benchmark 指南')}]({GH}/blob/ascend950/docs/ascend950/development.md)", "",
              "[JSON](data/development-v2.json) · [CI JSON](data/current-ci.json)"])
        write("archive", ["# " + t("Ascend 910B1 archive", "Ascend 910B1 历史备份"), "",
              t("Previous 910B1 data is preserved separately and is not merged into the Ascend 950 tables.", "此前 910B1 数据独立保留，未混入 Ascend 950 表格。"), "",
              f"- [{t('Original website source', '原网站源码')}]({SITE_GH}/tree/archive/ascend910b1-20261021)",
              f"- [{t('Complete raw snapshot and SHA-256 manifest', '完整原始快照及 SHA-256 清单')}]({SITE_GH}/tree/main/data/archive/ascend910b1-20261021)",
              f"- [{t('Original benchmark commit', '原始 benchmark 固定提交')}]({GH}/tree/c71531310ee6ce0a26b0a86b7a6ddc79f0201ab2)",
              "- [" + t("Archived cannbot comparison", "历史 cannbot 对照") + "](../archive/ascend910b1/cannbot.md)"])
    print(f"Rendered {len(ci['cases'])} correctness and {len(perf['cases'])} historical performance cases")


if __name__ == "__main__":
    main()
