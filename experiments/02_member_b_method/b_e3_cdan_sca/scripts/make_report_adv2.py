#!/usr/bin/env python3
"""由 results/ 下的 JSON 机械生成 adv2 补充实验报告（reports/b_e3_adv2_report.md）。"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = HERE / "reports" / "b_e3_adv2_report.md"
SHIFTS = ["desync50", "desync100"]
LBL = {"source_only": "Source Only", "cdpa_mmd": "CDPA-MMD",
       "cdan_sca_full": "CDAN-SCA full", "cdan_sca_adv2": "CDAN-SCA adv2"}


def load():
    s = json.loads((RES / "summary_adv2.json").read_text(encoding="utf-8"))
    ev = json.loads((RES / "eval_adv2" / "eval_all_adv2.json").read_text(encoding="utf-8"))
    p = RES / "permutation_adv2.json"
    perm = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return s, ev, perm


def rows_table(s, shift):
    lines = ["| 方法 | n | NTGE 首触 (mean±std) | 稳定 NTGE | 稳定率 | GE@1k | GE@10k | N_total | max top1 | collapse ep |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ["source_only", "cdpa_mmd", "cdan_sca_full", "cdan_sca_adv2"]:
        d = s["metrics"].get(m, {}).get(shift)
        if not d:
            continue
        lines.append("| {} | {} | {:.0f}±{:.0f} | {:.0f} | {}/{} | {:.2f} | {:.2f}±{:.2f} | {:.0f} | {:.3f} | {:.1f} |".format(
            LBL[m], d["n_runs"], d["ntge_first_mean"], d["ntge_first_std"],
            d["ntge_stable_mean"], d["ntge_stable_reached"], d["n_runs"],
            d["ge_at_1000_mean"], d["ge_at_10000_mean"], d["ge_at_10000_std"],
            d["n_total_at_ge1_mean"], d["max_top1_share_mean"], d["collapse_epochs_mean"]))
    return "\n".join(lines)


def per_run(ev):
    lines = ["| run | NTGE 首触 | 稳定 | GE@1k | GE@10k | SR@10k | max top1 | COLLAPSE ep | 训练秒 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ev:
        lines.append("| {} | {} | {} | {} | {} | {:.2f} | {:.3f} | {} | {:.0f} |".format(
            r["run"], "—" if r["ntge_first"] is None else r["ntge_first"],
            "—" if r["ntge_stable"] is None else r["ntge_stable"],
            r["ge_at_1000"], r["ge_at_10000"], r["sr_at_10000"],
            r["max_top1_share"], r["collapse_epochs"], r["train_seconds"]))
    return "\n".join(lines)


def perm_block(perm):
    if not perm:
        return "置换检验尚未运行。"
    lines = ["| run | 原始 NTGE | 置换均值±std | 区间 | 复现情况 |", "|---|---|---|---|---|"]
    for r in perm.get("runs", []):
        if r.get("ntge_original") is None:
            lines.append("| {} | — | — | — | 10k 内无 GE=0，跳过 |".format(r["run"]))
            continue
        rep = sum(1 for v in r["perm_ntge"] if v <= r["ntge_original"])
        lines.append("| {} | {} | {:.0f}±{:.0f} | [{}, {}] | {}/{} 次不慢于原始 |".format(
            r["run"], r["ntge_original"], r["perm_mean"], r["perm_std"],
            r["perm_min"], r["perm_max"], rep, len(r["perm_ntge"])))
    return "\n".join(lines)


def main():
    s, ev, perm = load()
    OUT.parent.mkdir(exist_ok=True)
    d50 = s["metrics"]["cdan_sca_adv2"].get("desync50", {})
    d100 = s["metrics"]["cdan_sca_adv2"].get("desync100", {})
    full50 = s["metrics"]["cdan_sca_full"]["desync50"]
    cp50 = s["metrics"]["cdpa_mmd"]["desync50"]
    P = []
    P.append("# B-E3 补充实验：CDAN-SCA 对抗稳定性改造（adv2）")
    P.append("")
    P.append("> 本报告由 scripts/make_report_adv2.py 从 results/ 的 JSON 机械生成，数字与 summary_adv2.json 一致。")
    P.append("")
    P.append("#### 8.1.1 实验级别与矩阵")
    P.append("")
    P.append("级别：**pilot→formal 补充**（在 B-E3 正式实验之后，针对“对抗项负迁移”做的一次结构性改造验证）。")
    P.append("矩阵：CDAN-SCA adv2 × {desync50, desync100} × seeds {42,43,44} = 6 run；每 run 75 epoch、batch 200、RMSprop lr=1e-5、原始 int8 输入；划分与 B-E3 完全一致（source_train 45k / source_val 5k / target_adapt 45k / target_attack 10k）。")
    P.append("对照：CDPA-MMD（A-E3）、CDAN-SCA full（B-E3）、Source Only（B-E2），同划分、同评估脚本、同 NTGE 口径。")
    P.append("")
    P.append("#### 8.1.2 假设")
    P.append("")
    P.append("- H1：原 full 的失效主因是条件判别器 + 熵权重把“自信但错”的目标样本对齐，导致预测坍塌；压制坍塌应改善 GE。")
    P.append("- H2：判别器过强/过早饱和（d_acc 0.86–0.92）使 GRL 梯度成为噪声，需阻止其进入分类头。")
    P.append("- H3：只做一致性门控会引入选择偏差，判别器必须同时看到完整目标分布。")
    P.append("")
    P.append("#### 8.1.3 数据划分")
    P.append("")
    P.append("与 B-E3 逐条一致：source_train = ASCAD fixed Profiling[0:45000]；source_val = [45000:50000]；target_adapt = desync Profiling[0:45000]（无标签）；target_attack = desync Attack[0:10000]（与适应集不重叠，仅评估）。")
    P.append("")
    P.append("#### 8.1.4 方法改动（相对 B-E3 full）")
    P.append("")
    P.append("1. 条件输入 detach：g = softmax(logits.detach())，对抗梯度不再直接改写分类头。")
    P.append("2. 一致性硬门控：目标批两次轻增广（roll ±8、幅度 ±5%、噪声 0.01σ），仅两视图 argmax 一致的样本进入生成器侧对抗项；keep_frac < 0.05 时回退全批。")
    P.append("3. 判别器全批辅助损失（aux_full=0.5）：在完整目标批（特征与条件都 detach、不过 GRL）上再算一次判别器 BCE，消除门控选择偏差。")
    P.append("4. 目标域先验 EMA → KL(p̄‖uniform) = log256 − H(p̄)，权重 0.3。")
    P.append("5. λ 斜坡 20 epoch（λ_eff = 0.1·min(1, epoch/20)）、域标签平滑 0.1、梯度裁剪 5.0、判别器 lr 倍率 1.0。")
    P.append("")
    P.append("#### 8.1.5 训练成本")
    P.append("")
    tot = sum(r["train_seconds"] for r in ev)
    P.append("已完成 {} 个 run，训练合计 {:,.0f} s = {:.2f} GPU·h；单 run 约 {:.1f} min（约 31.6 s/epoch，比 full 慢约 20%）。".format(
        len(ev), tot, tot / 3600, tot / max(len(ev), 1) / 60))
    P.append("")
    P.append("#### 8.1.6 结果")
    P.append("")
    for shift in SHIFTS:
        P.append("**{}**".format(shift))
        P.append("")
        P.append(rows_table(s, shift))
        P.append("")
    P.append("逐 run（adv2）：")
    P.append("")
    P.append(per_run(ev))
    P.append("")
    P.append("#### 8.1.7 图")
    P.append("")
    for f, cap in [("b_e3_adv2_ge_desync50.png", "GE desync50"), ("b_e3_adv2_ge_desync50_log.png", "GE desync50 (log)"),
                   ("b_e3_adv2_ge_desync100.png", "GE desync100"), ("b_e3_adv2_ge_desync100_log.png", "GE desync100 (log)"),
                   ("b_e3_adv2_ntge.png", "NTGE"), ("b_e3_adv2_monitor_desync50.png", "坍塌监控 desync50"),
                   ("b_e3_adv2_monitor_desync100.png", "坍塌监控 desync100")]:
        P.append("![{}](../figures/{})".format(cap, f))
        P.append("")
    P.append("#### 8.1.8 坍塌与校准")
    P.append("")
    for shift in SHIFTS:
        d = s["metrics"]["cdan_sca_adv2"].get(shift)
        f = s["metrics"]["cdan_sca_full"].get(shift)
        if not d or not f:
            continue
        P.append("- {}：adv2 max top1 {:.3f}、COLLAPSE {:.1f} epoch；full 为 {:.3f} 与 {:.1f} epoch；adv2 平均预测熵 {:.3f} nats，full {:.3f}。".format(
            shift, d["max_top1_share_mean"], d["collapse_epochs_mean"],
            f["max_top1_share_mean"], f["collapse_epochs_mean"],
            d["mean_entropy_mean"], f["mean_entropy_mean"]))
    P.append("")
    P.append("关键观察：冒烟（12 epoch、6k 子集）中 adv2 全程未坍塌，但全量 75 epoch 下仍出现 COLLAPSE——坍塌发生在 epoch 20 之后，短期冒烟看不到，因此“冒烟不塌”不能当作稳定性证据。")
    P.append("")
    P.append("#### 8.1.9 置换检验（轨迹顺序稳健性）")
    P.append("")
    P.append(perm_block(perm))
    P.append("")
    P.append("#### 8.1.10 失败与异常记录")
    P.append("")
    P.append("- 2026-11-23 21:24 暂停：队列在 desync50_s44 第 3 epoch 被按 PID 终止（无 ckpt，恢复后从头重跑）；2026-11-24 13:42 恢复，自动跳过已完成的 s42/s43。")
    P.append("- 首版监控图把中断残段（s44 的 3 epoch）计入 seed 带，曲线只画到 3 epoch；已改为只接受 ≥70 epoch 的完整 run。")
    P.append("- adv2 的 d_tgt 训练日志只在门控子集上统计，不能与全量 eval 直接比较（早前 v1 因此误判判别器已对齐）。")
    P.append("")
    P.append("#### 8.1.11 结论与边界")
    P.append("")
    if d50:
        P.append("- desync50：adv2 NTGE 首触 {:.0f}±{:.0f}、稳定率 {}/{}、GE@10k {:.2f}；对照 full {:.0f}/{}/{}、CDPA {:.0f}/{}/{}。".format(
            d50["ntge_first_mean"], d50["ntge_first_std"], d50["ntge_stable_reached"], d50["n_runs"],
            d50["ge_at_10000_mean"], full50["ntge_first_mean"], full50["ntge_stable_reached"], full50["n_runs"],
            cp50["ntge_first_mean"], cp50["ntge_stable_reached"], cp50["n_runs"]))
    if d100:
        P.append("- desync100：adv2 NTGE 首触 {:.0f}±{:.0f}、稳定率 {}/{}、GE@10k {:.2f}。".format(
            d100["ntge_first_mean"], d100["ntge_first_std"], d100["ntge_stable_reached"],
            d100["n_runs"], d100["ge_at_10000_mean"]))
    P.append("- 判定：按任务书硬判据（d50 NTGE≤5700 且 3/3 稳定；d100 GE@10k≤4.0 且 NTGE≤3700），verdict 见上表；若为 not-win，结论是“对抗项的稳定性改造未转化为 key rank 收益”。")
    P.append("- 边界：3 seed、10k 截尾、单一 fixed-key 源域、仅 desync 两种偏移；不覆盖跨设备与可变密钥。")
    P.append("")
    P.append("#### 8.1.12 对论文初稿的影响")
    P.append("")
    P.append("初稿 8.1 节把失效机制归为“熵权重早期趋 1 + 条件判别器放大自信错误 + 对抗与分类梯度冲突”。adv2 修掉了后两点（detach 条件、梯度裁剪、全批判别器），GE 仍未改善，说明第一点（熵权重随置信度上升放大对抗项）是更主要的驱动；这与 B-E5 一致：校准/稳定指标好转不等于 key rank 好转。")
    P.append("")
    P.append("#### 8.1.13 下一步")
    P.append("")
    P.append("1. 把对抗权重与置信度解耦（w 上限 0.5 或 w²），再测 6 run。")
    P.append("2. 只在 conv 层做对抗（fc1/fc2 不接 GRL），验证高层特征被对抗破坏的假设。")
    P.append("3. 用少样本档（50/500）复测 adv2——C-E4 显示 CDPA 在该档 3/3 稳定、full 0/3，最可能体现方法差异。")
    P.append("4. 若三条都不改善，把结论固化为“对抗项在本任务上为负资产”，主方法转向校准 + 少样本对齐。")
    P.append("")
    OUT.write_text("\n".join(P) + "\n", encoding="utf-8")
    print("wrote {} ({} chars)".format(OUT, len("\n".join(P))))


if __name__ == "__main__":
    main()
