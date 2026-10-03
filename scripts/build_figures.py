"""Build bilingual publication figures only from recorded data and predictions."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "research/results/research"
COLORS = {"last_month": "#8B9BA7", "ridge_25": "#233B53", "fair_tcn": "#007F82",
          "online_ensemble": "#D07745", "decomp_ridge": "#7872A3", "bias_0.5_0.5": "#B48643",
          "decomp_tcn": "#558B68", "multitask_0.05_weighted": "#BD5861"}
NAMES = {
    "last_month": ("沿用上月", "Last-month baseline"), "ridge_25": ("逐月 Ridge", "Monthly Ridge"),
    "ridge_5": ("Ridge α=5", "Ridge alpha=5"), "ridge_100": ("Ridge α=100", "Ridge alpha=100"),
    "hgb_7": ("梯度提升树 7叶", "Gradient boosting / 7 leaves"), "hgb_15": ("梯度提升树 15叶", "Gradient boosting / 15 leaves"),
    "nlinear": ("NLinear 形式", "NLinear-style linear model"), "legacy_tcn": ("六个月原型网络", "Six-month prototype CNN"),
    "fair_tcn": ("七个月上下文网络", "Seven-month contextual CNN"), "revin_tcn": ("窗口归一化网络", "Window-normalized CNN"),
    "decomp_ridge": ("双分支 Ridge", "Decomposed Ridge"), "decomp_tcn": ("双分支网络", "Two-branch CNN"),
    "decomp_revin": ("归一化双分支网络", "Normalized two-branch CNN"),
    "online_ensemble": ("在线组合", "Online ensemble"), "uniform_ensemble": ("均匀组合", "Uniform ensemble")}


def j(name): return json.loads((P / name).read_text(encoding="utf-8"))
def setup(lang):
    available = {f.name for f in font_manager.fontManager.ttflist}
    if lang == "zh" and not available.intersection({"Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC"}):
        raise RuntimeError("Chinese figures require a CJK font: Microsoft YaHei, SimHei or Noto Sans CJK SC. Use --language en otherwise.")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"] if lang=="zh" else ["DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path"})
def txt(lang, zh, en): return zh if lang=="zh" else en
def name(case, lang):
    if case.startswith("bias_"): return txt(lang,"校正 ","Bias correction ")+case[5:]
    if case.startswith("multitask_"): return txt(lang,"方向任务 ","Multitask ")+case[10:].replace("weighted",txt(lang,"加权","weighted")).replace("plain",txt(lang,"不加权","plain"))
    return NAMES.get(case,(case,case))[lang=="en"]
def save(fig, filename, lang):
    directory=ROOT/"docs/figures"/lang; directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(filename+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(directory/(filename+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
def matrix(frame,column): return frame.pivot(index="target_month",columns="city",values=column).sort_index().sort_index(axis=1).to_numpy()
def title(fig, heading, caption):
    fig.suptitle(heading,fontsize=16,fontweight="bold",color="#233B53")
    fig.text(.01,-.018,caption,fontsize=9,color="#566C78")
def ticks(ax,months,step=3):
    positions=np.arange(len(months)); ax.set_xticks(positions[::step],months[::step],rotation=30,ha="right")


def market(lang):
    raw=pd.read_csv(ROOT/"research/data/official_nbs_70city.csv")
    grouped=raw.groupby("period")
    months=sorted(raw.period.unique()); x=np.arange(len(months))
    fig,axes=plt.subplots(2,1,figsize=(13,7.5),sharex=True,gridspec_kw={"height_ratios":[1.3,1]},layout="constrained")
    for col,label,color in (("new_mom_pct",txt(lang,"新房","New homes"),"#007F82"),("second_mom_pct",txt(lang,"二手房","Second-hand homes"),"#D07745")):
        axes[0].plot(x,grouped[col].mean().reindex(months),color=color,linewidth=2,label=label)
    axes[0].axhline(0,color="#A8B4BC",lw=.8); axes[0].set_ylabel(txt(lang,"70城等权均值（%）","Equal-weight 70-city mean (%)"))
    axes[0].legend(ncol=2,loc="lower left",frameon=False)
    for lo,hi,color,label in (("2024-01","2024-12","#E3EFF1",txt(lang,"2024 滚动验证","2024 rolling validation")),("2025-01","2026-01","#F9EDE3",txt(lang,"回顾性评估","Retrospective evaluation"))):
        a,b=months.index(lo),months.index(hi)
        for ax in axes: ax.axvspan(a-.5,b+.5,color=color,zorder=0)
        axes[0].text((a+b)/2,.94,label,ha="center",transform=axes[0].get_xaxis_transform(),fontsize=10)
    shares=np.array([[(g.new_mom_pct<0).mean(),(g.new_mom_pct==0).mean(),(g.new_mom_pct>0).mean()] for _,g in grouped])*100
    axes[1].stackplot(x,shares.T,colors=["#8CA1B0","#DDE6EA","#3B9B91"],labels=[txt(lang,"下跌","Falling"),txt(lang,"持平","Flat"),txt(lang,"上涨","Rising")],alpha=.95)
    axes[1].set_ylabel(txt(lang,"新房方向占比（%）","New-home direction share (%)")); axes[1].set_ylim(0,100)
    axes[1].legend(loc="lower left",ncol=3,frameon=True); ticks(axes[1],months,6)
    for ax in axes: ax.grid(axis="y",alpha=.15); ax.margins(x=0)
    title(fig,txt(lang,"数据覆盖与观察到的市场变化","Data coverage and observed market variation"),txt(lang,"57个月 · 70城 · 3990条记录。均值是面板描述量，不是官方全国指数或年度涨幅。","57 months / 70 cities / 3,990 records. The panel mean is not an official national index or an annual growth rate."))
    save(fig,"01_market",lang)


def box(ax,x,y,w,h,heading,body,color="#EDF3F5",size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.03",facecolor=color,edgecolor="#B8CBD3",lw=1))
    ax.text(x+w/2,y+h*.74,heading,ha="center",va="center",fontsize=size,fontweight="bold",color="#233B53")
    ax.text(x+w/2,y+h*.31,body,ha="center",va="center",fontsize=size-1,color="#526C7A",linespacing=1.45)
def arrow(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,color="#6F8997",lw=1.3))


def protocol(lang):
    fig,axes=plt.subplots(2,1,figsize=(13,7.6),gridspec_kw={"height_ratios":[1,1.25]},layout="constrained")
    ax=axes[0]; ax.set_xlim(-.5,50.5); ax.set_ylim(-.7,3.4)
    for y,a,b,col,label in ((2,0,24,"#8CA1B0",txt(lang,"初始训练：25个标签月","Initial training: 25 target months")),(1,25,36,"#007F82",txt(lang,"2024验证：12次滚动起点","2024 validation: 12 origins")),(0,37,49,"#D07745",txt(lang,"回顾性评估：13次滚动起点","Retrospective evaluation: 13 origins"))):
        ax.barh(y,b-a+1,left=a-.5,height=.48,color=col)
        ax.text(a+.4,y,label,va="center",color="white",fontsize=10,fontweight="bold")
    ax.scatter([50],[0],s=80,color="#233B53",marker="D")
    ax.set_yticks([]); ax.set_xticks([0,24,36,49],["2021-12","2023-12","2024-12","2026-01"],rotation=25,ha="right")
    ax.set_title(txt(lang,"按目标月份划分；同月70城整体推进","Split by target month; all 70 cities advance together"),loc="left")
    ax.text(49.5,.72,txt(lang,"2026-02 截点预测","2026-02 as-of forecast"),ha="right",fontsize=9)
    ax.spines[["left","bottom"]].set_visible(False)
    ax=axes[1]; ax.set_xlim(0,13); ax.set_ylim(0,4.5); ax.axis("off")
    contents=[(txt(lang,"公布上一统计月数据","Release of month m-1"),txt(lang,"已知标签与特征\n最多到 m-1","Labels and features\nknown through m-1")),(txt(lang,"拟合当前模型","Fit current model"),txt(lang,"标准化 / 分类权重\n都只从过去计算","Scalers / class weights\ncomputed from past only")),(txt(lang,"预测目标月 m","Predict target month m"),txt(lang,"保存预测与旧状态\n不读取当前真实值","Store forecast and old state\nwithout observing current y")),(txt(lang,"公布后更新","Update after release"),txt(lang,"计算已实现误差\n供 m+1 起点使用","Compute realized errors\nfor the next origin m+1"))]
    for i,(h,b) in enumerate(contents):
        box(ax,.15+i*3.25,2.1,2.9,1.65,h,b,size=10)
        if i<3: arrow(ax,(3.05+i*3.25,2.9),(3.4+i*3.25,2.9))
    ax.text(.15,.75,txt(lang,"时点示例：2026-02-13公布1月指数 → 此后预测2月整月结果。2月实际值未进入项目。","Timing example: Jan 2026 indices released on 13 Feb → predict February after that release. February actuals are excluded."),fontsize=10,color="#233B53")
    title(fig,txt(lang,"时间协议与逐月更新","Temporal protocol and monthly updates"),txt(lang,"2024选择配置；2025-01至2026-01已被研究者查看，新增结果属于回顾性研究。","Configuration selection uses 2024. The researcher had previously inspected Jan 2025-Jan 2026; new results are retrospective."))
    save(fig,"02_protocol",lang)


def architecture(lang):
    fig,ax=plt.subplots(figsize=(14,8)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis("off")
    box(ax,.15,5.25,3.0,1.65,txt(lang,"七个月双通道序列","Seven-month sequence"),txt(lang,"B × 70城 × 7月 × 2\n新房 / 二手房环比","B x 70 cities x 7 months x 2\nNew / second-hand MoM"))
    box(ax,3.75,5.25,3.1,1.65,txt(lang,"时序卷积编码","Temporal convolution"),txt(lang,"Conv1d 2→12→12 / 核3\nReLU + 时间均值池化","Conv1d 2→12→12 / kernel 3\nReLU + temporal mean pooling"))
    box(ax,3.75,2.95,3.1,1.65,txt(lang,"同期已知上下文","Known context"),txt(lang,"19项数值特征\n+ 4维城市嵌入","19 numeric features\n+ 4-d city embedding"))
    box(ax,7.65,4.05,2.55,1.85,txt(lang,"修正量预测头","Correction head"),txt(lang,"35→24→1\nReLU / Dropout 0.15","35→24→1\nReLU / Dropout 0.15"))
    box(ax,10.85,4.05,2.9,1.85,txt(lang,"下一统计月输出","Next-month output"),txt(lang,"已知上期变化 + 修正量\n每城输出1个环比预测","Last known change + delta\nOne MoM prediction per city"),"#E5F2EE")
    arrow(ax,(3.15,6.1),(3.75,6.1)); arrow(ax,(6.85,6.1),(7.65,5.15)); arrow(ax,(6.85,3.7),(7.65,4.55)); arrow(ax,(10.2,4.97),(10.85,4.97))
    ax.text(.15,7.55,txt(lang,"上下文时序网络：1697参数，覆盖Ridge的lag6与19项上下文","Contextual temporal CNN: 1,697 parameters; covers Ridge's lag-6 and 19 context features"),fontsize=16,fontweight="bold",color="#233B53")
    for i,(h,b) in enumerate([(txt(lang,"窗口归一化对照","Window normalization"),txt(lang,"已知窗口均值 / 方差\n按新房通道还原尺度","Known-window mean / variance\nRestore the new-home scale")),(txt(lang,"全国与城市双分支","Market / city branches"),txt(lang,"预测全国等权均值 + 城市残差\n约束城市预测残差均值为0","Predict panel mean + city residual\nCenter city residual predictions")),(txt(lang,"方向辅助任务","Auxiliary direction task"),txt(lang,"共享表示预测涨 / 平 / 跌\nHuber + eta × CrossEntropy","Shared rise / flat / fall head\nHuber + eta x CrossEntropy"))]):
        box(ax,.15+i*4.65,.4,4.2,1.8,h,b,size=11)
    ax.text(.15,2.5,txt(lang,"独立结构对照；这些模块并未全部堆叠到最终网络。","Separate ablations: the final network does not stack all these modules."),fontsize=10,color="#566C78")
    fig.tight_layout(); save(fig,"03_architecture",lang)


def validation(lang):
    score=j("validation_screening_results.json"); f=pd.read_csv(P/"validation_screening_three_seeds.csv")
    names=list(score); n=len(names); gains=np.array([[score['ridge_25']['monthly_mae_pp'][m]-score[k]['monthly_mae_pp'][m] for m in sorted(score[k]['monthly_mae_pp'])] for k in names])
    fig,(ax,hx)=plt.subplots(1,2,figsize=(15,11),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
    y=np.arange(n)
    for i,k in enumerate(names):
        if len(score[k]['seed_mae_pp'])>1: ax.scatter(score[k]['seed_mae_pp'],np.full(len(score[k]['seed_mae_pp']),i),s=15,color="#AFBDC5",zorder=2)
        ax.scatter(score[k]['mae_pp'],i,color=COLORS.get(k,"#66808F"),s=60,zorder=3)
        ax.text(.334,i,f"{score[k]['mae_pp']:.4f}",va="center",ha="right",fontsize=9)
    ax.axvline(score['ridge_25']['mae_pp'],color="#D07745",ls="--",lw=1)
    ax.set_yticks(y,[name(k,lang) for k in names],fontsize=9); ax.set_ylim(n-.5,-.5)
    ax.set_xlim(.268,.335); ax.grid(axis="x",alpha=.15)
    ax.set_title(txt(lang,"整体误差；小灰点为各个种子MAE","Overall error; small grey dots = individual seeds"),loc="left",fontsize=11)
    ax.set_xlabel(txt(lang,"2024滚动验证 MAE（百分点）","2024 rolling-validation MAE (pp)"))
    hm=hx.imshow(gains,aspect="auto",cmap="RdBu",norm=TwoSlopeNorm(vcenter=0,vmin=-.14,vmax=.14))
    hx.set_yticks([]); hx.set_xticks(range(12),[f"{m:02}" for m in range(1,13)],fontsize=9)
    hx.set_xlabel(txt(lang,"2024目标月份","Target month in 2024")); hx.set_title(txt(lang,"每月相对Ridge的MAE收益（蓝色为改善）","Monthly MAE gain vs Ridge (blue = better)"),fontsize=11)
    cb=fig.colorbar(hm,ax=hx,shrink=.8,pad=.025); cb.set_label(txt(lang,"收益（百分点），色阶截至±0.14","Gain (pp); color clipped at +/-0.14"))
    title(fig,txt(lang,"全部20个配置的验证对照","Validation comparison of all 20 configurations"),txt(lang,"神经筛选统一3种子，实心点为平均预测的MAE。最终网络5种子验证为0.2803；Ridge为0.2781。","Neural screening uses 3 seeds; large dots score averaged predictions. Final CNN confirmation: 5 seeds, MAE 0.2803; Ridge 0.2781."))
    save(fig,"04_validation",lang)


def performance(lang):
    f=pd.read_csv(P/"retrospective_predictions.csv"); months=sorted(f.target_month.unique()); x=np.arange(len(months)); y=matrix(f,"actual_change_pct")
    names=["last_month","ridge_25","fair_tcn","online_ensemble"]
    predictions={k:matrix(f,"pred_"+k) for k in names}; gains=j("gain_vs_strong_baseline.json"); weights=j("ensemble_weights.json")
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    ax=axes[0,0]
    for k in names: ax.plot(x,np.abs(predictions[k]-y).mean(axis=1),label=name(k,lang),color=COLORS[k],marker="o",markersize=4,lw=1.7)
    ax.axvspan(4.5,7.5,color="#F3EEE5",zorder=0); ticks(ax,months,3); ax.set_ylabel(txt(lang,"每月MAE（百分点）","Monthly MAE (pp)")); ax.legend(ncol=2,fontsize=8,frameon=False)
    ax.set_title(txt(lang,"A  逐月误差；背景标出已知夏季诊断段","A  Monthly errors; shaded summer diagnostic"),loc="left",fontsize=11); ax.grid(alpha=.15)
    ax=axes[0,1]; comparison=["bias_0.5_0.5","decomp_ridge","fair_tcn","online_ensemble"]
    for i,k in enumerate(comparison):
        mean=gains[k]['mean_monthly_mae_gain_pp']; lo,hi=gains[k]['circular_3month_block_95pct_interval_pp']
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt="o",color=COLORS[k],capsize=4,markersize=7)
        ax.text(hi+.001,i,f"{mean:+.4f}",va="center",fontsize=9)
    ax.axvline(0,color="#97A5AE",lw=1); ax.set_yticks(range(4),[name(k,lang) for k in comparison],fontsize=9); ax.invert_yaxis()
    ax.set_xlim(-.012,.037); ax.set_xlabel(txt(lang,"相对逐月Ridge的MAE收益（百分点）","MAE gain over monthly Ridge (pp)")); ax.grid(axis="x",alpha=.15)
    ax.set_title(txt(lang,"B  三个月区块重采样95%区间","B  95% three-month block intervals"),loc="left",fontsize=11)
    ax=axes[1,0]; r=np.abs(predictions['ridge_25']-y).mean(axis=1); nn=np.abs(predictions['fair_tcn']-y).mean(axis=1)
    ax.plot([.16,.29],[.16,.29],ls="--",color="#A7B3BB"); ax.scatter(r,nn,c=["#D07745" if 5<=i<=7 else "#007F82" for i in x],s=65)
    for i,offset in {0:(5,5),3:(5,-13),5:(5,5),6:(5,-13),7:(5,5),12:(5,-13)}.items():
        ax.annotate(months[i][2:],(r[i],nn[i]),xytext=offset,textcoords="offset points",fontsize=8)
    ax.set_xlabel(txt(lang,"Ridge每月MAE（百分点）","Monthly Ridge MAE (pp)")); ax.set_ylabel(txt(lang,"网络每月MAE（百分点）","Monthly CNN MAE (pp)")); ax.grid(alpha=.15)
    ax.set_title(txt(lang,"C  配对比较；对角线下方为网络更好","C  Paired errors; below diagonal = CNN better"),loc="left",fontsize=11)
    ax=axes[1,1]; w=np.array([weights['months'][m]['weights_before_forecast'] for m in months])
    ax.stackplot(x,w.T,colors=[COLORS[k] for k in weights['experts']],labels=[name(k,lang) for k in weights['experts']],alpha=.93)
    ax.set_ylim(0,1); ticks(ax,months,3); ax.set_ylabel(txt(lang,"预测前权重","Weights before prediction")); ax.legend(loc="upper left",fontsize=8,frameon=True,facecolor="white",edgecolor="none",framealpha=.92)
    ax.set_title(txt(lang,"D  基于过去误差的在线专家权重","D  Online weights from past errors"),loc="left",fontsize=11)
    title(fig,txt(lang,"回顾性评估与配对不确定性","Retrospective performance and paired uncertainty"),txt(lang,"2025-01至2026-01，13个月。网络0.2067、Ridge0.2135、沿用上月0.2487；新增方法不是新盲测。","Jan 2025-Jan 2026, 13 months. CNN 0.2067 / Ridge 0.2135 / last month 0.2487 pp. New methods do not have a fresh blind test."))
    save(fig,"05_performance",lang)


def direction(lang):
    f=pd.read_csv(P/"retrospective_predictions.csv"); v=pd.read_csv(P/"validation_predictions_2024.csv")
    actual=f.actual_change_pct.to_numpy(); s=j("retrospective_results.json"); vs=j("validation_results.json"); k="multitask_0.05_weighted"
    probability=f[[f"prob_{k}_{c}" for c in ('fall','flat','rise')]].to_numpy(); predicted=probability.argmax(axis=1)-1
    truth=np.sign(actual).astype(int); conf=np.array([[((truth==a)&(predicted==b)).sum() for b in (-1,0,1)] for a in (-1,0,1)])
    frac=conf/conf.sum(axis=1,keepdims=True)
    fig,axes=plt.subplots(1,3,figsize=(16,5.4),gridspec_kw={"width_ratios":[1.15,1,1.1]},layout="constrained")
    ax=axes[0]; methods=["last_month","ridge_25","fair_tcn","decomp_tcn"]
    positions=np.arange(3); width=.18
    for i,m in enumerate(methods):
        errors=np.abs(f["pred_"+m].to_numpy()-actual); values=[errors[actual<0].mean(),errors[actual==0].mean(),errors[actual>0].mean()]
        ax.bar(positions+(i-1.5)*width,values,width,label=name(m,lang),color=COLORS[m])
    ax.set_xticks(positions,[txt(lang,"下跌 n=705","Fall n=705"),txt(lang,"持平 n=45","Flat n=45"),txt(lang,"上涨 n=160","Rise n=160")],fontsize=9)
    ax.set_ylabel(txt(lang,"MAE（百分点）","MAE (pp)")); ax.set_title(txt(lang,"A  按实际方向的幅度误差","A  Magnitude error by actual direction"),fontsize=11,loc="left"); ax.legend(fontsize=8,frameon=False); ax.grid(axis="y",alpha=.15)
    ax=axes[1]; ax.imshow(frac,cmap="Blues",vmin=0,vmax=1)
    for a in range(3):
        for b in range(3): ax.text(b,a,f"{conf[a,b]}\n{frac[a,b]:.1%}",ha="center",va="center",fontsize=10,color="white" if frac[a,b]>.6 else "#233B53")
    labels=[txt(lang,"下跌","Fall"),txt(lang,"持平","Flat"),txt(lang,"上涨","Rise")]
    ax.set_xticks(range(3),labels); ax.set_yticks(range(3),labels); ax.set_xlabel(txt(lang,"分类头预测","Classifier prediction")); ax.set_ylabel(txt(lang,"实际方向","Actual direction")); ax.set_title(txt(lang,"B  方向头混淆矩阵（按实际行归一）","B  Direction head / row-normalized confusion"),fontsize=11)
    ax=axes[2]; methods=['last_month','ridge_25','fair_tcn','decomp_tcn',k]; val_last=0
    # Derive the no-learning validation control from known historical values.
    raw=pd.read_csv(ROOT/'research/data/official_nbs_70city.csv').pivot(index='period',columns='city',values='new_mom_pct')
    periods=pd.period_range('2024-01','2024-12',freq='M'); ya=raw.loc[periods.astype(str)].to_numpy(); yp=raw.loc[(periods-1).astype(str)].to_numpy()
    val_last=float(np.mean([(yp[ya>0]>0).mean(),(yp[ya<0]<=0).mean()]))
    a=[val_last]+[vs[m]['classifier_balanced_binary_rise_accuracy'] if m==k else vs[m]['balanced_binary_rise_accuracy_excluding_actual_zero'] for m in methods[1:]]
    b=[s[m]['classifier_balanced_binary_rise_accuracy'] if m==k else s[m]['balanced_binary_rise_accuracy_excluding_actual_zero'] for m in methods]
    labels=[name(m,lang) if m!=k else txt(lang,'方向分类头','Direction head') for m in methods]
    ax.scatter(a,range(5),color="#233B53",label=txt(lang,"2024验证","2024 validation"),s=45)
    ax.scatter(b,range(5),color="#D07745",label=txt(lang,"回顾性","Retrospective"),s=45)
    for i in range(5): ax.plot([a[i],b[i]],[i,i],color="#D8E0E4",zorder=0)
    ax.set_yticks(range(5),labels,fontsize=9); ax.invert_yaxis(); ax.set_xlim(.45,.82); ax.axvline(.5,color="#AAB7BF",ls="--"); ax.grid(axis="x",alpha=.15)
    ax.set_xlabel(txt(lang,"涨跌平衡准确率","Balanced rise/non-rise accuracy")); ax.set_title(txt(lang,"C  跨阶段方向表现","C  Direction performance across periods"),fontsize=11,loc="left"); ax.legend(fontsize=8,frameon=False)
    title(fig,txt(lang,"类别不平衡与方向辅助任务","Class imbalance and auxiliary direction learning"),txt(lang,"二元指标排除实际持平；分类头独立于回归输出。四个方向任务均未通过2024验证门槛。","Binary metrics exclude actual flat cases. The classifier is separate from regression; all four multitask cases failed the 2024 direction gate."))
    save(fig,"06_direction",lang)


def shandong(lang):
    f=pd.read_csv(P/'shandong_retrospective.csv'); fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
    for ax,city,english in zip(axes.flat,('济南','青岛','烟台','济宁'),('Jinan','Qingdao','Yantai','Jining')):
        part=f[f.city.eq(city)].sort_values('target_month'); months=list(part.target_month); x=np.arange(13); actual=part.actual_change_pct.to_numpy()
        ax.plot(x,actual,color='#D07745',label=txt(lang,'实际','Actual'),lw=2.1,marker='o',markersize=3)
        for k in ('ridge_25','fair_tcn'): ax.plot(x,part['pred_'+k],color=COLORS[k],label=name(k,lang),lw=1.7,ls='--' if k=='fair_tcn' else '-')
        ax.axvspan(4.5,7.5,color='#F3EEE5',zorder=0); ax.axhline(0,lw=.7,color='#ABB8C0'); ax.grid(alpha=.15); ticks(ax,months,3)
        r=float(np.abs(part.pred_ridge_25-actual).mean()); nn=float(np.abs(part.pred_fair_tcn-actual).mean()); b=float(np.abs(part.pred_last_month-actual).mean())
        ax.set_title(txt(lang,city,english),loc='left',fontweight='bold')
        ax.text(.98,.96,txt(lang,f'MAE  基线 {b:.4f}\nRidge {r:.4f} / 网络 {nn:.4f}',f'MAE  baseline {b:.4f}\nRidge {r:.4f} / CNN {nn:.4f}'),ha='right',va='top',transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','edgecolor':'#E0E8EC','alpha':.9,'pad':4})
        ax.set_ylabel(txt(lang,'新房指数月环比（%）','New-home index MoM (%)'))
    axes[0,0].legend(ncol=3,loc='lower left',fontsize=8,frameon=False)
    title(fig,txt(lang,'山东四城的实际值与逐月预测','Actual values and monthly predictions in four Shandong cities'),txt(lang,'阴影为2025年6—8月；济宁的学习模型未超过沿用上月，济南与青岛Ridge优于网络。','Shading: Jun-Aug 2025. Learned models underperform last month in Jining; Ridge beats CNN in Jinan and Qingdao.'))
    save(fig,'07_shandong',lang)


def robustness(lang):
    segments=j('segment_diagnostics.json'); q=j('quarterly_causal_ensemble_check.json')['quarters']; fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained')
    methods=['last_month','ridge_25','fair_tcn','online_ensemble']; ax=axes[0]; x=np.arange(4); width=.18
    for i,k in enumerate(methods): ax.bar(x+(i-1.5)*width,[segments[f'2025_Q{n}'][k]['mae_pp'] for n in range(1,5)],width,color=COLORS[k],label=name(k,lang))
    ax.set_xticks(x,['Q1','Q2','Q3','Q4']); ax.set_ylim(0,.34); ax.set_ylabel(txt(lang,'MAE（百分点）','MAE (pp)')); ax.legend(loc='upper right',fontsize=8,frameon=False); ax.grid(axis='y',alpha=.15); ax.set_title(txt(lang,'A  2025各季度误差','A  Quarterly errors in 2025'),fontsize=11)
    ax=axes[1]
    for i,k in enumerate(methods):
        a,b=segments['2025_only'][k]['mae_pp'],segments['2026_Jan_only'][k]['mae_pp']; ax.plot([a,b],[i,i],color='#D8E0E4'); ax.scatter(a,i,color='#233B53',s=50,label=txt(lang,'2025全年','2025 full year') if i==0 else None); ax.scatter(b,i,color='#D07745',s=50,label=txt(lang,'2026年1月','Jan 2026') if i==0 else None)
    ax.set_yticks(range(4),[name(k,lang) for k in methods],fontsize=9); ax.invert_yaxis(); ax.set_xlabel(txt(lang,'MAE（百分点）','MAE (pp)')); ax.legend(fontsize=8,frameon=False); ax.grid(axis='x',alpha=.15); ax.set_title(txt(lang,'B  基期调整月份敏感性','B  Rebased-month sensitivity'),fontsize=11)
    ax=axes[2]; x=np.arange(3)
    ax.plot(x,[r['ridge25_mae_pp'] for r in q],marker='o',color=COLORS['ridge_25'],label=txt(lang,'固定Ridge25配置','Fixed Ridge-25 configuration'))
    ax.plot(x,[r['online_mae_pp'] for r in q],marker='o',color=COLORS['online_ensemble'],label=txt(lang,'按过去季度选择专家','Past-quarter expert selection'))
    ax.set_xticks(x,['2024 Q2','2024 Q3','2024 Q4']); ax.set_ylabel(txt(lang,'MAE（百分点）','MAE (pp)')); ax.legend(fontsize=8,frameon=False); ax.grid(alpha=.15); ax.set_title(txt(lang,'C  只用过去季度选择组合','C  Past-only quarterly ensemble selection'),fontsize=11)
    title(fig,txt(lang,'分阶段稳定性与选择过程检查','Period sensitivity and checks of the selection process'),txt(lang,'2026年1月官方基期与分类权重调整；单月结果不代表长期规律。季度选择使用原始3种子预测。','NBS changed the comparison base and category weights in Jan 2026. One month is not a long-term result. Quarterly selection uses original 3-seed predictions.'))
    save(fig,'08_robustness',lang)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--language',choices=['zh','en','both'],default='both'); args=parser.parse_args()
    for lang in (['zh','en'] if args.language=='both' else [args.language]):
        setup(lang)
        for fun in (market,protocol,architecture,validation,performance,direction,shandong,robustness):
            fun(lang); print(lang,fun.__name__,flush=True)
    write_manifest()


def write_manifest():
    """Record the exact inputs and exported PNG/SVG files, without absolute paths."""
    sources = [ROOT/'scripts/build_figures.py', ROOT/'research/research_models.py',
               ROOT/'research/data/official_nbs_70city.csv']
    sources += [P/f for f in ('locked_experiment_registry.json','validation_screening_results.json',
                             'validation_screening_three_seeds.csv','validation_results.json',
                             'retrospective_predictions.csv','retrospective_results.json',
                             'gain_vs_strong_baseline.json','ensemble_weights.json',
                             'segment_diagnostics.json','quarterly_causal_ensemble_check.json',
                             'shandong_retrospective.csv')]
    figures = sorted((ROOT/'docs/figures').glob('*/*.png')) + sorted((ROOT/'docs/figures').glob('*/*.svg'))
    files = sources + figures
    manifest = {'generation': 'recorded outputs; no retraining or score modification',
                'figure_types': 8, 'languages': ['zh','en'], 'formats': ['png','svg'],
                'sha256_by_file': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files}}
    (ROOT/'docs/figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
