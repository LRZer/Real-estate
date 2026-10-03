"""Create an eight-page Chinese research presentation from verified outputs."""
from pathlib import Path
import json
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
R = ROOT / "results/research"
DEST = ROOT / "output/pdf/70城新房指数预测_成果展示.pdf"
NAVY, TEAL, ORANGE, MUTED, PALE = map(HexColor, ("#233B53", "#007F82", "#D07745", "#566C78", "#EDF3F5"))
W, H = A4
M, CW = 46, A4[0]-92


def data(name): return json.loads((R / name).read_text(encoding="utf-8"))


def main():
    pdfmetrics.registerFont(TTFont("Chinese", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ChineseBold", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))
    pdfmetrics.registerFontFamily("Chinese", normal="Chinese", bold="ChineseBold", italic="Chinese", boldItalic="ChineseBold")
    styles = {
        "body": ParagraphStyle("body", fontName="Chinese", fontSize=10, leading=16, textColor=NAVY, wordWrap="CJK", spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Chinese", fontSize=8.5, leading=13, textColor=MUTED, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName="Chinese", fontSize=8.7, leading=13, textColor=NAVY, wordWrap="CJK"),
        "white": ParagraphStyle("white", fontName="Chinese", fontSize=11, leading=18, textColor=white, wordWrap="CJK"),
    }
    val, screen, test = data("validation_results.json"), data("validation_screening_results.json"), data("retrospective_results.json")
    selection, segments = data("selection_before_retrospective.json"), data("segment_diagnostics.json")
    direction = selection["direction_representative"]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(DEST), pagesize=A4)
    c.setTitle("中国70城新房指数自适应预测 - 算法与实验成果")
    c.setAuthor("房地产人工智能项目")
    page_no = 0
    def p(text, y, style="body", x=M, width=CW):
        para = Paragraph(text, styles[style]); _, h = para.wrap(width, H)
        if y-h < 55: raise ValueError(f"Page {page_no}: text exceeds bottom margin")
        para.drawOn(c, x, y-h); return y-h-9
    def heading(text, y):
        c.setFont("ChineseBold", 13); c.setFillColor(TEAL); c.drawString(M, y-13, text); return y-29
    def page(title, subtitle):
        nonlocal page_no
        page_no += 1
        c.setFillColor(NAVY); c.rect(0, H-13, W, 13, fill=1, stroke=0)
        c.setFillColor(MUTED); c.setFont("Chinese", 8); c.drawString(M, H-42, "中国70城新房指数预测  /  算法研究成果")
        c.setFillColor(NAVY); c.setFont("ChineseBold", 21); c.drawString(M, H-82, title)
        y = p(subtitle, H-100, "small")
        return y-12
    def end():
        c.setStrokeColor(HexColor("#CBD6DC")); c.line(M, 42, W-M, 42)
        c.setFont("Chinese", 8); c.setFillColor(MUTED)
        c.drawString(M, 27, "数据截点 2026.02.28  |  本轮完成 2026.10.04  |  回顾性研究")
        c.drawRightString(W-M, 27, f"{page_no} / 8"); c.showPage()
    def picture(filename, y, width=CW, max_height=310):
        image = ImageReader(str(filename)); iw, ih = image.getSize()
        height = min(width*ih/iw, max_height); width = height*iw/ih
        if y-height < 55: raise ValueError(f"Page {page_no}: image exceeds bottom margin")
        c.drawImage(image, M+(CW-width)/2, y-height, width, height, mask="auto")
        return y-height-12
    def table(rows, widths, y, size=8.7):
        content = [[Paragraph(str(cell), styles["cell"]) for cell in row] for row in rows]
        t = Table(content, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0),PALE), ("ROWBACKGROUNDS", (0,1),(-1,-1),[white,HexColor("#F7FAFB")]),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("LEFTPADDING",(0,0),(-1,-1),8),
            ("RIGHTPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LINEBELOW",(0,0),(-1,0),.5,HexColor("#CBD6DC")),
        ]))
        _, h = t.wrap(CW, H)
        if y-h < 55: raise ValueError(f"Page {page_no}: table exceeds bottom margin")
        t.drawOn(c, M, y-h); return y-h-14
    def link(title, url, y):
        para = Paragraph(f'<link href="{url}" color="#007F82">{title}</link>', styles["small"])
        _, h = para.wrap(CW,H); para.drawOn(c,M,y-h); return y-h-7

    y = page("中国70城新房指数自适应预测", "面向房地产开发商区域市场研判的公开数据人工智能项目")
    c.setFillColor(NAVY); c.roundRect(M, y-96, CW, 96, 9, fill=1, stroke=0)
    p("研究问题<br/>当市场阶段发生变化，神经网络能否公平地利用历史信息，<br/>并在只使用已公布数据的条件下改善下一统计月指数预测？", y-16, "white", x=M+17, width=CW-34)
    y -= 120
    card_w = (CW-20)/3
    for i,(num, text) in enumerate((("3,990", "官方城市月度记录"), ("22", "预先限定的研究配置"), ("1,697", "主展示网络参数"))):
        x = M+i*(card_w+10); c.setFillColor(PALE); c.roundRect(x,y-90,card_w,90,7,fill=1,stroke=0)
        c.setFillColor(TEAL); c.setFont("ChineseBold",25); c.drawString(x+13,y-37,num)
        c.setFont("Chinese",9); c.setFillColor(MUTED); c.drawString(x+13,y-66,text)
    y -= 113
    y = heading("已取得的成果", y)
    y = p("五个随机种子平均的神经网络回顾性MAE为 <b>0.2067 个百分点</b>，较沿用上月降低 <b>16.9%</b>。七个月输入与相同上下文缩小了原型网络的验证差距。", y)
    y = p("已完成强基线、窗口归一化、近期偏差校正、全国与城市双分支、方向辅助任务和在线组合的对照，保存了权重、逐月预测与训练截止记录。", y)
    y = heading("研究判断", y)
    y = p("<b>主预测模型仍选逐月Ridge。</b> 新网络验证期未通过替换门槛；它相对Ridge的回顾性收益区间跨过0。项目展示方法设计、完整实验和误差分析能力。", y)
    y = p("业务场景来自2026年3月山东房地产开发商实习。当前版本在2026年9月至10月重建和完善，信息截点模拟2026年2月底；未使用公司客户或销售数据。", y, "small")
    end()

    y = page("数据与真实可用时间", "指数变化是研究目标；发布日期决定每个预测起点能够看到的信息。")
    y = heading("官方面板",y)
    y = table([["项目","定义"],["覆盖范围","2021-05至2026-01，57个月 × 70城 = 3990条"],["字段","月份、城市、新房环比、二手房环比、来源链接"],["完整性","每月70城；无缺失，无城市月份重复"],["目标转换","官方上月=100的指数减100；99.7对应 -0.3%"],["山东案例","济南、青岛、烟台、济宁"]],[115,CW-115],y)
    y = heading("逐月只用已经公布的数据",y)
    y = table([["阶段","目标月份","第一个训练截止"],["滚动验证","2024-01至2024-12","标签至2023-12，1750行"],["回顾性评估","2025-01至2026-01","标签至2024-12，2590行"],["截点预测","2026-02","标签至2026-01，3500行"]],[100,185,CW-285],y)
    y = p("例如2026年1月指数于 <b>2026年2月13日</b> 发布，收到月报后预测2月整月结果。因此这是进入目标月后的月度指数预测，不能说2月1日就已知1月指数。标准化、训练和状态更新均遵守这一公布顺序。",y)
    y = p("数据中的旧时期也有下跌：70城等权平均新房月环比在2022、2024、2025年分别为-0.196%、-0.493%、-0.260%。这些是面板月环比的描述性平均，不能当作年度涨幅或官方全国指数。",y)
    y = p("2026年1月基期和分类权重调整，因此单独报告2025全年。原网页在2026年9月重新采集，保留来源和文件校验值，但没有原始发布日网页快照。",y,"small")
    link("国家统计局：2026年1月70城价格指数与统计附注", "https://www.stats.gov.cn/sj/zxfb/202602/t20260213_1962617.html",y)
    end()

    y = page("输入公平的神经预测网络", "短面板采用小型网络；同等已知信息、固定预算和随机种子构成比较基础。")
    y = picture(R / "model_architecture.png",y,max_height=265)
    y = table([["模块","结构与规模"],["序列输入","70城 × 7月 × 2通道，覆盖t-6至t"],["数值上下文","当前值2 + 面板统计3 + 周期2 + 滞后8 + 滚动均值4"],["时序编码","Conv1d 2→12→12，核3，ReLU，时间平均池化"],["预测头","12维时序 + 19项特征 + 4维城市嵌入 → 24维 → 1维修正"],["最终输出","上一期新房变化 + 预测修正量；共1697参数"]],[108,CW-108],y)
    y = p("训练使用Huber损失（beta=0.25）、AdamW（学习率0.006、权重衰减0.02）、梯度裁剪1、Dropout0.15。每月固定60轮，输出层零初始化。筛选种子13/31/47；最终网络复核13/31/47/61/79，取预测平均。",y)
    y = p("输入补齐还改变了上下文预测头，因此结果支持这一完整配置有效，不能只归因于多加一个月份。城市嵌入与Ridge的独热编码都含城市身份，但参数化不同。",y,"small")
    end()

    y = page("适应市场变化的研究设计", "20个单模型配置与2种组合已预先限定；复杂方法是否有效由验证决定。")
    y = table([["方法","本项目的实现与对照"],["强基线与输入对照 8组","Ridge alpha=5/25/100；树模型7/15叶；NLinear形式；六个月原型；七个月上下文网络"],["窗口归一化 1组","只由已知窗口计算每城均值与方差；标准差下限0.05；仅按新房通道恢复修正量尺度"],["近期偏差校正 4组","以过去滚动样本外预测误差更新EWMA；rho=0.5/0.8，修正强度=0.5/1"],["全国与城市双分支 3组","g=70城均值，r=y-g；预测g_hat与零均值r_hat相加；线性、神经、归一化对照"],["方向辅助任务 4组","共享城市表示增加涨/平/跌三分类；eta=0.01/0.05，过去频率加权或不加权"],["专家组合 2组","最强简单基线 + 最优非线性网络 + 沿用上月；均匀或过去风险加权"]],[135,CW-135],y)
    y = heading("损失与更新顺序",y)
    y = p("双分支损失：城市最终输出Huber + 0.25 × 全国均值Huber。方向任务额外增加 eta × 三分类交叉熵；训练类别权重从过去标签计算，最大权重截到3。分类头不会强行覆盖回归值。",y)
    y = p("偏差 e = 70城平均(预测 - 真实)，状态 b ← rho × b + (1-rho) × e；下一次预测减去强度 × 旧b。组合风险同样先预测再更新，预测前权重 = softmax(-过去风险 / 0.02)。",y)
    y = p("全国分支只用过去训练月份的真实均值构造监督；预测用预测均值。历史价格、全国同期统计、类别权重和归一化均不读取目标月结果。",y,"small")
    y = p("以下方法参考相应研究，是本项目的小面板独立适配；未复现论文完整结构或原数据集成绩，未声称首创这些通用方法。",y,"small")
    y = link("参考：RevIN（ICLR2022）作者实现", "https://github.com/ts-kim/RevIN",y)
    y = link("参考：NLinear（AAAI2023）作者实现", "https://github.com/cure-lab/LTSF-Linear",y)
    link("参考：OneNet（NeurIPS2023）论文", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html",y)
    end()

    y = page("验证结果与模型选择", "用2024年选择配置，之后锁定代表模型；全部20个配置和失败结果均保留。")
    y = picture(R / "validation_ablation.png",y,max_height=290)
    y = p("三种子下，六个月原型验证MAE <b>0.2951</b>，输入补齐网络 <b>0.2791</b>，Ridge <b>0.2781</b>。五种子网络复核为 <b>0.2803</b>；它接近Ridge，但没有稳定验证优势。",y)
    y = table([["保留门槛","验证结果"],["主预测替换","MAE至少降低3%；至少8个月改善；任一半年误差恶化≤5%"],["神经候选","五种子验证比Ridge误差高0.82%；仅3个月胜出，未通过"],["偏差校正","最佳验证0.2760，改善0.75%，未过3%门槛"],["方向专项","四个配置均未同时满足平衡准确率、上涨误差和总体误差约束"]],[125,CW-125],y)
    y = p("季度检查只用此前季度的三种子记录选择下季度配置，排除五种子全年复核结果。第一季度选择Ridge alpha=100，之后选择Ridge alpha=25；没有靠季度换模型获得持续优势。",y,"small")
    end()

    y = page("回顾性表现与不确定性", "2025-01至2026-01共13个月、910条城市月份记录；已查看的历史区间不构成新盲测。")
    main_names = [("沿用上月","last_month"),("逐月Ridge 主模型","ridge_25"),("输入补齐网络 五种子","fair_tcn"),("偏差校正Ridge","bias_0.5_0.5"),("双分支Ridge","decomp_ridge"),("在线组合","online_ensemble")]
    y = table([["方法","总体MAE","2025夏季MAE"]]+[[name,f"{test[n]['mae_pp']:.4f}",f"{segments['2025_Jun_Aug'][n]['mae_pp']:.4f}"] for name,n in main_names],[245,125,CW-370],y)
    y = picture(R / "monthly_performance.png",y,max_height=265)
    y = p("网络相对Ridge总体误差下降3.2%，夏季下降14.6%；但相对Ridge的平均MAE收益三个月区块95%区间为 <b>[-0.0049, 0.0206]</b> 个百分点，跨0。仅13个月且研究者已查看该区间，不能据此宣称稳定优势。",y)
    y = p("双分支Ridge的0.2032是本轮代表模型中的最低回顾性误差，但其2024验证较差，未追认为主模型。在线组合也保留其专家选择不确定性。",y,"small")
    end()

    y = page("山东案例与上涨样本难点", "总误差改善不代表每个城市和每种市场方向都改善。")
    y = picture(R / "shandong_cases.png",y,max_height=260)
    sd = pd.read_csv(R / "shandong_metrics.csv")
    rows = [["城市","沿用上月","主模型Ridge","神经网络"]]
    for city,g in sd.groupby("city"):
        indexed=g.set_index("case"); rows.append([city]+[f"{indexed.loc[n,'mae_pp']:.4f}" for n in ("last_month","ridge_25","fair_tcn")])
    y = table(rows,[110,130,130,CW-370],y)
    y = p("910条回顾性记录中有705条下跌、160条上涨、45条持平。上涨MAE：沿用上月 <b>0.2888</b>，Ridge <b>0.3491</b>，新网络 <b>0.3247</b>。网络改善Ridge，但上涨仍不及简单基线。",y)
    y = p(f"方向头回顾性上涨召回率 {test[direction]['classifier_rising_recall']:.1%}、精确率 {test[direction]['classifier_rising_precision']:.1%}；2024上涨召回率只有 {val[direction]['classifier_rising_recall']:.1%}。该任务未通过验证方向门槛，保留为独立研究对照。",y)
    y = p("济宁未完全超过沿用上月，济南和青岛Ridge比新网络更好。全国均值与城市残差误差、转涨样本和各季度分段结果均已保存。",y,"small")
    end()

    y = page("可复现交付与应用范围", "成果包含算法、数据、检查证据和展示材料；公开指标研究尚未验证实际销售收益。")
    y = table([["交付","内容"],["公开数据","3990条清洗数据，57期来源索引与SHA256"],["研究配置","20个单模型+2种组合，完整2024验证，锁定代表评估"],["模型权重","Ridge系数及标准化统计；5个神经模型权重"],["运行检查","315条截止审计；1820条最终城市月份输出核验"],["防泄漏验证","8类代表模型扰动目标及未来信息不改变当月预测；先预测后更新"],["无重训复现","保存权重重算2026-02预测，最大差异小于2×10^-7个百分点"],["研究材料","最终实验报告、算法说明、山东案例、面试介绍及来源"]],[130,CW-130],y)
    y = heading("截点输出示例",y)
    forecasts = pd.read_csv(R / "forecast_2026_02.csv")
    rows=[["城市","2026-02 Ridge预测","神经模型预测"]]
    for city in ("济南","青岛","烟台","济宁"):
        row = forecasts.loc[forecasts.city.eq(city)].iloc[0]
        rows.append([city,f"{row.pred_ridge_25:.3f}%",f"{row.pred_fair_tcn:.3f}%"])
    y = table(rows,[110,200,CW-310],y)
    y = p("以上仅为2026年2月底信息集下的月度指数预测，2月实际值未用于评分。安装项目依赖后，执行 <b>predict_saved_models.py</b> 即可从权重复现；无需重新训练。",y)
    y = heading("可用于申请展示的能力",y)
    y = p("官方数据治理、特征与时点设计、PyTorch小型网络实现、结构消融、过去数据适应、种子复核、区块误差分析和完整复现。项目是围绕实习业务的公开数据研究，尚无公司级销量、楼盘定价或部署效果评估。",y)
    y = p("实际完成日期为2026年10月4日；模拟的信息截点为2026年2月28日。对外展示保持二者区分。详细公式、20个配置和分段得分见《最终实验报告》。",y,"small")
    end()
    assert page_no == 8
    c.save()
    print(str(DEST))


if __name__ == "__main__": main()
