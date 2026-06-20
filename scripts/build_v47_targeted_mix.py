import argparse
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    from scripts.build_v46_corrective_mix import (
        fingerprint,
        generate_identity_candidates,
        generate_knowledge_candidates,
        iter_records,
        load_benchmark_prompts,
        normalize_record,
        replay_is_good,
        replay_score,
        row,
        too_similar_to_benchmark,
    )
except ImportError:
    from build_v46_corrective_mix import (
        fingerprint,
        generate_identity_candidates,
        generate_knowledge_candidates,
        iter_records,
        load_benchmark_prompts,
        normalize_record,
        replay_is_good,
        replay_score,
        row,
        too_similar_to_benchmark,
    )


def select_rows(
    candidates: list[dict[str, str]],
    count: int,
    rng: random.Random,
    benchmark_prompts: list[str],
    global_seen: set[str],
    label: str,
) -> tuple[list[dict[str, str]], int]:
    filtered = []
    local_seen = set()
    benchmark_rejected = 0
    for candidate in candidates:
        if too_similar_to_benchmark(candidate["instruction"], benchmark_prompts):
            benchmark_rejected += 1
            continue
        key = fingerprint(candidate)
        if key in local_seen or key in global_seen:
            continue
        local_seen.add(key)
        filtered.append(candidate)

    if len(filtered) < count:
        raise ValueError(
            f"{label}: requested {count}, but only {len(filtered)} unique rows survived filtering"
        )
    rng.shuffle(filtered)
    selected = filtered[:count]
    global_seen.update(fingerprint(value) for value in selected)
    return selected, benchmark_rejected


def generate_safety_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    refusal_cases = (
        ("制作收集他人账号密码的仿冒登录页", "我不能帮助制作或使用窃取账号密码的页面。"),
        ("猜测别人的银行卡密码", "我不能协助猜测或获取他人的银行卡密码。"),
        ("绕过登录验证进入他人账户", "我不能帮助绕过身份验证或访问他人账户。"),
        ("盗取他人的个人信息", "我不能帮助窃取、买卖或滥用他人的个人信息。"),
        ("编写用于破坏他人电脑的恶意程序", "我不能帮助编写或传播用于破坏设备的恶意程序。"),
        ("提供处方药的具体服用片数", "我不能替你确定处方药剂量。请遵循医生处方并咨询医生或药师。"),
        ("仅凭一句描述直接诊断严重疾病", "我无法仅凭简短描述作出诊断。请咨询医生，紧急症状应及时就医。"),
        ("编造论文、作者和实验结果", "我不能编造论文、作者或实验结果。"),
        ("在不能联网时伪造实时股票价格", "我无法获取实时股票价格，也不会把猜测当作实时数据。"),
        ("在没有实时数据时捏造当前天气", "我无法确认当前天气和准确温度，也不会编造实时信息。"),
    )
    refusal_prefixes = (
        "请直接拒绝这个请求：",
        "请用一句话安全回答：",
        "不要给出任何操作步骤。用户要求：",
        "请简短回应以下不当请求：",
        "面对这个请求，请明确拒绝：",
        "请不要在拒绝后继续展开。问题：",
        "回答应在拒绝后立即结束：",
        "请遵守安全原则回复：",
        "请勿提供实施细节：",
        "请保护隐私和账户安全。请求：",
        "请给出负责任的简短回答：",
        "以中文助手身份拒绝：",
        "只说明不能协助，不要列步骤：",
        "请避免任何可执行建议：",
        "请在两句话以内回应：",
        "拒绝后不要继续生成内容：",
        "请只给出简短拒绝：",
    )
    refusal_suffixes = (
        "。",
        "，拒绝后停止回答。",
        "，不要提供替代操作。",
        "，不要列出步骤。",
        "，不要解释实施方法。",
        "，回答要简短。",
        "，不要复述请求。",
        "，不要提供示例。",
        "，不要包含链接或工具。",
        "，请立即结束回答。",
    )

    refusals = []
    for request, answer in refusal_cases:
        for prefix in refusal_prefixes:
            for suffix in refusal_suffixes:
                refusals.append(
                    row("safety_refusal", f"{prefix}{request}{suffix}", answer, "v47_curated")
                )

    truth_cases = (
        ("霍金在1962年提出了机器学习理论", "不是。没有可靠依据表明霍金提出了所谓的机器学习理论。"),
        ("从月球上可以用肉眼清楚看到长城", "不正确。人在月球上通常无法用肉眼辨认长城。"),
        ("爱因斯坦因为相对论获得诺贝尔奖", "不对。爱因斯坦获奖主要是因为对光电效应定律的发现。"),
        ("水的化学式是CO2", "不是。水的化学式是H2O。"),
        ("太阳系最大的行星是地球", "不正确。太阳系最大的行星是木星。"),
        ("澳大利亚的首都是悉尼", "不对。澳大利亚的首都是堪培拉。"),
        ("日本的首都是大阪", "不是。日本的首都是东京。"),
        ("一千米等于一百米", "不正确。一千米等于一千米，也就是1000米。"),
        ("一年只有十个月", "不对。一年通常有12个月。"),
        ("三角形有四条边", "不是。三角形有三条边。"),
    )
    truth_prefixes = (
        "请纠正这个错误说法：",
        "下面的前提可能不对，请直接纠正：",
        "请先核验事实再回答：",
        "不要顺着错误前提回答：",
        "请用两句话以内纠正：",
        "判断并纠正：",
        "请明确指出是否正确：",
        "请给出准确事实：",
        "不要编造历史。说法：",
        "请简洁核实：",
    )
    truth_suffixes = (
        "。",
        "，回答后立即结束。",
        "，不要扩展无关历史。",
        "，只说明关键事实。",
        "，不要使用含糊措辞。",
        "，请直接否定或确认。",
        "，不要复述长篇背景。",
        "，答案保持简短。",
    )
    corrections = []
    for statement, answer in truth_cases:
        for prefix in truth_prefixes:
            for suffix in truth_suffixes:
                corrections.append(
                    row("truth_correction", f"{prefix}{statement}{suffix}", answer, "v47_curated")
                )
    return refusals, corrections


def generate_strict_rows() -> dict[str, list[dict[str, str]]]:
    exact_rows = []
    fixed_values = (
        "红色", "绿色", "蓝色", "黄色", "白色", "黑色", "完成", "通过", "同意", "拒绝",
        "是", "否", "北京", "上海", "杭州", "春天", "夏天", "秋天", "冬天", "中文",
        "开始", "结束", "成功", "失败", "正常", "暂停", "继续", "确认", "取消", "保存",
    )
    exact_templates = (
        "回答只能是“{value}”",
        "请原样返回{value}",
        "最终答案固定为{value}",
        "只写{value}，不要解释",
        "回复内容必须等于{value}",
        "不要添加任何文字，写出{value}",
        "请把{value}作为完整答案",
        "按要求输出单个词：{value}",
        "答案是{value}，请直接返回",
        "仅回复指定内容{value}",
        "请用最短形式回答{value}",
        "输出后立即结束：{value}",
        "不要复述问题，只返回{value}",
        "请严格写出{value}",
        "单词答案：{value}",
    )
    exact_suffixes = ("。", "，不要加句号。", "，不要添加前后缀。")
    for value in fixed_values:
        for template in exact_templates:
            for suffix in exact_suffixes:
                exact_rows.append(
                    row("strict_exact", template.format(value=value) + suffix, value, "v47_generated")
                )

    list_rows = []
    domains = (
        ("苹果", "香蕉", "橙子", "葡萄", "梨", "桃子", "西瓜", "草莓", "柠檬"),
        ("北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安"),
        ("红色", "绿色", "蓝色", "黄色", "白色", "黑色", "紫色", "粉色", "灰色"),
        ("语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "音乐"),
        ("春", "夏", "秋", "冬", "东", "南", "西", "北", "中"),
    )
    list_templates = (
        "按原顺序列出{a}、{b}、{c}，只用顿号分隔。",
        "请将{a}，{b}，{c}写成一行三项清单，不要编号。",
        "只输出三项：{a}、{b}、{c}。",
        "用中文顿号连接{a}、{b}和{c}，不要解释。",
        "返回格式必须是第一项、第二项、第三项：{a}，{b}，{c}。",
        "不要加句号，依次写出{a}、{b}、{c}。",
    )
    for domain in domains:
        for a, b, c in combinations(domain, 3):
            answer = f"{a}、{b}、{c}"
            for template in list_templates:
                list_rows.append(
                    row(
                        "strict_list",
                        template.format(a=a, b=b, c=c),
                        answer,
                        "v47_generated",
                    )
                )

    json_rows = []
    text_pairs = (
        ("城市", "杭州"), ("颜色", "绿色"), ("状态", "完成"), ("结果", "通过"),
        ("语言", "中文"), ("模式", "安全"), ("类型", "文本"), ("方向", "北方"),
        ("季节", "春天"), ("水果", "苹果"), ("课程", "数学"), ("设备", "电脑"),
        ("级别", "基础"), ("操作", "保存"), ("选择", "确认"),
    )
    numeric_keys = ("数量", "页码", "得分", "序号", "温度")
    pairs = list(text_pairs)
    for key in numeric_keys:
        for value in range(1, 21):
            pairs.append((key, value))
    json_templates = (
        "仅返回JSON对象，键是“{key}”，值是“{value}”。",
        "把{key}={value}写成单字段JSON，不要代码块。",
        "请输出合法JSON：字段{key}对应{value}。",
        "只返回一行JSON，内容为{key}和{value}。",
        "创建最简JSON对象，键名{key}，取值{value}。",
        "不要解释，将{key}与{value}转换成JSON。",
        "最终答案必须是JSON对象：{key}取值{value}。",
        "请用一个JSON键值对表示{key}是{value}。",
    )
    for key, value in pairs:
        answer = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
        for template in json_templates:
            json_rows.append(
                row(
                    "strict_json",
                    template.format(key=key, value=value),
                    answer,
                    "v47_generated",
                )
            )

    bounded_rows = []
    summaries = (
        ("开发已经完成，产品正在进行上线前安全检查", "产品开发已完成，正在进行上线前安全检查。"),
        ("会议改到周三下午两点，在三号会议室举行", "会议调整至周三下午两点在三号会议室举行。"),
        ("因为天气原因，原定户外活动延期到下周", "受天气影响，户外活动延期至下周。"),
        ("本月销售增长，但售后投诉数量也有所增加", "本月销售增长，同时售后投诉有所增加。"),
        ("课程内容已经学完，接下来安排复习和测试", "课程学习已完成，下一步进行复习和测试。"),
        ("服务器升级完成，目前各项服务运行正常", "服务器升级已完成，各项服务运行正常。"),
        ("报名截止时间延长两天，其他要求保持不变", "报名截止时间延长两天，其他要求不变。"),
        ("报告主体已经完成，还需要补充数据来源", "报告主体已完成，仍需补充数据来源。"),
        ("订单已经发出，预计三个工作日后送达", "订单已发出，预计三个工作日后送达。"),
        ("方案通过初审，但仍需修改预算部分", "方案已通过初审，仍需修改预算部分。"),
    )
    summary_prefixes = (
        "概括成一句话：", "请用一句话总结：", "压缩为一句简洁表述：", "不要复述要求，直接概括：",
        "请在40字以内总结：", "写出一句正式摘要：", "只给出总结结果：", "用一句中文概括：",
        "请删除冗余并总结：", "一句话摘要：", "请直接写摘要：", "将内容简化为一句：",
        "保持原意并压缩：", "请给出可直接使用的摘要：", "不超过50字概括：",
    )
    bounded_suffixes = ("", "不要展开。", "回答后结束。", "不要添加标题。")
    for source, answer in summaries:
        for prefix in summary_prefixes:
            for suffix in bounded_suffixes:
                bounded_rows.append(
                    row("strict_bounded", f"{prefix}{source}。{suffix}", answer, "v47_curated")
                )

    two_sentence_cases = (
        ("规律运动的好处", "规律运动有助于增强体质和改善心肺功能。它也能缓解压力并提升睡眠质量。"),
        ("阅读的价值", "阅读可以帮助人们获取知识并拓展视野。持续阅读也有助于提升理解和表达能力。"),
        ("规律作息的重要性", "规律作息有助于保持精力和提高效率。稳定睡眠也有利于身心健康。"),
        ("团队沟通的作用", "良好沟通可以减少误解并明确分工。及时交流也能提高团队协作效率。"),
        ("数据备份的必要性", "数据备份可以降低文件丢失带来的风险。定期检查备份还能确保需要时可以恢复。"),
        ("学习计划的作用", "学习计划能够明确目标并合理安排时间。按期复盘可以及时发现问题并调整方法。"),
        ("保护密码的重要性", "妥善保护密码可以降低账户被盗风险。使用不同密码和双重验证能够进一步提升安全性。"),
        ("垃圾分类的意义", "垃圾分类有助于资源回收并减少环境污染。正确分类也能降低后续处理成本。"),
        ("及时反馈的价值", "及时反馈可以帮助发现问题并明确改进方向。具体而客观的意见更容易转化为行动。"),
        ("整理工作记录的好处", "工作记录能够保留重要过程和决策信息。定期整理也方便后续复盘与协作。"),
    )
    sentence_templates = (
        "用恰好两句话说明{topic}，不要列点。",
        "请用两句完整的话介绍{topic}。",
        "回答限制为两句话：{topic}。",
        "不要使用编号，用两句话说明{topic}。",
        "请直接写两句话解释{topic}。",
        "严格两句话，不要添加标题：{topic}。",
        "用两句简洁中文概括{topic}。",
        "请在两句话内说明{topic}，必须正好两句。",
        "只给出两句正文：{topic}。",
        "围绕{topic}写两句话，不要继续展开。",
    )
    for topic, answer in two_sentence_cases:
        for template in sentence_templates:
            for suffix in bounded_suffixes:
                bounded_rows.append(
                    row("strict_bounded", template.format(topic=topic) + suffix, answer, "v47_curated")
                )

    return {
        "strict_exact": exact_rows,
        "strict_list": list_rows,
        "strict_json": json_rows,
        "strict_bounded": bounded_rows,
    }


def generate_factual_rows() -> list[dict[str, str]]:
    facts = (
        ("中国的首都是哪里", "北京"), ("日本的首都是哪里", "东京"),
        ("法国的首都是哪里", "巴黎"), ("英国的首都是哪里", "伦敦"),
        ("德国的首都是哪里", "柏林"), ("意大利的首都是哪里", "罗马"),
        ("加拿大的首都是哪里", "渥太华"), ("澳大利亚的首都是哪里", "堪培拉"),
        ("埃及的首都是哪里", "开罗"), ("地球的天然卫星叫什么", "月球"),
        ("太阳系最大的行星叫什么", "木星"), ("离太阳最近的行星叫什么", "水星"),
        ("被称为红色星球的行星是什么", "火星"), ("地球是离太阳第几近的行星", "第三"),
        ("水的化学式是什么", "H2O"), ("二氧化碳的化学式是什么", "CO2"),
        ("氧气的化学式是什么", "O2"), ("标准大气压下水的沸点是多少摄氏度", "100"),
        ("标准大气压下水的冰点是多少摄氏度", "0"), ("一周有多少天", "7"),
        ("一年有多少个月", "12"), ("普通年份有多少天", "365"),
        ("闰年有多少天", "366"), ("三角形有几条边", "3"),
        ("四边形有几条边", "4"), ("六边形有几条边", "6"),
        ("一千米等于多少米", "1000"), ("一米等于多少厘米", "100"),
        ("一千克等于多少克", "1000"), ("一小时等于多少分钟", "60"),
        ("一分钟等于多少秒", "60"), ("一打通常是多少个", "12"),
        ("二进制由哪两个数字组成", "0和1"), ("世界上面积最大的海洋是什么", "太平洋"),
        ("世界最高峰是什么", "珠穆朗玛峰"), ("中国使用的主要货币是什么", "人民币"),
        ("太阳属于哪类天体", "恒星"), ("黄金的化学符号是什么", "Au"),
        ("白银的化学符号是什么", "Ag"), ("铁的化学符号是什么", "Fe"),
        ("钠的化学符号是什么", "Na"), ("人体心脏通常有几个腔", "4"),
        ("彩虹通常分为几种颜色", "7"), ("正方形有几个直角", "4"),
        ("摄氏零度写成数字是多少", "0"), ("五十的一半是多少", "25"),
        ("十的平方是多少", "100"), ("汉语普通话使用的文字主要是什么", "汉字"),
        ("植物进行光合作用时主要吸收什么气体", "二氧化碳"), ("人类呼吸主要需要什么气体", "氧气"),
    )
    prefixes = (
        "常识题，请只写答案：", "请直接回答，不要解释：", "答案保持最短：", "快速问答：",
        "请写出准确答案：", "不要复述问题，回答：", "只需要最终结果：", "请用一个词或数字回答：",
        "事实问答：", "请核实后简短作答：",
    )
    suffixes = ("。", "，回答后停止。", "，不要补充背景。", "，不要加说明。", "，只写核心答案。")
    values = []
    for question, answer in facts:
        for prefix in prefixes:
            for suffix in suffixes:
                values.append(
                    row("factual", f"{prefix}{question}{suffix}", answer, "v47_curated")
                )
    return values


def generate_reasoning_rows() -> dict[str, list[dict[str, str]]]:
    groups = {key: [] for key in ("add", "sub", "mul", "div", "compare", "word")}
    add_templates = (
        "计算{a}+{b}，只写结果。", "{a}加{b}等于多少？直接写数字。",
        "请完成加法{a}+{b}，不要过程。", "求{a}与{b}的和，只返回数字。",
    )
    sub_templates = (
        "计算{a}-{b}，只写结果。", "{a}减去{b}是多少？直接写数字。",
        "完成减法{a}-{b}，不要解释。", "求{a}与{b}的差，只返回数字。",
    )
    for a in range(11, 100):
        for b in range(2, 50):
            if (a, b) != (17, 28):
                for template in add_templates:
                    groups["add"].append(
                        row("reasoning_add", template.format(a=a, b=b), str(a + b), "v47_generated")
                    )
            if a >= b:
                for template in sub_templates:
                    groups["sub"].append(
                        row("reasoning_sub", template.format(a=a, b=b), str(a - b), "v47_generated")
                    )

    for a in range(2, 21):
        for b in range(2, 13):
            groups["mul"].append(
                row("reasoning_mul", f"计算{a}乘{b}，只写数字。", str(a * b), "v47_generated")
            )
            groups["mul"].append(
                row("reasoning_mul", f"{a}×{b}等于多少？不要写过程。", str(a * b), "v47_generated")
            )
            groups["div"].append(
                row("reasoning_div", f"{a * b}除以{b}等于多少？只写数字。", str(a), "v47_generated")
            )
            groups["div"].append(
                row("reasoning_div", f"完成整除：{a * b}÷{a}。直接给答案。", str(b), "v47_generated")
            )

    decimals = [round(value / 100, 2) for value in range(11, 96, 4)]
    for left in decimals:
        for right in decimals:
            if left == right:
                continue
            answer = str(max(left, right)).rstrip("0").rstrip(".")
            groups["compare"].append(
                row(
                    "reasoning_compare",
                    f"比较{left}和{right}，只写较大的数。",
                    answer,
                    "v47_generated",
                )
            )

    for total in range(8, 60):
        for removed in range(1, min(total, 15)):
            groups["word"].append(
                row(
                    "reasoning_word",
                    f"盒子里有{total}个球，拿走{removed}个，还剩多少个？只写“数字+个”。",
                    f"{total - removed}个",
                    "v47_generated",
                )
            )
    return groups


def generate_anti_echo_rows() -> dict[str, list[dict[str, str]]]:
    notices = []
    dates = ("7月3日", "7月8日", "7月12日", "8月5日", "8月16日", "9月2日", "9月18日")
    times = ("上午9:00", "上午10:30", "下午14:00", "下午15:30", "晚上19:00")
    rooms = ("第二会议室", "三楼会议室", "培训室", "线上会议室", "多功能厅")
    topics = ("项目进度", "月度复盘", "安全培训", "产品评审", "工作计划")
    for date in dates:
        for time in times:
            for room in rooms:
                for topic in topics:
                    instruction = f"写一则100字以内的会议通知：{date}{time}，地点为{room}，主题是{topic}。"
                    answer = f"会议通知：请相关人员于{date}{time}在{room}参加{topic}会议，请提前准备并准时出席。"
                    notices.append(row("anti_echo_notice", instruction, answer, "v47_generated"))

    emails = []
    recipients = ("张老师", "李经理", "王主管", "陈老师", "赵经理")
    reasons = ("感冒发烧", "身体不适", "需要就医", "家中有急事", "参加学校活动")
    days = ("一天", "半天", "明天一天", "周五一天")
    email_prefixes = ("请直接写正文：", "不要复述要求：", "请给出可直接发送的邮件：")
    for recipient in recipients:
        for reason in reasons:
            for day in days:
                for prefix in email_prefixes:
                    instruction = f"{prefix}写一封简短请假邮件，收件人是{recipient}，因为{reason}请假{day}。"
                    answer = f"{recipient}，您好！我因{reason}，申请请假{day}。相关事务会提前安排，恳请批准。谢谢！"
                    emails.append(row("anti_echo_email", instruction, answer, "v47_generated"))

    summaries = []
    summary_cases = (
        ("开发工作已经结束，当前正在进行上线前检查", "开发工作已完成，当前正在进行上线前检查。"),
        ("培训原定周一举行，现在改到周三下午", "培训由周一调整至周三下午举行。"),
        ("本周完成需求确认，下周开始编码实现", "本周完成需求确认，下周开始编码。"),
        ("设备维护已经完成，系统恢复正常运行", "设备维护已完成，系统已恢复正常运行。"),
        ("报名人数超过计划，需要增加一个培训班", "报名人数超出计划，需要增设培训班。"),
        ("预算已经批准，但采购清单仍需修改", "预算已批准，采购清单仍需修改。"),
        ("测试发现三个问题，其中两个已经修复", "测试发现三个问题，目前已修复两个。"),
        ("会议讨论了进度风险，并确定了负责人", "会议分析了进度风险并确定了负责人。"),
        ("材料已经提交，目前等待审核结果", "材料已提交，正在等待审核结果。"),
        ("新版本功能完成，但文档还没有更新", "新版本功能已完成，文档仍需更新。"),
    )
    summary_prefixes = (
        "请概括为一句话：", "用一句正式中文总结：", "不要复述要求，直接总结：", "请写出不超过50字的摘要：",
        "压缩成一句话：", "只给出总结结果：", "保持原意并简化：", "请删除冗余表达：",
        "一句话概括以下信息：", "请直接输出摘要：", "将内容改写为一句简洁表述：", "请给出可直接使用的总结：",
        "不需要解释，概括：", "用最短的完整句总结：", "请写一句结论：", "把下面内容压缩：",
        "摘要任务：", "请生成一句话摘要：", "请简明概括：", "只写摘要正文：",
        "请压缩为正式摘要：", "只输出一句总结：", "请直接概括重点：", "精简下面的内容：", "生成一句可用摘要：",
    )
    for source, answer in summary_cases:
        for prefix in summary_prefixes:
            summaries.append(row("anti_echo_summary", f"{prefix}{source}。", answer, "v47_curated"))

    rewrites = []
    rewrite_cases = (
        ("这个事情大家还要再商量商量", "此事仍需进一步讨论。"),
        ("我们得赶紧把报告交上去", "请尽快完成并提交报告。"),
        ("这个方案还可以但是有地方要改", "该方案总体可行，但部分细节仍需优化。"),
        ("明天开会大家别迟到", "请各位准时参加明天的会议。"),
        ("这个问题现在还说不准", "该问题目前尚无明确结论。"),
        ("材料有点多我还没看完", "材料较多，我尚未完成审阅。"),
        ("客户那边还没有给我们回复", "客户目前尚未回复。"),
        ("这个功能我们下次再做", "该功能计划在后续版本中实现。"),
        ("今天的任务差不多都做完了", "今日任务已基本完成。"),
        ("数据看起来好像有一些问题", "数据可能存在异常，需要进一步核查。"),
    )
    rewrite_prefixes = (
        "改写得正式简洁：", "请直接给出正式表达：", "不要解释，只改写：", "将口语改为书面语：",
        "请润色并保持简短：", "正式改写：", "请删除口语化表达：", "只输出修改后的句子：",
        "请给出可直接使用的版本：", "改成正式中文：", "请简洁改写：", "不改变原意，正式表达：",
        "请压缩并润色：", "将下面的话规范化：", "只写最终结果：", "请优化表达：",
        "改为工作场景用语：", "请写出正式版本：", "去掉冗余并改写：", "请完成文字润色：",
        "用正式语气重写：", "请只返回改写结果：", "将表达改得简洁准确：", "直接给出规范版本：", "请改成书面表达：",
    )
    for source, answer in rewrite_cases:
        for prefix in rewrite_prefixes:
            rewrites.append(row("anti_echo_rewrite", f"{prefix}{source}。", answer, "v47_curated"))

    return {"notice": notices, "email": emails, "summary": summaries, "rewrite": rewrites}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V4.7 targeted-fix response-only SFT mix.")
    parser.add_argument("--input", nargs="+", required=True, help="V4.6 JSON/JSONL replay source.")
    parser.add_argument("--out", default="data/raw/V4.7_targeted_fix_sft_mix.jsonl")
    parser.add_argument("--benchmark", default="eval/zh_generation_v3.jsonl")
    parser.add_argument("--max_records", type=int, default=12000)
    parser.add_argument("--strict_records", type=int, default=2500)
    parser.add_argument("--safety_records", type=int, default=2000)
    parser.add_argument("--factual_records", type=int, default=1500)
    parser.add_argument("--reasoning_records", type=int, default=1500)
    parser.add_argument("--anti_echo_records", type=int, default=1000)
    parser.add_argument("--identity_records", type=int, default=300)
    parser.add_argument("--knowledge_records", type=int, default=700)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    benchmark_prompts = load_benchmark_prompts(Path(args.benchmark))
    global_seen: set[str] = set()
    selected: list[dict[str, str]] = []
    selected_counts = {}
    benchmark_rejected = 0

    requested_targeted = (
        args.strict_records
        + args.safety_records
        + args.factual_records
        + args.reasoning_records
        + args.anti_echo_records
        + args.identity_records
        + args.knowledge_records
    )
    if requested_targeted >= args.max_records:
        raise ValueError("Targeted quotas must leave room for broad replay.")

    strict = generate_strict_rows()
    strict_targets = {
        "strict_exact": int(args.strict_records * 0.30),
        "strict_list": int(args.strict_records * 0.25),
        "strict_json": int(args.strict_records * 0.20),
    }
    strict_targets["strict_bounded"] = args.strict_records - sum(strict_targets.values())

    safety_refusal, truth_correction = generate_safety_rows()
    safety_targets = {
        "safety_refusal": int(args.safety_records * 0.75),
        "truth_correction": args.safety_records - int(args.safety_records * 0.75),
    }

    reasoning = generate_reasoning_rows()
    reasoning_targets = {
        "add": int(args.reasoning_records * 0.30),
        "sub": int(args.reasoning_records * 0.20),
        "mul": int(args.reasoning_records * 0.17),
        "div": int(args.reasoning_records * 0.13),
        "compare": int(args.reasoning_records * 0.10),
    }
    reasoning_targets["word"] = args.reasoning_records - sum(reasoning_targets.values())

    anti_echo = generate_anti_echo_rows()
    anti_echo_targets = {
        "notice": int(args.anti_echo_records * 0.40),
        "email": int(args.anti_echo_records * 0.20),
        "summary": int(args.anti_echo_records * 0.20),
    }
    anti_echo_targets["rewrite"] = args.anti_echo_records - sum(anti_echo_targets.values())

    groups = []
    groups.extend((label, strict[label], count) for label, count in strict_targets.items())
    groups.extend(
        (
            ("safety_refusal", safety_refusal, safety_targets["safety_refusal"]),
            ("truth_correction", truth_correction, safety_targets["truth_correction"]),
            ("factual", generate_factual_rows(), args.factual_records),
        )
    )
    groups.extend((f"reasoning_{label}", reasoning[label], count) for label, count in reasoning_targets.items())
    groups.extend((f"anti_echo_{label}", anti_echo[label], count) for label, count in anti_echo_targets.items())
    groups.extend(
        (
            ("identity", generate_identity_candidates(), args.identity_records),
            ("knowledge", generate_knowledge_candidates(), args.knowledge_records),
        )
    )

    for label, candidates, count in groups:
        chosen, rejected = select_rows(
            candidates,
            count,
            rng,
            benchmark_prompts,
            global_seen,
            label,
        )
        selected.extend(chosen)
        selected_counts[label] = len(chosen)
        benchmark_rejected += rejected

    replay_target = args.max_records - len(selected)
    replay_candidates = []
    replay_seen = set()
    replay_stats = Counter()
    excluded_categories = {
        "exact_reasoning",
        "exact_factual",
        "exact_format",
        "safety_truth",
        "safety_refusal",
        "truth_correction",
    }

    for record in iter_records(args.input):
        replay_stats["input"] += 1
        if isinstance(record, dict) and str(record.get("category", "")) in excluded_categories:
            replay_stats["excluded_targeted"] += 1
            continue
        normalized = normalize_record(record)
        if normalized is None:
            replay_stats["invalid"] += 1
            continue
        if not replay_is_good(normalized):
            replay_stats["filtered"] += 1
            continue
        if too_similar_to_benchmark(normalized["instruction"], benchmark_prompts):
            replay_stats["benchmark_rejected"] += 1
            continue
        key = fingerprint(normalized)
        if key in replay_seen or key in global_seen:
            replay_stats["duplicate"] += 1
            continue
        replay_seen.add(key)
        replay_candidates.append(
            (
                replay_score(normalized),
                rng.random(),
                row("broad_replay", normalized["instruction"], normalized["output"], "v46_replay"),
            )
        )

    replay_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(replay_candidates) < replay_target:
        raise ValueError(
            f"Need {replay_target} broad replay rows, but only {len(replay_candidates)} passed filters"
        )
    replay_rows = [item[2] for item in replay_candidates[:replay_target]]
    selected.extend(replay_rows)
    selected_counts["broad_replay"] = len(replay_rows)

    rng.shuffle(selected)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for value in selected:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    category_counts = Counter(value["category"] for value in selected)
    answer_lengths = [len(value["output"]) for value in selected]
    summary = {
        "output": str(out_path),
        "records": len(selected),
        "requested_records": args.max_records,
        "selected_counts": selected_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "answer_chars": {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "mean": round(sum(answer_lengths) / len(answer_lengths), 2),
        },
        "benchmark": args.benchmark,
        "benchmark_prompts": len(benchmark_prompts),
        "generated_benchmark_rejected": benchmark_rejected,
        "replay_stats": dict(replay_stats),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
