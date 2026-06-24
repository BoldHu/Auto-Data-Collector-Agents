# carbon_fiber_keywords.py
# 生成碳纤维领域 3000+ 长尾关键词

from itertools import product

def build_carbon_fiber_keywords():
    # 一些基础“主语”
    bases = [
        "carbon fiber", "carbon fibre", "CFRP",
        "carbon fiber composite", "carbon fiber material",
        "carbon fiber reinforced polymer", "CFRP composite"
    ]

    # 产业链环节
    stages = [
        "precursor", "PAN precursor", "pitch based precursor",
        "fiber spinning", "dry jet wet spinning", "wet spinning",
        "oxidation", "stabilization", "carbonization", "graphitization",
        "surface treatment", "electrolytic surface treatment",
        "sizing process", "coating", "weaving", "3D weaving",
        "braiding", "unidirectional tape", "prepreg", "layup",
        "RTM molding", "VARTM infusion", "autoclave curing",
        "compression molding", "filament winding",
        "machining", "drilling", "trimming", "bonding", "assembly",
        "testing", "fatigue testing", "impact testing",
        "non destructive testing", "NDT inspection", "ultrasonic testing",
        "CT scan", "X ray inspection", "thermography inspection",
        "quality control", "failure analysis", "microscopy", "SEM observation"
    ]

    # 场景 / 地点
    scenes = [
        "factory", "production line", "industrial plant", "workshop",
        "laboratory", "R&D lab", "materials lab", "testing laboratory",
        "clean room", "autoclave room", "winding workshop",
        "weaving workshop", "composite shop", "prototype workshop"
    ]

    # 视角 / 表现形式
    views = [
        "close up", "macro view", "detail view", "texture background",
        "microstructure", "cross section", "fracture surface",
        "specimen preparation", "sample on test machine",
        "operator working", "robotic handling", "automatic machine",
        "equipment front view", "control panel",
        "material storage", "prepreg roll storage",
        "spools on rack", "tow bundle close up",
    ]

    # 应用领域
    applications = [
        "aerospace", "aircraft structure", "spacecraft structure",
        "rocket motor case", "satellite structure",
        "automotive body", "car body panel", "monocoque chassis",
        "motorcycle parts", "bicycle frame", "sports equipment",
        "wind turbine blade", "pressure vessel", "hydrogen tank",
        "medical device", "prosthetic leg", "robotic arm",
    ]

    # 中文核心词，直接作为种子
    chinese_seeds = [
        "碳纤维 预氧化炉 生产线",
        "碳纤维 碳化炉 生产现场",
        "碳纤维 表面处理 生产线",
        "碳纤维 上浆 设备",
        "碳纤维 编织 织机 车间",
        "碳纤维 预浸料 冷库 储存",
        "碳纤维 热压罐 固化 现场",
        "碳纤维 纤维缠绕 压力容器",
        "碳纤维 层板 拉伸 试验",
        "碳纤维 复合材料 冲击 试验",
        "碳纤维 复合材料 超声 无损检测",
        "碳纤维 复合材料 CT 扫描 缺陷",
        "碳纤维 断面 SEM 显微 观察",
        "碳纤维 复合材料 分层 缺陷",
        "碳纤维 汽车 车身 结构",
        "碳纤维 飞机 机翼 结构",
        "碳纤维 氢气瓶 高压 容器",
        "碳纤维 风电 叶片",
        "碳纤维 自行车 车架",
        "碳纤维 义肢 假肢 支撑",
    ]

    # 先收集一批“手工种子”，保证高质量覆盖
    manual_seeds = [
        # 原料与纺丝
        "carbon fiber PAN precursor chips close up",
        "PAN precursor spinning line in factory",
        "carbon fiber precursor wet spinning bath",
        "dry jet wet spinning PAN fiber equipment",
        "carbon fiber tow bundle macro view",
        "carbon fiber filament spool in workshop",
        "carbon fiber yarn on creel rack",
        "carbon fiber fabric twill weave texture",
        "unidirectional carbon fiber tape close up",
        "carbon fiber prepreg roll in cold storage",

        # 氧化 / 碳化
        "carbon fiber oxidation furnace production line",
        "carbon fiber stabilization oven with tow running",
        "carbon fiber carbonization furnace high temperature",
        "graphitization furnace carbon fiber high modulus",
        "operator monitoring carbonization furnace control panel",

        # 表面处理 / 上浆
        "carbon fiber surface treatment line electrolytic bath",
        "carbon fiber sizing process rollers and bath",
        "carbon fiber washing and drying line after sizing",

        # 编织 / 预成型
        "carbon fiber weaving loom in operation",
        "3D weaving carbon fiber preform",
        "braided carbon fiber sleeve on mandrel",
        "automatic fiber placement carbon fiber tape laying",
        "robot laying carbon fiber plies on tool",

        # 成型工艺
        "carbon fiber RTM molding setup with dry preform",
        "vacuum assisted resin infusion VARTM carbon fiber",
        "autoclave carbon fiber curing aerospace part",
        "carbon fiber compression molding press machine",
        "filament winding carbon fiber pressure vessel",

        # 结构和产品
        "CFRP laminate plate edges close up",
        "CFRP sandwich panel with honeycomb core",
        "carbon fiber tube profile and rods",
        "carbon fiber automotive body panel production",
        "CFRP aircraft wing spar structure",
        "CFRP fuselage barrel section",

        # 测试与无损检测
        "CFRP tensile testing machine with specimen",
        "CFRP bending test setup in laboratory",
        "CFRP fatigue testing rig with coupons",
        "drop weight impact test on carbon fiber composite",
        "ultrasonic NDT scanning carbon fiber laminate",
        "X ray inspection of CFRP aerospace structure",
        "CT scan image of carbon fiber composite defect",
        "infrared thermography inspection of CFRP panel",

        # 显微与缺陷
        "SEM micrograph carbon fiber fracture surface",
        "SEM micrograph fiber pullout in CFRP",
        "CFRP cross section microstructure with voids",
        "CFRP delamination defect micrograph",
        "CFRP fiber waviness defect micrograph",
        "CFRP matrix cracking microstructure",

        # 应用
        "carbon fiber bicycle frame in workshop",
        "carbon fiber motorcycle parts on display",
        "carbon fiber automotive monocoque chassis",
        "carbon fiber wind turbine blade manufacturing",
        "carbon fiber hydrogen storage tank filament winding",
        "carbon fiber prosthetic leg in laboratory",
    ]

    # 组合生成长尾关键词
    generated = []

    # 组合1：base + stage + scene
    for b, s, sc in product(bases, stages, scenes):
        phrase = f"{b} {s} {sc}"
        generated.append(phrase)

    # 组合2：base + stage + view
    for b, s, v in product(bases, stages, views):
        phrase = f"{b} {s} {v}"
        generated.append(phrase)

    # 组合3：base + stage + application + view
    for b, s, a, v in product(bases, stages[:12], applications, views[:6]):
        phrase = f"{b} {s} {a} {v}"
        generated.append(phrase)

    # 去重 + 清理
    all_kw = []

    # 先放手工种子和中文种子
    all_kw.extend(manual_seeds)
    all_kw.extend(chinese_seeds)

    # 再放自动生成的
    seen = set(all_kw)
    for k in generated:
        k_norm = " ".join(k.split())  # 压缩多空格
        if k_norm not in seen:
            seen.add(k_norm)
            all_kw.append(k_norm)

    # 控制总量：至少 3000，最多留个几千就够了
    if len(all_kw) > 6000:
        all_kw = all_kw[:6000]

    return all_kw


# 供外部导入
CARBON_FIBER_KEYWORDS = build_carbon_fiber_keywords()

if __name__ == "__main__":
    kws = build_carbon_fiber_keywords()
    print("Total keywords:", len(kws))
    print("Sample 20:")
    for k in kws[:20]:
        print(" -", k)