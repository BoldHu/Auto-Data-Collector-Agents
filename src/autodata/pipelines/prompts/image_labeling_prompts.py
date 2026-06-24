"""Image labeling and captioning prompt templates v1.0.

Provides prompts for:
1. Image labeling — classify image category, material form, process stage, etc.
2. Image captioning — generate short and technical captions
3. Quality assessment — evaluate image quality for corpus/benchmark inclusion
4. Benchmark candidate generation — create multimodal benchmark items from images

All prompts enforce strict JSON output with provenance preservation.
"""

from __future__ import annotations

PROMPT_VERSION = "v1.0"

# ── Category definitions for domain labeling ──────────────────────────

CATEGORY_DEFINITIONS = """碳纤维/复合材料领域图像类别定义：
- fiber: 纤维原料（碳纤维丝、原丝、纱线等）
- fabric: 织物（碳纤维布、编织物、单向布等）
- prepreg: 预浸料（预浸布、预浸带等）
- composite_part: 复合材料制件（层压板、CFRP部件、成型件等）
- microstructure: 微观结构（SEM图像、断面、纤维截面等）
- equipment: 设备（热压罐、缠绕机、拉挤机、试验机等）
- process: 工艺过程（固化、缠绕、铺层、拉挤等现场照片）
- application: 应用场景（航空航天、汽车、体育用品等）
- testing: 测试/检测（力学测试、无损检测、CT扫描等）
- defect: 缺陷/损伤（分层、脱粘、孔隙、冲击损伤等）
- chart_diagram: 图表/示意图（工艺流程图、性能曲线、结构图等）
- paper_screenshot: 论文截图/文献图片
- irrelevant: 与碳纤维/复合材料无关的图片"""

MODALITY_DEFINITIONS = """图像模态定义：
- photo: 真实照片（实物拍摄）
- microscopy: 显微镜/SEM/TEM图像
- diagram: 示意图/流程图/结构图
- chart: 数据图表/曲线图/柱状图
- mixed: 混合类型（照片+文字说明等）
- unknown: 无法确定"""

MATERIAL_FORM_DEFINITIONS = """材料形态定义：
- raw_fiber: 纤维原料
- tow: 纤维束/纱线
- fabric: 织物/布
- prepreg: 预浸料
- laminate: 层压板
- cfrp_part: CFRP制件
- powder: 粉末
- unknown: 无法确定"""

PROCESS_STAGE_DEFINITIONS = """工艺阶段定义：
- precursor: 原丝制备
- spinning: 纺丝
- stabilization: 预氧化/稳定化
- carbonization: 碳化
- graphitization: 石墨化
- surface_treatment: 表面处理
- sizing: 上胶
- weaving: 编织
- layup: 预浸料铺层
- curing: 固化成型
- testing: 测试检测
- application: 应用
- unknown: 无法确定"""

APPLICATION_DOMAIN_DEFINITIONS = """应用领域定义：
- aerospace: 航空航天
- automotive: 汽车
- sports: 体育用品
- civil_engineering: 土木工程
- energy: 能源（风电、储氢等）
- industrial: 工业通用
- biomedical: 生物医学
- unknown: 无法确定"""


# ── Image labeling prompt ─────────────────────────────────────────────

LABELING_SYSTEM_PROMPT = """你是碳纤维/复合材料领域图像分类专家。
你的任务是根据图像内容，对其进行多维度分类标注。

关键规则：
1. 仅根据图像中可见内容进行判断，不做推测
2. 如果图像内容模糊或无法判断，标记为unknown并说明原因
3. 如果图像与碳纤维/复合材料完全无关，标记为irrelevant
4. 所有判断必须基于视觉证据，不要根据图像文件名或文件夹名推断"""

LABELING_USER_PROMPT_TEMPLATE = """请对以下碳纤维领域图像进行分类标注。

{category_defs}
{modality_defs}
{material_form_defs}
{process_stage_defs}
{application_domain_defs}

请严格按照以下JSON格式输出：
{
  "primary_category": "主类别（从上述类别列表中选择一个）",
  "secondary_categories": ["次要类别列表（最多3个，从上述类别列表中选择）"],
  "modality": "图像模态",
  "material_form": "材料形态",
  "process_stage": "工艺阶段",
  "application_domain": "应用领域",
  "domain_relevance": 0.0到1.0之间的浮点数，表示与碳纤维领域的相关程度，
  "label_confidence": 0.0到1.0之间的浮点数，表示标注置信度,
  "requires_human_review": true/false，置信度低于0.5时设为true,
  "visual_evidence": ["列出支持每个分类判断的视觉证据，例如：可见碳纤维织物纹理、可见热压罐设备等"],
  "uncertainty_notes": "对不确定或模糊判断的说明，空字符串表示完全确定"
}

请仔细观察图像并输出JSON。"""


# ── Image captioning prompt ───────────────────────────────────────────

CAPTIONING_SYSTEM_PROMPT = """你是碳纤维/复合材料领域图像描述专家。
你的任务是生成准确、技术性的图像描述。

关键规则：
1. 短标题（short_caption）必须简洁，不超过30字，用中文
2. 技术描述（technical_caption）可以详细，用中文，最多200字
3. 描述必须100%基于图像中可见内容，严禁推测或添加不可见信息
4. 如果图像中有文字，在visible_text中列出
5. 对推测性内容必须在uncertainty_notes中说明"""

CAPTIONING_USER_PROMPT_TEMPLATE = """请为以下碳纤维领域图像生成描述。

请严格按照以下JSON格式输出：
{
  "short_caption": "简短描述（不超过30字，中文）",
  "technical_caption": "详细技术描述（最多200字，中文，描述可见内容、材料、工艺、设备等）",
  "visible_objects": ["可见物体列表，如：碳纤维布、热压罐、拉伸试验机等"],
  "visible_materials": ["可见材料列表，如：碳纤维、环氧树脂、预浸料等"],
  "visible_processes": ["可见工艺列表，如：铺层、固化、缠绕等"],
  "visible_equipment": ["可见设备列表，如：热压罐、缠绕机、SEM等"],
  "visible_text": ["图像中可见的文字列表，如：标尺刻度、品牌名、参数标注等"],
  "visual_evidence": ["支持描述的视觉证据列表"],
  "inferred_domain_context": ["基于视觉内容推断的领域上下文（必须注明'推断'）"],
  "uncertainty_notes": "对不确定描述的说明，空字符串表示完全确定"
}

请仔细观察图像并输出JSON。"""


# ── Image quality assessment prompt ──────────────────────────────────

QUALITY_SYSTEM_PROMPT = """你是碳纤维领域图像质量评估专家。
你的任务是评估图像是否适合用于语料库和基准评测数据集。

关键规则：
1. clarity: 图像清晰度（0-1），模糊/低分辨率图像得分低
2. domain_relevance: 领域相关性（0-1），与碳纤维/复合材料无关的图像得分低
3. visual_informativeness: 视觉信息量（0-1），信息丰富的图像得分高
4. captionability: 可描述性（0-1），能否生成准确描述
5. reasoning_potential: 推理潜力（0-1），是否适合作为基准评测题目
6. 对得分低于0.4的维度必须在uncertainty_notes中说明原因"""

QUALITY_USER_PROMPT_TEMPLATE = """请评估以下图像的质量和适用性。

请严格按照以下JSON格式输出：
{
  "clarity": 0.0到1.0浮点数，
  "domain_relevance": 0.0到1.0浮点数，
  "visual_informativeness": 0.0到1.0浮点数，
  "captionability": 0.0到1.0浮点数，
  "reasoning_potential": 0.0到1.0浮点数，
  "metadata_completeness": 0.0到1.0浮点数（有元数据=0.7，有URL=0.8，有关键词标签=0.9），
  "quality_status": "keep/review/drop",
  "drop_reason": "如果是drop，说明原因；空字符串表示保留",
  "uncertainty_notes": "对不确定评估的说明"
}

请仔细观察图像并输出JSON。"""


# ── Multimodal benchmark candidate generation prompt ──────────────────

BENCHMARK_SYSTEM_PROMPT = """你是碳纤维/复合材料领域基准评测题目生成专家。
你的任务是根据图像内容，生成多模态基准评测候选题目。

关键规则：
1. 题目必须基于图像中可见内容，严禁推测不可见信息
2. 每个题目必须有明确答案，答案必须可以从图像内容推导
3. 解题过程必须涉及视觉推理（不能仅靠文字知识）
4. 防止幻觉：答案和解释必须与图像内容严格一致
5. 避免过于简单的题目（如"这是什么？"），要考察理解/推理能力"""

BENCHMARK_USER_PROMPT_HEADER = """请根据以下碳纤维领域图像生成1-3个基准评测候选题目。

可选任务类型：
- visual_qa: 视觉问答
- multiple_choice: 多项选择题
- short_answer: 简答题
- process_reasoning: 工艺推理题
- defect_diagnosis: 缺陷诊断题
- chart_reading: 图表阅读题
- diagram_reasoning: 示意图推理题
- ocr_reasoning: OCR推理题
- cross_modal_reasoning: 跨模态推理题"""

BENCHMARK_OUTPUT_SCHEMA = """
请严格按照以下JSON格式输出题目列表：
{
  "candidates": [
    {
      "task_type": "任务类型",
      "question": "题目（中文）",
      "options": ["选项列表（仅multiple_choice类型需要，4个选项）"],
      "answer": "答案",
      "explanation": "解题解释（必须引用图像中可见内容作为证据）",
      "visual_evidence": ["支持答案的视觉证据列表"],
      "required_knowledge": ["解题所需的知识点列表"],
      "reasoning_steps": ["解题推理步骤列表"],
      "difficulty": "easy/medium/hard",
      "answerability": "image_only或image_plus_metadata或image_plus_domain_knowledge或not_answerable",
      "hallucination_risk": "low/medium/high"
    }
  ]
}

请仔细观察图像，确保题目基于可见内容生成。"""


def build_benchmark_user_prompt(primary_category, material_form, process_stage, domain_relevance):
    """Build benchmark user prompt with variable info, avoiding format() with curly braces."""
    info_section = (
        "图像标注信息（仅供参考，题目必须基于图像可见内容）：\n"
        f"- 主类别: {primary_category}\n"
        f"- 材料形态: {material_form}\n"
        f"- 工艺阶段: {process_stage}\n"
        f"- 领域相关度: {domain_relevance}\n"
    )
    return BENCHMARK_USER_PROMPT_HEADER + "\n\n" + info_section + BENCHMARK_OUTPUT_SCHEMA


def build_critic_user_prompt(task_type, question, options, answer, explanation,
                             visual_evidence, primary_category, label_confidence):
    """Build critic validation prompt with variable info, avoiding format() with curly braces."""
    info_section = (
        "请验证以下碳纤维领域基准评测候选题目。\n\n"
        "题目信息：\n"
        f"- 任务类型: {task_type}\n"
        f"- 题目: {question}\n"
        f"- 选项: {options}\n"
        f"- 答案: {answer}\n"
        f"- 解释: {explanation}\n"
        f"- 视觉证据: {visual_evidence}\n\n"
        "图像标注信息：\n"
        f"- 主类别: {primary_category}\n"
        f"- 置信度: {label_confidence}\n"
    )
    return info_section + CRITIC_OUTPUT_SCHEMA


# ── Critic validation prompt ──────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """你是碳纤维/复合材料领域独立质量审核专家。
你的任务是验证图像标注、描述和基准评测候选题目的质量。

关键验证维度：
1. answerability_score: 题目是否可以从图像内容回答（0-1）
2. visual_grounding_score: 答案是否基于图像可见内容（0-1）
3. domain_reasoning_score: 题目是否考察领域推理能力（0-1）
4. hallucination_risk: 答案/解释是否包含不可见内容的推测
5. ambiguity_score: 题目是否存在歧义（0-1，越高越有歧义）"""

CRITIC_OUTPUT_SCHEMA = """
请严格按照以下JSON格式输出验证结果：
{
  "validation_status": "passed或failed",
  "answerability_score": 0.0到1.0浮点数,
  "visual_grounding_score": 0.0到1.0浮点数,
  "domain_reasoning_score": 0.0到1.0浮点数,
  "hallucination_risk": "low或medium或high",
  "ambiguity_score": 0.0到1.0浮点数,
  "critic_notes": "审核意见",
  "revision_suggestion": "改进建议（如题目需修改，给出具体修改方案）"
}

请根据图像内容和题目信息进行独立验证。"""


# ── Combined output schema (not an f-string, safe curly braces) ───────

COMBINED_OUTPUT_SCHEMA = """{
  "primary_category": "主类别",
  "secondary_categories": ["次要类别列表"],
  "modality": "图像模态",
  "material_form": "材料形态",
  "process_stage": "工艺阶段",
  "application_domain": "应用领域",
  "domain_relevance": 0.0到1.0,
  "label_confidence": 0.0到1.0,
  "requires_human_review": true或false,
  "short_caption": "简短描述（不超过30字）",
  "technical_caption": "详细技术描述（最多200字）",
  "visible_objects": ["可见物体列表"],
  "visible_materials": ["可见材料列表"],
  "visible_processes": ["可见工艺列表"],
  "visible_equipment": ["可见设备列表"],
  "visible_text": ["可见文字列表"],
  "visual_evidence": ["视觉证据列表"],
  "inferred_domain_context": ["推断的领域上下文（标注推断）"],
  "clarity": 0.0到1.0,
  "visual_informativeness": 0.0到1.0,
  "captionability": 0.0到1.0,
  "reasoning_potential": 0.0到1.0,
  "quality_status": "keep或review或drop",
  "drop_reason": "",
  "uncertainty_notes": ""
}

请仔细观察图像并输出完整JSON。"""


# ── Prompt builder functions ──────────────────────────────────────────

def get_labeling_prompt(image_path: str) -> tuple[dict, dict]:
    """Build labeling prompt messages for a single image.

    Returns (system_message, user_message) tuple.
    The user message includes the image as base64 inline content.
    """
    from src.autodata.utils.image_utils import resize_image_for_api

    b64_url = resize_image_for_api(image_path)
    user_content = [
        {"type": "text", "text": LABELING_USER_PROMPT_TEMPLATE.format(
            category_defs=CATEGORY_DEFINITIONS,
            modality_defs=MODALITY_DEFINITIONS,
            material_form_defs=MATERIAL_FORM_DEFINITIONS,
            process_stage_defs=PROCESS_STAGE_DEFINITIONS,
            application_domain_defs=APPLICATION_DOMAIN_DEFINITIONS,
        )},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return (
        {"role": "system", "content": LABELING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )


def get_captioning_prompt(image_path: str) -> tuple[dict, dict]:
    """Build captioning prompt messages for a single image."""
    from src.autodata.utils.image_utils import resize_image_for_api

    b64_url = resize_image_for_api(image_path)
    user_content = [
        {"type": "text", "text": CAPTIONING_USER_PROMPT_TEMPLATE},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return (
        {"role": "system", "content": CAPTIONING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )


def get_quality_prompt(image_path: str) -> tuple[dict, dict]:
    """Build quality assessment prompt messages for a single image."""
    from src.autodata.utils.image_utils import resize_image_for_api

    b64_url = resize_image_for_api(image_path)
    user_content = [
        {"type": "text", "text": QUALITY_USER_PROMPT_TEMPLATE},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return (
        {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )


def get_combined_labeling_prompt(image_path: str) -> list[dict]:
    """Build combined labeling + captioning + quality prompt for efficiency.

    This sends all three tasks in one LLM call, reducing API overhead.
    The model produces one comprehensive JSON output covering all dimensions.
    """
    from src.autodata.utils.image_utils import resize_image_for_api

    b64_url = resize_image_for_api(image_path)

    # Use string concatenation instead of f-string to avoid
    # curly braces in JSON template being interpreted as format specifiers
    combined_prompt = (
        "请对以下碳纤维领域图像进行全面标注，包括分类、描述和质量评估。\n\n"
        + CATEGORY_DEFINITIONS + "\n"
        + MODALITY_DEFINITIONS + "\n"
        + MATERIAL_FORM_DEFINITIONS + "\n"
        + PROCESS_STAGE_DEFINITIONS + "\n"
        + APPLICATION_DOMAIN_DEFINITIONS + "\n\n"
        + "请严格按照以下JSON格式输出完整标注：\n"
        + COMBINED_OUTPUT_SCHEMA
    )

    user_content = [
        {"type": "text", "text": combined_prompt},
        {"type": "image_url", "image_url": {"url": b64_url}},
    ]

    return [
        {"role": "system", "content": LABELING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]