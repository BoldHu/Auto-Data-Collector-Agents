"""Prompts for exam question extraction and quality verification.

Uses string concatenation to avoid f-string issues with JSON curly braces.
"""

# ── Extraction System Prompt ──────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """你是一位专业的碳纤维和复合材料领域考试题目提取专家。你的任务是从考试文档中提取所有题目，包括选择题、填空题、简答题、计算题等。

关键规则：
1. 必须保留题目的原始文本，不得修改或简化
2. 必须正确提取所有选项（A、B、C、D等）
3. 必须正确匹配答案和解析
4. 如果答案来自文档中的答案区域，标记 answer_source = "explicit_answer_key"
5. 如果答案是内嵌在题目中的，标记 answer_source = "inline_solution"
6. 如果需要推断答案，标记 answer_source = "model_inferred"，并设置 extraction_confidence <= 0.7
7. 如果没有答案，标记 answer_source = "missing"，不得编造答案
8. 只提取碳纤维、复合材料、纤维纺丝、碳化、预浸料、力学性能、材料测试等相关领域的题目
9. 无关题目（如英语、政治等）直接跳过
10. 保留公式、单位、化学式等技术细节"""

# ── Extraction User Prompt Template ──────────────────────────────────────

EXTRACTION_USER_PROMPT = """请从以下考试文档文本中提取所有题目。

文档来源：{source_file}
文本内容：
{text_content}

请严格按照以下JSON格式输出：
{output_schema}

注意：
- question_type 必须是以下之一：single_choice, multiple_choice, true_false, fill_blank, short_answer, calculation, case_analysis, unknown
- difficulty 必须是：easy, medium, hard
- answer_source 必须是：explicit_answer_key, inline_solution, model_inferred, missing
- options 只在选择题中出现，格式为 [{{"key": "A", "text": "..."}}]
- knowledge_points 是字符串数组
- extraction_confidence 范围 0.0-1.0
- domain_relevance 范围 0.0-1.0，评估题目与碳纤维领域的相关性
- 如果文本中没有碳纤维相关题目，返回空数组 []

请直接输出JSON数组，不要添加额外说明。"""

# ── Output Schema ─────────────────────────────────────────────────────────

EXTRACTION_OUTPUT_SCHEMA = """[
  {
    "question_number": "1",
    "question_type": "single_choice",
    "question_text": "完整的题目文本",
    "options": [
      {"key": "A", "text": "选项A文本"},
      {"key": "B", "text": "选项B文本"},
      {"key": "C", "text": "选项C文本"},
      {"key": "D", "text": "选项D文本"}
    ],
    "answer": "A",
    "answer_source": "explicit_answer_key",
    "explanation": "解析文本（如果有）",
    "knowledge_points": ["碳纤维", "力学性能"],
    "difficulty": "medium",
    "requires_calculation": false,
    "contains_formula": false,
    "contains_table": false,
    "contains_image_reference": false,
    "domain_relevance": 0.9,
    "extraction_confidence": 0.95,
    "uncertainty_notes": []
  }
]"""

# ── Quality Verification System Prompt ────────────────────────────────────

QUALITY_SYSTEM_PROMPT = """你是一位独立的考试题目质量验证专家。你的任务是评估提取的考试题目的质量。

评估维度：
1. clarity（清晰度）：题目表述是否清晰无歧义
2. completeness（完整性）：题目、选项、答案是否完整
3. answerability（可答性）：题目是否可以回答
4. option_integrity（选项完整性）：选择题选项是否完整
5. answer_consistency（答案一致性）：答案是否与题目/选项匹配
6. domain_relevance（领域相关性）：是否与碳纤维领域相关
7. difficulty_reasonableness（难度合理性）：难度评估是否合理
8. benchmark_usefulness（基准有用性）：是否适合作为基准题目

检测问题：
- 缺少选项
- 重复题目
- 答案格式错误
- 答案与选项不匹配
- 编造的答案
- OCR损坏的题目
- 无关题目
- 重复的样板文本
- 不完整的计算题
- 模糊或多答案问题"""

# ── Quality Verification User Prompt ──────────────────────────────────────

QUALITY_USER_PROMPT = """请验证以下考试题目的质量。

题目信息：
{question_json}

原始文本证据：
{raw_evidence}

请严格按照以下JSON格式输出：
{quality_schema}

注意：
- quality_status 必须是：keep, review, drop
- 所有分数范围 0.0-1.0
- detected_issues 是字符串数组
- 如果发现问题，在 revision_suggestion 中给出修改建议"""

# ── Quality Output Schema ─────────────────────────────────────────────────

QUALITY_OUTPUT_SCHEMA = """{
  "quality_status": "keep",
  "clarity": 0.9,
  "completeness": 0.95,
  "answerability": 0.9,
  "option_integrity": 0.95,
  "answer_consistency": 0.9,
  "domain_relevance": 0.85,
  "difficulty_reasonableness": 0.8,
  "benchmark_usefulness": 0.85,
  "detected_issues": [],
  "revision_suggestion": ""
}"""

# ── Builder Functions ─────────────────────────────────────────────────────

def get_extraction_prompt(source_file: str, text_content: str) -> str:
    """Build extraction user prompt."""
    return EXTRACTION_USER_PROMPT.replace("{source_file}", source_file).replace(
        "{text_content}", text_content
    ).replace("{output_schema}", EXTRACTION_OUTPUT_SCHEMA)


def get_quality_prompt(question_json: str, raw_evidence: str) -> str:
    """Build quality verification user prompt."""
    return QUALITY_USER_PROMPT.replace("{question_json}", question_json).replace(
        "{raw_evidence}", raw_evidence
    ).replace("{quality_schema}", QUALITY_OUTPUT_SCHEMA)


# ── Domain Filter Keywords ────────────────────────────────────────────────

CARBON_FIBER_KEYWORDS = [
    "碳纤维", "碳素纤维", "碳丝", "碳化", "石墨化", "PAN", "聚丙烯腈",
    "预浸料", "prepreg", "复合材料", "composite", "基体", "增强",
    "纤维", "fiber", "fibre", "纺丝", "spinning", "原丝",
    "力学性能", "拉伸", "压缩", "弯曲", "剪切", "模量", "强度",
    "固化", "cure", "树脂", "resin", "环氧", "epoxy",
    "T300", "T700", "T800", "M40", "M55", "M60",
    "碳纤维布", "碳纤维板", "碳纤维管", "碳纤维加固",
    "复合材料制备", "成型工艺", "RTM", "真空袋", "热压罐",
    "层间剪切", "界面", "interfacial", "表面处理",
    "碳纤维生产", "碳纤维制造", "氧化炉", "碳化炉",
]


def is_domain_relevant(text: str) -> bool:
    """Check if text is relevant to carbon fiber domain."""
    text_lower = text.lower()
    for keyword in CARBON_FIBER_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False
