"""Turning the coded 36-field household record into the Chinese persona paragraph.

The synthetic panel stores bracket codes, not prose. This module is the only
place that knows how each code reads in Chinese, so a change to wording here
changes every condition identically and the within-household comparison stays
valid.

Household fields that are missing or coded ``unknown`` render as 未填写 rather
than being dropped: the plan treats "refused to answer" as a distinct state, and
roughly 19% of the panel has an unknown risk preference.
"""

from __future__ import annotations

UNKNOWN = "未填写"

AGE = {
    "18_24": "18–24 岁", "25_34": "25–34 岁", "35_44": "35–44 岁",
    "45_54": "45–54 岁", "55_64": "55–64 岁", "65_plus": "65 岁以上",
}
GENDER = {"male": "男性", "female": "女性"}
EDUCATION = {
    "primary_or_below": "小学及以下", "junior": "初中", "senior_vocational": "高中或中专",
    "college": "大学", "graduate": "研究生",
}
MARITAL = {
    "married": "已婚", "cohabiting": "同居", "previously_married": "曾婚", "never_married": "未婚",
}
EMPLOYMENT = {
    "employed": "在业", "self_employed": "自雇", "farming": "务农",
    "not_working": "未工作", "retired": "退休",
}
OCCUPATION = {
    "manager": "管理人员", "professional": "专业人员", "clerical": "办事人员",
    "service": "服务业从业者", "agriculture": "农业从业者",
    "production_transport": "生产运输从业者",
}
REGION = {"east": "东部", "central": "中部", "west": "西部", "northeast": "东北"}
URBAN_RURAL = {"urban": "城镇", "rural": "农村"}
HOUSEHOLD_SIZE = {"1": "1", "2": "2", "3": "3", "4": "4", "5_plus": "5 人及以上"}
YES_NO = {"yes": "有", "no": "无"}
HOUSING = {"owned": "自有住房", "rented": "租房", "other": "其他住房"}
INCOME_SOURCE = {
    "wage": "工资", "business": "经营收入", "agriculture": "农业收入",
    "property": "财产性收入", "transfer": "转移性收入", "mixed": "多种来源",
}
INCOME = {
    "b1": "不足 2 万元", "b2": "2 万至不足 5 万元", "b3": "5 万至不足 10 万元",
    "b4": "10 万至不足 20 万元", "b5": "20 万元及以上",
}
ASSET = {
    "b1": "不足 1 万元", "b2": "1 万至不足 5 万元", "b3": "5 万至不足 20 万元",
    "b4": "20 万至不足 50 万元", "b5": "50 万元及以上",
}
DEBT_AMOUNT = {
    "none": "无债务",
    "d1": "0 至 1 万元", "d2": "1 万至 2 万元", "d3": "2 万至 5 万元",
    "d4": "5 万至 10 万元", "d5": "10 万至 20 万元", "d6": "20 万至 30 万元",
    "d7": "30 万至 50 万元", "d8": "50 万至 100 万元", "d9": "100 万至 200 万元",
    "d10": "200 万至 500 万元", "d11": "500 万元以上",
}
DEBT_TYPE = {
    "consumer_credit": "消费信贷", "education": "教育", "medical": "医疗",
    "housing": "住房", "none": "无",
}
CASH_FLOW = {"surplus": "有结余", "balance": "收支相当", "deficit": "收入不够支出"}
RISK = {
    "high": "愿意承担高风险", "slightly_high": "略高风险", "average": "平均风险",
    "slightly_low": "略低风险", "none": "不愿承担任何风险",
}
LITERACY = {
    "both_correct": "利息题和通胀题都答对", "interest_only": "仅利息题答对",
    "inflation_only": "仅通胀题答对", "neither_correct": "两题均未答对",
}
TRUST = {"trust": "通常信任他人", "depends": "视情况而定", "distrust": "通常不信任他人"}
INSURANCE = {
    "commercial_life": "商业人寿保险", "commercial_medical": "商业医疗保险",
    "social_medical": "社会医疗保险", "commercial_pension": "商业养老保险",
}

# Field names in the stored record, grouped by the sentence they feed.
FIELDS = {
    "age": "decision_maker.age_group",
    "gender": "decision_maker.gender",
    "education": "decision_maker.education",
    "marital": "decision_maker.marital_status",
    "employment": "decision_maker.employment_status",
    "occupation": "decision_maker.occupation",
    "region": "household.region",
    "urban_rural": "household.urban_rural",
    "size": "household.household_size",
    "child": "household.has_child_under_18",
    "elder": "household.has_elder_65plus",
    "housing": "household.housing_tenure",
    "income": "income.household_income_bracket",
    "personal_income": "income.personal_income_bracket",
    "income_source": "income.main_income_source",
    "consumption": "consumption.total",
    "asset": "extended.financial_asset_bracket",
    "debt_amount": "extended.debt_amount_bracket",
    "debt_type": "extended.debt_type",
    "cash_flow": "extended.income_expenditure_status",
    "risk": "extended.risk_preference",
    "literacy": "extended.financial_literacy",
    "trust": "extended.general_trust",
    "insurance": "extended.insurance_coverage",
}


def lookup(table: dict[str, str], key: str | None) -> str:
    """Chinese label for a code, or 未填写 when the household did not answer."""
    if key in (None, "", "unknown"):
        return UNKNOWN
    return table.get(key, key)


def _join(table: dict[str, str], value: str | None) -> str:
    if value in (None, "", "unknown"):
        return UNKNOWN
    return "、".join(table.get(part, part) for part in value.split("|"))


def debt_type_label(value: str | None) -> str:
    return _join(DEBT_TYPE, value)


def insurance_label(value: str | None) -> str:
    return _join(INSURANCE, value)


def from_record(record: dict) -> dict:
    """Pull the 36-field profile out of a panel row, keyed by short names."""
    stored = record["profile"]
    return {name: stored[path] for name, path in FIELDS.items()}


def persona(profile: dict) -> str:
    """The four-sentence household description shown to the model."""
    return (
        f"{lookup(AGE, profile['age'])}，{lookup(GENDER, profile['gender'])}，"
        f"{lookup(EDUCATION, profile['education'])}，{lookup(MARITAL, profile['marital'])}，"
        f"{lookup(EMPLOYMENT, profile['employment'])}，{lookup(OCCUPATION, profile['occupation'])}。"
        f"住在{lookup(REGION, profile['region'])}{lookup(URBAN_RURAL, profile['urban_rural'])}，"
        f"{lookup(HOUSEHOLD_SIZE, profile['size'])}人户，"
        f"未成年子女{lookup(YES_NO, profile['child'])}，"
        f"65 岁以上老人{lookup(YES_NO, profile['elder'])}，"
        f"{lookup(HOUSING, profile['housing'])}。"
        f"家庭年收入{lookup(INCOME, profile['income'])}，"
        f"个人年收入{lookup(INCOME, profile['personal_income'])}，"
        f"主要来自{lookup(INCOME_SOURCE, profile['income_source'])}；"
        f"年消费{lookup(INCOME, profile['consumption'])}。"
        f"家庭金融资产{lookup(ASSET, profile['asset'])}，"
        f"负债{lookup(DEBT_AMOUNT, profile['debt_amount'])}，"
        f"债务类型{debt_type_label(profile['debt_type'])}，"
        f"{lookup(CASH_FLOW, profile['cash_flow'])}。"
        f"{lookup(RISK, profile['risk'])}；{lookup(LITERACY, profile['literacy'])}；"
        f"{lookup(TRUST, profile['trust'])}；{insurance_label(profile['insurance'])}。"
        "目前没有持有基金或股票。"
    )


def income_and_spending(profile: dict) -> str:
    """The separate annual-flow field used in the fifteen-condition template."""
    return (
        f"家庭年收入{lookup(INCOME, profile['income'])}，"
        f"全年总消费{lookup(INCOME, profile['consumption'])}"
    )
