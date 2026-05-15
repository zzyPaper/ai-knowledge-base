"""THS industry board → A-share ETF/fund code mapping.

Add entries as they become relevant. Sectors without ETF mappings will show
as "行业组合" in reports (not an error — they can still be traded via individual stocks).
"""

SECTOR_ETF_MAP: dict[str, dict[str, str]] = {
    "半导体": {"code": "512480", "name": "半导体ETF"},
    "白酒": {"code": "512690", "name": "酒ETF"},
    "白色家电": {"code": "159996", "name": "家电ETF"},
    "保险": {"code": "167301", "name": "保险主题(LOF)"},
    "证券": {"code": "512880", "name": "证券ETF"},
    "银行": {"code": "512800", "name": "银行ETF"},
    "房地产": {"code": "512200", "name": "房地产ETF"},
    "煤炭": {"code": "515220", "name": "煤炭ETF"},
    "钢铁": {"code": "515210", "name": "钢铁ETF"},
    "军工": {"code": "512660", "name": "军工ETF"},
    "有色金属": {"code": "512400", "name": "有色金属ETF"},
    "新能源": {"code": "515700", "name": "新能源ETF"},
    "光伏设备": {"code": "515790", "name": "光伏ETF"},
    "电力": {"code": "159611", "name": "电力ETF"},
    "电池": {"code": "159840", "name": "锂电池ETF"},
    "汽车整车": {"code": "515030", "name": "新能源车ETF"},
    "医疗器械": {"code": "159828", "name": "医疗ETF"},
    "生物制品": {"code": "512170", "name": "医药ETF"},
    "中药": {"code": "159647", "name": "中药ETF"},
    "化学制药": {"code": "512170", "name": "医药ETF"},
    "医疗服务": {"code": "512170", "name": "医药ETF"},
    "计算机应用": {"code": "512720", "name": "计算机ETF"},
    "计算机设备": {"code": "512720", "name": "计算机ETF"},
    "通信设备": {"code": "515880", "name": "通信ETF"},
    "消费电子": {"code": "159997", "name": "电子ETF"},
    "传媒": {"code": "512980", "name": "传媒ETF"},
    "养殖业": {"code": "159865", "name": "养殖ETF"},
    "种植业与林业": {"code": "159825", "name": "农业ETF"},
    "农产品加工": {"code": "159825", "name": "农业ETF"},
    "食品加工": {"code": "515170", "name": "食品饮料ETF"},
    "饮料制造": {"code": "512690", "name": "酒ETF"},
    "环保": {"code": "512580", "name": "环保ETF"},
    "环保工程": {"code": "512580", "name": "环保ETF"},
    "自动化设备": {"code": "562500", "name": "机器人ETF"},
    "通用设备": {"code": "562500", "name": "机器人ETF"},
    "机器人": {"code": "562500", "name": "机器人ETF"},
    "人工智能": {"code": "515070", "name": "AIETF"},
    "通信服务": {"code": "515880", "name": "通信ETF"},
    "国防军工": {"code": "512660", "name": "军工ETF"},
    "物流": {"code": "516910", "name": "物流ETF"},
    "医疗器械": {"code": "159883", "name": "医疗器械ETF"},
    "电子化学品": {"code": "159997", "name": "电子ETF"},
    "风电设备": {"code": "516160", "name": "新能源ETF"},
    "电机": {"code": "159633", "name": "工业ETF"},
    "工程机械": {"code": "560660", "name": "机械ETF"},
    "贵金属": {"code": "159934", "name": "黄金ETF"},
    "汽车零部件": {"code": "515030", "name": "新能源车ETF"},
    "小金属": {"code": "512400", "name": "有色金属ETF"},
    "能源金属": {"code": "512400", "name": "有色金属ETF"},
    "其他电子": {"code": "159997", "name": "电子ETF"},
    "元件": {"code": "159997", "name": "电子ETF"},
    "工业金属": {"code": "512400", "name": "有色金属ETF"},
    "光学光电子": {"code": "159997", "name": "电子ETF"},
    "文化传媒": {"code": "512980", "name": "传媒ETF"},
    "化学制品": {"code": "159865", "name": "化工ETF"},
    "服装家纺": {"code": "159987", "name": "消费ETF"},
    "包装印刷": {"code": "159987", "name": "消费ETF"},
    "厨卫电器": {"code": "159996", "name": "家电ETF"},
    "互联网电商": {"code": "159985", "name": "电商ETF"},
    "纺织制造": {"code": "159987", "name": "消费ETF"},
    "食品加工制造": {"code": "515170", "name": "食品饮料ETF"},
    "建筑材料": {"code": "516750", "name": "建材ETF"},
    "建筑装饰": {"code": "516750", "name": "建材ETF"},
    "仪器仪表": {"code": "562500", "name": "机器人ETF"},
    "非汽车交运": {"code": "516910", "name": "物流ETF"},
    "石油加工": {"code": "159981", "name": "能源化工ETF"},
    "石油矿业开采": {"code": "159981", "name": "能源化工ETF"},
    "基础化学": {"code": "159865", "name": "化工ETF"},
    "化工合成材料": {"code": "159865", "name": "化工ETF"},
}


def get_etf(sector_name: str) -> str:
    """Return ETF code + name for a sector, or a fallback string."""
    mapping = SECTOR_ETF_MAP.get(sector_name)
    if mapping:
        return f"{mapping['code']}({mapping['name']})"
    return "行业组合"


def get_etf_code(sector_name: str) -> str:
    """Return just the ETF fund code."""
    mapping = SECTOR_ETF_MAP.get(sector_name)
    return mapping["code"] if mapping else "—"
