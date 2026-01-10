"""
自定义因子数据处理器
用于加载挖掘出的自定义因子
"""
from pathlib import Path
from ruamel.yaml import YAML
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.log import get_module_logger

logger = get_module_logger("CustomFactorHandler")


class CustomFactorHandler(DataHandlerLP):
    """自定义因子数据处理器"""
    
    def __init__(
        self,
        instruments,
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        custom_factors=None,
        custom_factors_file=None,
        label_expr=None,
        **kwargs,
    ):
        """
        初始化自定义因子处理器
        
        Parameters
        ----------
        custom_factors : dict
            自定义因子字典，格式：{"因子名": "因子表达式"}
        custom_factors_file : str
            因子配置文件路径（YAML格式），如果提供则从文件加载good_factors
        label_expr : str
            标签表达式，例如："Ref($close, -1)/$close - 1"
        """
        # 优先从文件加载因子
        if custom_factors_file:
            self.custom_factors = self._load_factors_from_file(custom_factors_file)
        else:
            self.custom_factors = custom_factors or {}
        
        # 如果没有提供label_expr，根据文件类型自动生成
        if label_expr is None:
            if custom_factors_file and 'longterm' in str(custom_factors_file).lower():
                # 长期目标：未来20日平均价格
                ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 21)])
                self.label_expr = f"(({ref_list}) / 20) / $close - 1"
            else:
                # 短期目标：默认
                self.label_expr = "Ref($close, -2) / Ref($open, -1) - 1"
        else:
            self.label_expr = label_expr
        
        if len(self.custom_factors) == 0:
            raise ValueError("custom_factors不能为空，请提供至少一个因子或因子配置文件")
        
        logger.info(f"加载 {len(self.custom_factors)} 个自定义因子")
        
        # 构建特征配置
        feature_fields = list(self.custom_factors.values())
        feature_names = list(self.custom_factors.keys())
        
        # 构建标签配置
        label_fields = [self.label_expr]
        label_names = ["LABEL0"]
        
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (feature_fields, feature_names),
                    "label": (label_fields, label_names),
                },
                "freq": freq,
            },
        }
        
        # 使用 check_transform_proc 自动为 processors 添加 fit_start_time 和 fit_end_time
        # 这与 Alpha158 handler 的处理方式一致
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        
        # fit_start_time 和 fit_end_time 不应该传递给 DataHandler.__init__()
        # 它们已经通过 check_transform_proc 添加到 processors 的 kwargs 中了
        handler_kwargs = kwargs.copy()
        handler_kwargs.pop('fit_start_time', None)
        handler_kwargs.pop('fit_end_time', None)
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **handler_kwargs,
        )
    
    def _load_factors_from_file(self, config_file: str) -> dict:
        """
        从YAML文件加载因子配置
        
        Parameters
        ----------
        config_file : str
            配置文件路径
        
        Returns
        -------
        dict
            因子字典
        """
        config_path = Path(config_file)
        if not config_path.is_absolute():
            # 相对路径，尝试从项目目录查找
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"因子配置文件不存在: {config_path}")
        
        yaml = YAML(typ="safe", pure=True)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
        
        if config is None:
            raise ValueError(f"配置文件为空: {config_path}")
        
        # 从good_factors中加载因子
        good_factors = config.get('good_factors', {})
        if not isinstance(good_factors, dict):
            raise ValueError(f"配置文件格式错误，good_factors应该是字典: {config_path}")
        
        if len(good_factors) == 0:
            raise ValueError(f"配置文件中没有好因子: {config_path}")
        
        logger.info(f"从文件 {config_path} 加载了 {len(good_factors)} 个因子")
        return good_factors

