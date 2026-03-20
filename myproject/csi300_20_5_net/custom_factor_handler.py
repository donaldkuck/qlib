"""
自定义因子数据处理器
用于加载挖掘出的自定义因子
"""
from pathlib import Path
from ruamel.yaml import YAML
import yaml
import pandas as pd
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.log import get_module_logger

logger = get_module_logger("CustomFactorHandler")


def _extract_factor_info(good_factors):
    """
    从因子配置中提取表达式和性能参数，支持新旧两种格式
    
    Parameters
    ----------
    good_factors : dict
        因子配置，可能是：
        - 旧格式：{factor_name: expression}
        - 新格式：{factor_name: {expression: ..., ic: ..., ...}}
        - 嵌套格式：{factor_name: {expression: {expression: ..., ic: ...}, ic: ...}}
        
    Returns
    -------
    dict
        表达式字典 {factor_name: expression}
    dict
        性能参数字典 {factor_name: {ic: ..., icir: ..., stability_score: ..., ...}}
    """
    expression_dict = {}
    metadata_dict = {}
    
    for factor_name, factor_value in good_factors.items():
        if isinstance(factor_value, str):
            # 旧格式：{factor_name: expression}
            expression_dict[factor_name] = factor_value
            metadata_dict[factor_name] = {}
        elif isinstance(factor_value, dict):
            # 新格式：{factor_name: {expression: ..., ic: ..., ...}}
            if 'expression' in factor_value:
                # 如果expression本身也是字典（嵌套格式），需要进一步提取
                expr_value = factor_value['expression']
                if isinstance(expr_value, dict) and 'expression' in expr_value:
                    # 嵌套格式：{factor_name: {expression: {expression: ..., ic: ...}, ic: ...}}
                    # 提取最内层的expression字符串
                    expression_dict[factor_name] = expr_value['expression']
                    # 合并性能参数：优先使用外层，其次使用内层
                    metadata = {k: v for k, v in factor_value.items() if k != 'expression'}
                    inner_metadata = {k: v for k, v in expr_value.items() if k != 'expression'}
                    metadata_dict[factor_name] = {**inner_metadata, **metadata}  # 外层覆盖内层
                elif isinstance(expr_value, str):
                    # 标准格式：{factor_name: {expression: "表达式字符串", ic: ..., ...}}
                    expression_dict[factor_name] = expr_value
                    # 提取性能参数（排除expression）
                    metadata = {k: v for k, v in factor_value.items() if k != 'expression'}
                    metadata_dict[factor_name] = metadata
                else:
                    # expression字段不是字符串也不是字典，尝试转换为字符串
                    expression_dict[factor_name] = str(expr_value)
                    metadata = {k: v for k, v in factor_value.items() if k != 'expression'}
                    metadata_dict[factor_name] = metadata
            else:
                # 格式错误，尝试作为表达式使用
                expression_dict[factor_name] = str(factor_value)
                metadata_dict[factor_name] = {}
        else:
            # 其他格式，转换为字符串
            expression_dict[factor_name] = str(factor_value)
            metadata_dict[factor_name] = {}
    
    return expression_dict, metadata_dict


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
        add_stock_id_onehot=True,  # 是否添加股票ID的one-hot编码
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
        self._factor_config_label_expr = None

        # 优先从文件加载因子
        if custom_factors_file:
            self.custom_factors = self._load_factors_from_file(custom_factors_file)
        else:
            self.custom_factors = custom_factors or {}
        
        # 如果没有显式提供 label_expr，优先从 factor config YAML 顶层读取
        if label_expr is None and custom_factors_file:
            label_expr = self._factor_config_label_expr or self._load_label_expr_from_file(custom_factors_file)
            if label_expr:
                logger.info(f"✓ 从因子配置文件读取 label_expr: {label_expr}")

        # 如果没有提供label_expr，根据文件类型自动生成
        if label_expr is None:
            if custom_factors_file and 'longterm' in str(custom_factors_file).lower():
                # 长期目标：未来20日平均价格
                ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 21)])
                self.label_expr = f"(({ref_list}) / 20) / $close - 1"
                logger.warning(f"⚠️  label_expr未提供，自动生成长期标签: {self.label_expr}...")
            else:
                # 短期目标：默认
                ref_list = " + ".join([f"Ref($close, -{i})" for i in range(1, 6)])
                self.label_expr = f"(({ref_list}) / 5) / $close - 1"
                logger.warning(f"⚠️  label_expr未提供，自动生成短期标签: {self.label_expr}...")
        else:
            self.label_expr = label_expr
            logger.info(f"✓ 使用YAML配置的label_expr: {self.label_expr}")
        
        if len(self.custom_factors) == 0:
            raise ValueError("custom_factors不能为空，请提供至少一个因子或因子配置文件")
        
        logger.info(f"加载 {len(self.custom_factors)} 个自定义因子")
        
        # 保存是否添加股票ID one-hot编码
        # 注意：add_stock_id_onehot 参数可能来自构造函数参数或kwargs
        self.add_stock_id_onehot = add_stock_id_onehot  # 优先使用构造函数参数
        if 'add_stock_id_onehot' in kwargs:
            kwargs.pop('add_stock_id_onehot')  # 从kwargs中移除，避免传递给父类
        if self.add_stock_id_onehot:
            logger.info(f"✓ 已启用股票ID one-hot编码")
        
        # 构建特征配置
        feature_fields = list(self.custom_factors.values())
        feature_names = list(self.custom_factors.keys())
        
        # 如果需要添加股票ID的one-hot编码，需要在数据加载后处理
        # 这里先保存配置，后续在fetch时添加
        
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
        
        # 如果使用固定日期获取成分股，需要转换为股票列表
        processed_instruments = instruments
        if isinstance(instruments, dict) and 'market' in instruments:
            use_fixed_date = instruments.get('use_fixed_date', False)
            fixed_date = instruments.get('fixed_date', None)
            
            if use_fixed_date and fixed_date:
                # 使用固定日期获取成分股列表
                try:
                    from qlib.data import D
                    market = instruments.get('market', '')
                    instruments_obj = D.instruments(market=market, filter_pipe=instruments.get('filter_pipe', []))
                    stock_list = D.list_instruments(
                        instruments=instruments_obj,
                        start_time=fixed_date,
                        end_time=fixed_date,
                        as_list=True
                    )
                    if not isinstance(stock_list, list):
                        stock_list = list(stock_list)
                    
                    # 验证股票数量
                    expected_count = {'csi100': 100, 'csi300': 300, 'csi500': 500}.get(market.lower(), None)
                    if expected_count:
                        if len(stock_list) != expected_count:
                            logger.warning(f"{market} 在 {fixed_date} 的成分股数量为 {len(stock_list)}，期望 {expected_count}")
                            if len(stock_list) < expected_count * 0.8:
                                # 如果数量明显不足，使用时间范围获取
                                logger.info(f"尝试使用时间范围获取成分股...")
                                stock_list = D.list_instruments(
                                    instruments=instruments_obj,
                                    start_time=start_time,
                                    end_time=end_time,
                                    as_list=True
                                )
                                if not isinstance(stock_list, list):
                                    stock_list = list(stock_list)
                                # 去重并排序，取前expected_count只
                                stock_list = sorted(list(set(stock_list)))[:expected_count]
                            elif len(stock_list) > expected_count:
                                # 如果数量过多，只取前expected_count只（按代码排序）
                                stock_list = sorted(stock_list)[:expected_count]
                                logger.info(f"限制股票数量为 {expected_count} 只")
                        
                        # 最终验证
                        if len(stock_list) == expected_count:
                            logger.info(f"✓ {market} 成分股数量正确: {len(stock_list)} 只")
                        else:
                            logger.warning(f"⚠️  {market} 成分股数量为 {len(stock_list)}，期望 {expected_count}")
                    
                    processed_instruments = stock_list
                    logger.info(f"使用固定日期 {fixed_date} 获取 {market} 成分股: {len(processed_instruments)} 只股票")
                except Exception as e:
                    logger.warning(f"使用固定日期获取成分股失败: {e}，使用原始配置")
                    # 继续使用原始配置
        
        super().__init__(
            instruments=processed_instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **handler_kwargs,
        )
        
        # 如果启用股票ID one-hot编码，初始化股票列表
        if self.add_stock_id_onehot:
            self._init_stock_id_mapping()

    def _load_label_expr_from_file(self, config_file: str):
        """从 factor config YAML 顶层读取 label_expr。"""
        try:
            config_path = Path(config_file).expanduser()
            if not config_path.is_absolute():
                config_path = (Path.cwd() / config_path).resolve()
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            label_expr = data.get("label_expr")
            if isinstance(label_expr, str) and label_expr.strip():
                return label_expr.strip()
        except Exception as e:
            logger.warning(f"从因子配置文件读取 label_expr 失败: {e}")
        return None
    
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
            因子字典 {factor_name: expression}，只包含表达式字符串
        """
        config_path = Path(config_file)
        # if not config_path.is_absolute():
        #     # 相对路径，尝试从项目目录查找
        #     project_root = Path(__file__).parent.parent
        #     config_path = project_root / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"因子配置文件不存在: {config_path}")
        
        yaml = YAML(typ="safe", pure=True)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
        
        if config is None:
            raise ValueError(f"配置文件为空: {config_path}")
        
        # 从good_factors中加载因子
        good_factors = config.get('good_factors', {})
        file_label_expr = config.get("label_expr")
        if isinstance(file_label_expr, str) and file_label_expr.strip():
            self._factor_config_label_expr = file_label_expr.strip()
        if not isinstance(good_factors, dict):
            raise ValueError(f"配置文件格式错误，good_factors应该是字典: {config_path}")
        
        if len(good_factors) == 0:
            raise ValueError(f"配置文件中没有好因子: {config_path}")
        
        # 提取因子表达式（支持新旧两种格式）
        factor_expressions, _ = _extract_factor_info(good_factors)
        
        logger.info(f"从文件 {config_path} 加载了 {len(factor_expressions)} 个因子")
        return factor_expressions
    
    
    def _init_stock_id_mapping(self):
        """初始化股票ID到索引的映射（用于one-hot编码）"""
        from qlib.data import D
        
        try:
            # 如果instruments已经是列表，直接使用
            if isinstance(self.instruments, list):
                stock_list = self.instruments
                logger.info(f"从instruments列表获取股票: {len(stock_list)} 只")
            # 获取所有股票列表
            elif isinstance(self.instruments, dict) and 'market' in self.instruments:
                market = self.instruments['market']
                
                # 检查是否使用固定日期获取成分股
                use_fixed_date = self.instruments.get('use_fixed_date', False)
                fixed_date = self.instruments.get('fixed_date', None)
                
                if use_fixed_date and fixed_date:
                    # 使用固定日期获取成分股（确保股票数量一致）
                    logger.info(f"使用固定日期 {fixed_date} 获取 {market} 成分股")
                    instruments_obj = D.instruments(market=market, filter_pipe=self.instruments.get('filter_pipe', []))
                    stock_list = D.list_instruments(
                        instruments=instruments_obj,
                        start_time=fixed_date,
                        end_time=fixed_date,
                        as_list=True
                    )
                    if not isinstance(stock_list, list):
                        stock_list = list(stock_list)
                    
                    # 验证股票数量
                    expected_count = {'csi100': 100, 'csi300': 300, 'csi500': 500}.get(market.lower(), None)
                    if expected_count:
                        if len(stock_list) != expected_count:
                            logger.warning(f"{market} 在 {fixed_date} 的成分股数量为 {len(stock_list)}，期望 {expected_count}")
                            if len(stock_list) < expected_count * 0.8:  # 如果数量明显不足
                                logger.info(f"尝试使用时间范围获取成分股...")
                                stock_list = D.list_instruments(
                                    instruments=instruments_obj,
                                    start_time=self.start_time,
                                    end_time=self.end_time,
                                    as_list=True
                                )
                                if not isinstance(stock_list, list):
                                    stock_list = list(stock_list)
                                # 取最近的成分股（去重并排序）
                                stock_list = sorted(list(set(stock_list)))[:expected_count]
                                logger.info(f"从时间范围获取了 {len(stock_list)} 只股票")
                            elif len(stock_list) > expected_count:
                                # 如果数量过多，只取前expected_count只（按代码排序）
                                stock_list = sorted(stock_list)[:expected_count]
                                logger.info(f"限制股票数量为 {expected_count} 只")
                        
                        # 最终验证
                        if len(stock_list) == expected_count:
                            logger.info(f"✓ {market} 成分股数量正确: {len(stock_list)} 只")
                        else:
                            logger.warning(f"⚠️  {market} 成分股数量为 {len(stock_list)}，期望 {expected_count}")
                else:
                    # 正常处理：使用时间范围获取
                    instruments_obj = D.instruments(market=market, filter_pipe=self.instruments.get('filter_pipe', []))
                    stock_list = D.list_instruments(
                        instruments=instruments_obj,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        as_list=True
                    )
            else:
                instruments_obj = self.instruments
                stock_list = D.list_instruments(
                    instruments=instruments_obj,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    as_list=True
                )
            
            if not isinstance(stock_list, list):
                stock_list = list(stock_list)
            
            # 排序以确保一致性
            stock_list = sorted(stock_list)
            
            # 创建股票ID到索引的映射
            self.stock_id_to_idx = {stock: idx for idx, stock in enumerate(stock_list)}
            self.stock_idx_to_id = {idx: stock for stock, idx in self.stock_id_to_idx.items()}
            self.n_stocks = len(stock_list)
            
            logger.info(f"初始化股票ID映射: {self.n_stocks} 只股票")
            logger.info(f"股票列表示例: {stock_list[:5]}")
            
        except Exception as e:
            logger.warning(f"初始化股票ID映射失败: {e}，将禁用股票ID one-hot编码")
            self.add_stock_id_onehot = False
            self.n_stocks = 0
    
    def _add_stock_id_onehot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为数据添加股票ID的one-hot编码特征
        
        Parameters
        ----------
        data : pd.DataFrame
            原始数据（MultiIndex: [datetime, instrument]）
            
        Returns
        -------
        pd.DataFrame
            添加了股票ID one-hot编码特征的数据
        """
        if not self.add_stock_id_onehot:
            logger.debug("股票ID one-hot编码未启用")
            return data
        
        if self.n_stocks == 0:
            logger.warning("n_stocks为0，无法添加股票ID one-hot编码")
            return data
        
        if not isinstance(data.index, pd.MultiIndex):
            logger.warning("数据索引不是MultiIndex格式，无法添加股票ID one-hot编码")
            return data
        
        logger.debug(f"开始添加股票ID one-hot编码: 数据形状={data.shape}, n_stocks={self.n_stocks}")
        
        # 获取股票ID
        if 'instrument' in data.index.names:
            stock_ids = data.index.get_level_values('instrument')
        elif data.index.nlevels >= 2:
            stock_ids = data.index.get_level_values(1)  # 假设第二层是股票ID
        else:
            logger.warning("无法从索引中获取股票ID")
            return data
        
        # 创建one-hot编码矩阵
        onehot_matrix = np.zeros((len(data), self.n_stocks), dtype=np.float32)
        
        matched_count = 0
        for idx, stock_id in enumerate(stock_ids):
            if stock_id in self.stock_id_to_idx:
                stock_idx = self.stock_id_to_idx[stock_id]
                onehot_matrix[idx, stock_idx] = 1.0
                matched_count += 1
            else:
                logger.debug(f"股票ID {stock_id} 不在映射表中")
        
        logger.debug(f"one-hot编码匹配: {matched_count}/{len(stock_ids)} 只股票")
        
        # 创建one-hot特征列名
        onehot_columns = [f"stock_id_{i}" for i in range(self.n_stocks)]
        
        # 创建one-hot特征DataFrame
        onehot_df = pd.DataFrame(
            onehot_matrix,
            index=data.index,
            columns=pd.MultiIndex.from_product([['feature'], onehot_columns])
        )
        
        # 合并原始特征和one-hot特征
        if isinstance(data.columns, pd.MultiIndex):
            # 如果原始数据是MultiIndex列，直接合并
            result = pd.concat([data, onehot_df], axis=1)
        else:
            # 如果原始数据是普通列，需要转换为MultiIndex
            data_multi = pd.DataFrame(
                data.values,
                index=data.index,
                columns=pd.MultiIndex.from_product([['feature'], data.columns])
            )
            result = pd.concat([data_multi, onehot_df], axis=1)
        
        # 验证合并结果
        if isinstance(result.columns, pd.MultiIndex):
            feature_cols = result.columns.get_level_values(1)
            onehot_cols = [col for col in feature_cols if str(col).startswith('stock_id_')]
            logger.debug(f"成功添加one-hot编码: {len(onehot_cols)} 维，总特征数: {len(result.columns)}")
        else:
            logger.warning("合并后的数据列不是MultiIndex格式")
        
        return result
    
    def setup_data(self, **kwargs):
        """
        重写setup_data方法，在数据处理后添加股票ID one-hot编码特征
        """
        logger.info(f"CustomFactorHandler.setup_data() 被调用，add_stock_id_onehot={self.add_stock_id_onehot}")
        self.add_stock_id_onehot = False
        # 如果启用股票ID one-hot编码但还没有初始化映射，先初始化
        if self.add_stock_id_onehot and (not hasattr(self, 'n_stocks') or self.n_stocks == 0):
            logger.info("初始化股票ID映射...")
            self._init_stock_id_mapping()
        
        # 调用父类的setup_data方法（这会创建_learn和_infer数据）
        logger.debug("调用父类setup_data...")
        super().setup_data(**kwargs)
        logger.debug(f"父类setup_data完成，_learn存在: {hasattr(self, '_learn') and self._learn is not None}, _infer存在: {hasattr(self, '_infer') and self._infer is not None}")
        
        # 如果启用股票ID one-hot编码，在_learn和_infer数据中添加one-hot编码
        if self.add_stock_id_onehot:
            logger.info(f"开始添加股票ID one-hot编码，n_stocks={getattr(self, 'n_stocks', 0)}")
            # 确保n_stocks已初始化
            if not hasattr(self, 'n_stocks') or self.n_stocks == 0:
                logger.warning("股票ID映射未初始化，尝试重新初始化...")
                self._init_stock_id_mapping()
            
            if self.n_stocks > 0:
                if hasattr(self, '_learn') and self._learn is not None:
                    logger.info(f"准备为_learn数据添加股票ID one-hot编码: {self.n_stocks} 维")
                    original_cols = len(self._learn.columns) if isinstance(self._learn.columns, pd.MultiIndex) else 0
                    self._learn = self._add_stock_id_onehot_features(self._learn)
                    # 验证是否成功添加
                    if isinstance(self._learn, pd.DataFrame) and isinstance(self._learn.columns, pd.MultiIndex):
                        feature_cols = self._learn.columns.get_level_values(1)
                        onehot_cols = [col for col in feature_cols if str(col).startswith('stock_id_')]
                        new_cols = len(self._learn.columns) if isinstance(self._learn.columns, pd.MultiIndex) else 0
                        logger.info(f"✓ 为_learn数据添加了股票ID one-hot编码: {len(onehot_cols)} 维（期望 {self.n_stocks} 维）")
                        logger.info(f"  原始特征数: {original_cols}, 添加后特征数: {new_cols}")
                    else:
                        logger.warning("⚠️ 无法验证one-hot编码是否成功添加")
                else:
                    logger.warning(f"⚠️ _learn数据不存在，无法添加one-hot编码 (hasattr: {hasattr(self, '_learn')}, is not None: {hasattr(self, '_learn') and self._learn is not None if hasattr(self, '_learn') else 'N/A'})")
                
                if hasattr(self, '_infer') and self._infer is not None:
                    logger.info(f"准备为_infer数据添加股票ID one-hot编码: {self.n_stocks} 维")
                    original_cols = len(self._infer.columns) if isinstance(self._infer.columns, pd.MultiIndex) else 0
                    self._infer = self._add_stock_id_onehot_features(self._infer)
                    # 验证是否成功添加
                    if isinstance(self._infer, pd.DataFrame) and isinstance(self._infer.columns, pd.MultiIndex):
                        feature_cols = self._infer.columns.get_level_values(1)
                        onehot_cols = [col for col in feature_cols if str(col).startswith('stock_id_')]
                        new_cols = len(self._infer.columns) if isinstance(self._infer.columns, pd.MultiIndex) else 0
                        logger.info(f"✓ 为_infer数据添加了股票ID one-hot编码: {len(onehot_cols)} 维（期望 {self.n_stocks} 维）")
                        logger.info(f"  原始特征数: {original_cols}, 添加后特征数: {new_cols}")
                    else:
                        logger.warning("⚠️ 无法验证one-hot编码是否成功添加")
                else:
                    logger.warning(f"⚠️ _infer数据不存在，无法添加one-hot编码 (hasattr: {hasattr(self, '_infer')}, is not None: {hasattr(self, '_infer') and self._infer is not None if hasattr(self, '_infer') else 'N/A'})")
            else:
                logger.warning("⚠️ n_stocks为0，无法添加股票ID one-hot编码")
        else:
            logger.debug("股票ID one-hot编码未启用")
