"""
长期因子挖掘兼容入口。

核心逻辑已合并到 analyze_custom_factors.py。
本文件仅保留长期默认参数，避免两份实现继续分叉。
"""

from pathlib import Path

from analyze_custom_factors import analyze_custom_factors, make_future_avg_price_label_expr


def analyze_custom_factors_longterm(
    instruments="csi1000",
    label_expr=None,
    custom_factors=None,
    start_time=None,
    end_time=None,
    freq="day",
    limit_data_days=None,
    min_valid_days=10,
    output_dir=None,
    train_start_time=None,
    train_end_time=None,
    valid_start_time=None,
    valid_end_time=None,
    min_ic_train=0.02,
    min_ic_valid=0.02,
    max_ic_decay=1.5,
    require_same_sign=False,
    use_rolling_evaluation=False,
    rolling_train_years=5,
    rolling_valid_years=2,
    rolling_step_months=12,
    min_rolling_windows=3,
    use_segment_evaluation=True,
    segment_months=6,
    min_segment_count=4,
    segment_min_ic_abs=None,
    force_full_scan=False,
    use_result_cache=True,
    resume_from_cache=True,
):
    """长期默认参数封装，核心逻辑委托给 analyze_custom_factors。"""
    if label_expr is None:
        label_expr = make_future_avg_price_label_expr(20)

    if output_dir is None:
        output_dir = Path(__file__).parent / "factor_evaluation_results_longterm"

    if start_time is None:
        start_time = "2015-01-01"
    if end_time is None:
        end_time = "2025-12-31"
    if train_start_time is None:
        train_start_time = "2015-01-01"
    if train_end_time is None:
        train_end_time = "2024-12-31"
    if valid_start_time is None:
        valid_start_time = "2025-01-01"
    if valid_end_time is None:
        valid_end_time = "2025-12-31"

    return analyze_custom_factors(
        instruments=instruments,
        label_expr=label_expr,
        custom_factors=custom_factors,
        config_path=Path(__file__).parent / "factor_config_longterm.yaml",
        result_prefix="custom_factors_longterm",
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        limit_data_days=limit_data_days,
        min_valid_days=min_valid_days,
        output_dir=output_dir,
        train_start_time=train_start_time,
        train_end_time=train_end_time,
        valid_start_time=valid_start_time,
        valid_end_time=valid_end_time,
        min_ic_train=min_ic_train,
        min_ic_valid=min_ic_valid,
        max_ic_decay=max_ic_decay,
        require_same_sign=require_same_sign,
        use_rolling_evaluation=use_rolling_evaluation,
        rolling_train_years=rolling_train_years,
        rolling_valid_years=rolling_valid_years,
        rolling_step_months=rolling_step_months,
        min_rolling_windows=min_rolling_windows,
        use_segment_evaluation=use_segment_evaluation,
        segment_months=segment_months,
        min_segment_count=min_segment_count,
        segment_min_ic_abs=segment_min_ic_abs,
        force_full_scan=force_full_scan,
        use_result_cache=use_result_cache,
        resume_from_cache=resume_from_cache,
    )


def main():
    """长期因子挖掘默认入口。"""
    print("=" * 80)
    print("长期因子挖掘入口（核心逻辑已合并到 analyze_custom_factors.py）")
    print("=" * 80)

    results = analyze_custom_factors_longterm(
        instruments="csi300",
    )

    if results is not None:
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
    else:
        print("\n分析失败，请检查错误信息。")


if __name__ == "__main__":
    main()
