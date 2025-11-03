from wma import WMAConfig, run_weighted_factor
from ic_analysis import run_ic_analysis

factor_name = "cnn_10_03_72h_v4"
start_time = "2025-04-01 00:00:00"
end_time = "2025-08-31 23:00:00"

for time_span_hours in [12]:
    for decay in [1, 0.8, 0.6, 0.4, 0.2]:
        output_factor_name = f"{factor_name}_wma_{time_span_hours}h_{decay:.1f}"
        config = WMAConfig(
            factor_name=factor_name,
            start_time=start_time,
            end_time=end_time,
            time_span_hours=time_span_hours,
            decay=decay,
            output_factor_name=output_factor_name,
        )

        print(f"🚀 Generating WMA factor: {output_factor_name}...")
        output_files = run_weighted_factor(config)

        print(f"📊 Running IC analysis for factor: {output_factor_name}...")
        run_ic_analysis(
            factor_name=output_factor_name,
            start_time=start_time,
            end_time=end_time,
        )