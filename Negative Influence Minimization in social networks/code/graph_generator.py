'''import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import util
import numpy as np
import os

def generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df):
    print("Generating graph...")
    print(welfare_analysis_df.columns)
    print(welfare_analysis_df)

    all_methods =  [ "pagerank", "cmia-o", "rbf", "protect", "naive_protect", "rl_fnim", "hifm",'rf',"gb"]
    x_ticks = [0, 5, 10, 15, 20, 25]  # Define the x-ticks to be used in all plots

    if not os.path.exists(welfare_output_folder):
        os.makedirs(welfare_output_folder)

    for t_k in tqdm(x_ticks):
        if t_k not in pp_dict_dict:
            continue
        plt.figure(figsize=(20, 10))
        temp_df = pd.DataFrame()
        prob_dict = {}
        for policy in methods:
            if policy not in pp_dict_dict[t_k]:
                continue
            for prob, count in pp_dict_dict[t_k][policy].items():
                if prob not in prob_dict:
                    prob_dict[prob] = {}
                prob_dict[prob][policy] = count
        for prob, policy_dict in prob_dict.items():
            temp_df = temp_df._append({"prob": prob, **policy_dict}, ignore_index=True)
        ax = plt.gca()
        temp_df.sort_values(by="prob", ascending=True, inplace=True)
        temp_df.plot.bar(x="prob", y=methods, ax=ax)
        plt.xticks(x_ticks)
        plt.xlabel("save probability")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"{welfare_output_folder}/bar_chart_{t_k}.png", bbox_inches='tight')
        plt.close()

    util.plot_k_vs_df_columns(welfare_analysis_df, "impressed", methods,
                              "Number of Negatively Influenced", f"{welfare_output_folder}/impressed.png", x_ticks)
    util.plot_k_vs_df_columns(welfare_analysis_df, "saved_fraction", methods,
                              "Fraction of saved nodes", f"{welfare_output_folder}/saved_fraction.png", x_ticks)
    util.plot_k_vs_df_columns(welfare_analysis_df, "min_prob", methods,
                              "Maximin sp", f"{welfare_output_folder}/min_prob.png", x_ticks)
    util.plot_k_vs_df_columns(welfare_analysis_df, "avg_prob", methods,
                              "Average sp", f"{welfare_output_folder}/avg_prob.png", x_ticks)

    util.plot_bar_df(welfare_analysis_df, "partition_diff", methods, "Difference in Average Save Probability", f"{welfare_output_folder}/partition_diff.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency", methods,
                              "Consistency", f"{welfare_output_folder}/consistency.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency2", methods,
                              "Consistency", f"{welfare_output_folder}/consistency2.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "consistency3", methods,
                              "Consistency", f"{welfare_output_folder}/consistency3.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "zero_prob_count", methods,
                              "Number of nodes with zero sp", f"{welfare_output_folder}/zero_prob_count.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr", methods,
                              "Correlation", f"{welfare_output_folder}/corr.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr2", methods,
                              "Correlation", f"{welfare_output_folder}/corr2.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr3", methods,
                              "Correlation", f"{welfare_output_folder}/corr3.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "corr4", methods,
                              "Correlation", f"{welfare_output_folder}/corr4.png", x_ticks)

    util.plot_bar_df(welfare_analysis_df, "partition_min_diff", methods, "Difference in Maximin sp", f"{welfare_output_folder}/partition_min_diff.png", x_ticks)

    util.plot_bar_df(welfare_analysis_df, "partition_prox_min_diff", methods, "Difference in Minimum distance from T", f"{welfare_output_folder}/partition_prox_min_diff.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "num_nodes_closer_to_T", methods,
                              "Number of vulnerable nodes closer to T than M", f"{welfare_output_folder}/num_nodes_closer_to_T.png", x_ticks)

    util.plot_bar_df(welfare_analysis_df, "avg_proximity_of_T", methods, "Average distance of T from M", f"{welfare_output_folder}/avg_proximity_of_T.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "avg_sp_at_distance", methods, "Average save probability", f"{welfare_output_folder}/avg_sp_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "std_dev_sp_at_distance", methods,
                            "Standard deviation of save probability", f"{welfare_output_folder}/std_dev_sp_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "avg_prox_at_distance", methods, "Average proximity from T", f"{welfare_output_folder}/avg_prox_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "improved_sp_at_distance", methods, "Improvement in save probability", f"{welfare_output_folder}/improved_sp_at_distance.png", x_ticks)

    util.plot_fairness_bars(welfare_analysis_df, "gini_coeff_list", methods, "Gini Coefficient", f"{welfare_output_folder}/gini_coeff_list.png", x_ticks)

    util.plot_k_vs_df_columns(welfare_analysis_df, "gini_coef", methods, "Gini Coefficient", f"{welfare_output_folder}/gini_coef.png", x_ticks)

    print("Graphs plotted and saved individually")

    # Generate trade-off and Pareto analysis graphs
    print("Generating trade-off and Pareto analysis graphs...")
    results = util.run_tradeoff_analysis(welfare_analysis_df, methods)
    pareto_efficient_points, pareto_methods = util.identify_pareto_frontier(results)

    util.plot_tradeoff_curve(results, f"{welfare_output_folder}/tradeoff_curve.png", x_ticks)
    util.plot_pareto_frontier(results, pareto_efficient_points, pareto_methods, f"{welfare_output_folder}/pareto_frontier.png", x_ticks)

    print("Trade-off and Pareto analysis graphs plotted and saved individually")'''
    

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import util
import numpy as np
import os

def generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df):
    print("Generating graph...")
    print(welfare_analysis_df.columns)
    print(welfare_analysis_df)

    all_methods =  [ "pagerank", "cmia-o", "rbf", "protect", "naive_protect", "rl_fnim", "hifm",'rf',"gb"]
    x_ticks = [0, 5, 10, 15, 20, 25]  # Define the x-ticks to be used in all plots

    if not os.path.exists(welfare_output_folder):
        os.makedirs(welfare_output_folder)

    for t_k in tqdm(x_ticks):
        if t_k not in pp_dict_dict:
            continue
        plt.figure(figsize=(20, 10))
        temp_df = pd.DataFrame()
        prob_dict = {}
        for policy in methods:
            if policy not in pp_dict_dict[t_k]:
                continue
            for prob, count in pp_dict_dict[t_k][policy].items():
                if prob not in prob_dict:
                    prob_dict[prob] = {}
                prob_dict[prob][policy] = count
        for prob, policy_dict in prob_dict.items():
            temp_df = temp_df._append({"prob": prob, **policy_dict}, ignore_index=True)
        ax = plt.gca()
        temp_df.sort_values(by="prob", ascending=True, inplace=True)
        temp_df.plot.bar(x="prob", y=methods, ax=ax)
        plt.xticks(x_ticks)
        plt.xlabel("save probability")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"{welfare_output_folder}/bar_chart_{t_k}.png", bbox_inches='tight')
        plt.close()

    # Ensure that 'seed count' is properly flattened before calling plotting functions
    welfare_analysis_df["seed count"] = welfare_analysis_df["seed count"].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) else x).astype(float)

    # Check if the DataFrame is empty after filtering
    def safe_plot_k_vs_df_columns(column_name, y_label, filename):
        df_filtered = welfare_analysis_df[welfare_analysis_df["seed count"] <= max(x_ticks)]
        if df_filtered.empty:
            print(f"No data to plot for {column_name} with seed count <= {max(x_ticks)}")
        else:
            util.plot_k_vs_df_columns(welfare_analysis_df, column_name, methods, y_label, filename, x_ticks)

    safe_plot_k_vs_df_columns("impressed", "Number of Negatively Influenced", f"{welfare_output_folder}/impressed.png")
    safe_plot_k_vs_df_columns("saved_fraction", "Fraction of saved nodes", f"{welfare_output_folder}/saved_fraction.png")
    safe_plot_k_vs_df_columns("min_prob", "Maximin sp", f"{welfare_output_folder}/min_prob.png")
    safe_plot_k_vs_df_columns("avg_prob", "Average sp", f"{welfare_output_folder}/avg_prob.png")

    util.plot_bar_df(welfare_analysis_df, "partition_diff", methods, "Difference in Average Save Probability", f"{welfare_output_folder}/partition_diff.png", x_ticks)

    safe_plot_k_vs_df_columns("consistency", "Consistency", f"{welfare_output_folder}/consistency.png")
    safe_plot_k_vs_df_columns("consistency2", "Consistency", f"{welfare_output_folder}/consistency2.png")
    safe_plot_k_vs_df_columns("consistency3", "Consistency", f"{welfare_output_folder}/consistency3.png")
    safe_plot_k_vs_df_columns("zero_prob_count", "Number of nodes with zero sp", f"{welfare_output_folder}/zero_prob_count.png")
    safe_plot_k_vs_df_columns("corr", "Correlation", f"{welfare_output_folder}/corr.png")
    safe_plot_k_vs_df_columns("corr2", "Correlation", f"{welfare_output_folder}/corr2.png")
    safe_plot_k_vs_df_columns("corr3", "Correlation", f"{welfare_output_folder}/corr3.png")
    safe_plot_k_vs_df_columns("corr4", "Correlation", f"{welfare_output_folder}/corr4.png")

    util.plot_bar_df(welfare_analysis_df, "partition_min_diff", methods, "Difference in Maximin sp", f"{welfare_output_folder}/partition_min_diff.png", x_ticks)
    util.plot_bar_df(welfare_analysis_df, "partition_prox_min_diff", methods, "Difference in Minimum distance from T", f"{welfare_output_folder}/partition_prox_min_diff.png", x_ticks)
    
    safe_plot_k_vs_df_columns("num_nodes_closer_to_T", "Number of vulnerable nodes closer to T than M", f"{welfare_output_folder}/num_nodes_closer_to_T.png")

    util.plot_bar_df(welfare_analysis_df, "avg_proximity_of_T", methods, "Average distance of T from M", f"{welfare_output_folder}/avg_proximity_of_T.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "avg_sp_at_distance", methods, "Average save probability", f"{welfare_output_folder}/avg_sp_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "std_dev_sp_at_distance", methods, "Standard deviation of save probability", f"{welfare_output_folder}/std_dev_sp_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "avg_prox_at_distance", methods, "Average proximity from T", f"{welfare_output_folder}/avg_prox_at_distance.png", x_ticks)
    util.plot_fairness_bars(welfare_analysis_df, "improved_sp_at_distance", methods, "Improvement in save probability", f"{welfare_output_folder}/improved_sp_at_distance.png", x_ticks)

    util.plot_fairness_bars(welfare_analysis_df, "gini_coeff_list", methods, "Gini Coefficient", f"{welfare_output_folder}/gini_coeff_list.png", x_ticks)

    safe_plot_k_vs_df_columns("gini_coef", "Gini Coefficient", f"{welfare_output_folder}/gini_coef.png")

    print("Graphs plotted and saved individually")

    # Generate trade-off and Pareto analysis graphs
    print("Generating trade-off and Pareto analysis graphs...")
    results = util.run_tradeoff_analysis(welfare_analysis_df, methods)
    pareto_efficient_points, pareto_methods = util.identify_pareto_frontier(results)

    util.plot_tradeoff_curve(results, f"{welfare_output_folder}/tradeoff_curve.png", x_ticks)
    util.plot_pareto_frontier(results, pareto_efficient_points, pareto_methods, f"{welfare_output_folder}/pareto_frontier.png", x_ticks)

    print("Trade-off and Pareto analysis graphs plotted and saved individually")

