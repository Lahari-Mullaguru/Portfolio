#["degree", "pagerank", "cmia-o","biog","rbf","fwrrs","fair-cmia-o","myopic_maximin","protect","naive_protect"]


#dataset = "HICHBA_10000_40000_0.9_0.9_0.9_el"
"""
This file is the main file for the experiments on the socfb-Caltech36 dataset.
"""
import util
import seed_selection
import graph_generator as gg
import main_utils as mu

dataset = "HICHBA_10000_40000_0.9_0.9_0.9_el"
k_n = 10
k_p = 25
imp_prob = 0.1
neg_policy = "degree"
weight = "value"
print(f"dataset: {dataset}, k_n: {k_n}, k_p: {k_p}, imp_prob: {imp_prob}, neg_policy: {neg_policy}, weight: {weight}")
network, neg_seeds, network_new_labels, neg_seeds_new_labels = util.init_settings(dataset, weight, imp_prob, neg_policy, k_n)


folder = f"../outputs/{dataset}_{weight}_weight_{neg_policy}_neg_policy_{imp_prob}"
methods =  [ "pagerank", "cmia-o", "rbf", "protect", "naive_protect", "rl_fnim", "hifm",'rf',"gb"]

remaining_methods =[]

# main method
if __name__ == '__main__':
    print(dataset, " dataset loaded the graph successfully! There are ",
          network.number_of_nodes(), " nodes and ", network.number_of_edges(), " edges")
    print("Negative seed set is", neg_seeds)
    if True:
        for method in remaining_methods:
            print("Running method:", method)
            pos_seeds = seed_selection.select_seeds(network, method, k_p, neg_seeds, imp_prob)
            pos_seed_file = f"{folder}/pos_seeds_{method}.txt"
            with open(pos_seed_file, 'w') as f:
                for seed in pos_seeds:
                    f.write("%s\n" % str(seed))
    if True:
        pp_dict_dict, welfare_analysis_df = mu.run_simulations(folder, neg_policy, network_new_labels, neg_seeds_new_labels, methods, remaining_methods,
                           df_columns_added=False)
    else:
        with open(f"{folder}/pp_dict_dict1.pkl", 'rb') as f:
            pp_dict_dict = pickle.load(f)
        welfare_analysis_df = pd.read_csv(f"{folder}/welfare_analysis_df1.csv", converters={'avg_sp_at_distance': eval,
                                                                                             'avg_prox_at_distance': eval,
                                                                                             'avg_vul_at_distance': eval,
                                                                                             'std_dev_sp_at_distance': eval,
                                                                                            'improved_sp_at_distance': eval,
                                                                                            'gini_coeff_list': eval
                                                                                            })
    welfare_output_folder = f"{folder}/figs/"
    gg.generate_graph(pp_dict_dict, methods, welfare_output_folder, welfare_analysis_df)
