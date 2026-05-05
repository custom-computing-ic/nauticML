from nautic import taskx

from heapq import nlargest
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import os

class Results:
    @taskx
    def create_result_table(ctx):
        strat = ctx.strategy
        log = ctx.log

        table = Results.get_top_n_results(strat)

        log.artifact(key='all-top-results-table', table=table)

        # save table as markdown in figures subfolder of save dir
        df = pd.DataFrame(table)

        columns = ["Opt-Mode"] + [c for c in df.columns if c != "Opt-Mode"]
        df = df[columns]
        md_df = df.to_markdown(index=False)

        log.info("\nTop strategy results:\n%s", md_df)

        md_path = Results.get_figures_path("all_top_results_table.md", strat.save_dir.get())
        with open(md_path, "w") as f:
            f.write("# Top Strategy Results\n\n")
            f.write(md_df)

    @taskx
    def create_4d_pareto_figures(ctx):
        strat = ctx.strategy
        log = ctx.log

        results = strat.results
        optimal_table = Results.get_top_n_results(strat)

        # collect all the points into a list
        all_points = []
        for _, results_list in results.items():
            for r in results_list:
                metrics = r["metrics"].copy()
                if "score" in metrics:
                    del metrics["score"]

                metric_names_local = list(metrics.keys())

                if len(metric_names_local) != 4:
                    log.error("❌ We are trying to display a 4D-pareto front without 4 metrics. Not supported")
                    return

                all_points.append([metrics[m] for m in metric_names_local])
        
        points = np.array(all_points)
        maximize_local = [True]*4

        # get only the points on the pareto frontier
        pareto_mask = Results.is_pareto_efficient(points, maximize_local)
        pareto_points = points[pareto_mask]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Axes: first 3 metrics, 4th metric is color bar
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        color_metric = points[:, 3]

        # Plot all the points of the bayes opt
        sc_normal = ax.scatter(x, y, z, c=color_metric, cmap='coolwarm', s=50, label='Configurations')

        # Plot all the points from the convex hull as the pareto frontier
        if len(pareto_points) >= 4:
            hull = ConvexHull(pareto_points[:, :3])
            # Hull lines
            for simplex in hull.simplices:
                ax.plot(pareto_points[simplex, 0],
                        pareto_points[simplex, 1],
                        pareto_points[simplex, 2],
                        "k-", linewidth=1.5)
                
            # Pareto surface points (black outline)
            hull_indices = np.unique(hull.simplices.flatten())
            pareto_surface_points = pareto_points[hull_indices]
            ax.scatter(pareto_surface_points[:, 0],
                    pareto_surface_points[:, 1],
                    pareto_surface_points[:, 2],
                    c=color_metric[pareto_mask][hull_indices],
                    cmap='coolwarm',
                    s=100,
                    edgecolors='k',
                    linewidths=2,
                    label='Pareto front (surface)')

        # Highlight the optimal points found from the different strategies by labelling them and adding a green cross
        if len(optimal_table) > 0:
            opt_points = []
            opt_labels = []
            for row in optimal_table:
                metrics = {k: v for k, v in row.items() if k not in ['Opt-Mode', 'hyperparameters']}
                opt_points.append([metrics[m] for m in metric_names_local])
                opt_labels.append(row['Opt-Mode'])

            # Convert to numpy array and deduplicate
            opt_points = np.array(opt_points)
            opt_labels = np.array(opt_labels)
            unique_rows, unique_indices = np.unique(opt_points, axis=0, return_index=True)
            opt_points = unique_rows
            opt_labels = opt_labels[unique_indices]

            ax.scatter(opt_points[:, 0], opt_points[:, 1], opt_points[:, 2],
                    marker='x', color='green', s=120, linewidths=3, zorder=10, label='Optimal solutions')

            for i in range(len(opt_points)):
                ax.text(opt_points[i, 0], opt_points[i, 1], opt_points[i, 2],
                        opt_labels[i], color='green', fontsize=8)

        # Colorbar and labels
        plt.colorbar(sc_normal, ax=ax, label=metric_names_local[3])
        ax.set_xlabel(metric_names_local[0])
        ax.set_ylabel(metric_names_local[1])
        ax.set_zlabel(metric_names_local[2])
        ax.set_title("4D Pareto Front of Bayes Opt results")

        ax.legend()

        plt_path = Results.get_figures_path("4d-pareto-front-plot.png", strat.save_dir.get())
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')

        # TODO: actually fix the artifact to upload a real image
        log.artifact(key='4d-pareto-front-plot', image=plt_path)

    @taskx
    def create_correlation_matrix(ctx):
        strat = ctx.strategy
        log = ctx.log

        points = []
        for _, results_list in strat.results.items():
            for result in results_list:
                row = {}

                row.update(result.get("hyperparameters", {}))
                row.update(result.get("metrics", {}))

                if "score" in row:
                    del row["score"]

                points.append(row)
        
        if len(points) < 2:
            log.warning("⚠️ There are not enough points (< 2) to do Spearman correlation on these results.")

        df = pd.DataFrame(points)
        corr = df.corr(method="pearson")

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Correlation Matrix between metrics and hyperparameters across all configurations")
        plt.tight_layout()

        plt_path = Results.get_figures_path("correlation_heatmap.png", strat.save_dir.get())
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')

        # TODO: actually fix the artifact to upload a real image
        log.artifact(key='4d-pareto-front-plot', image=plt_path)

    @staticmethod
    def get_figures_path(path, save_dir):
        figures_dir = os.path.join(save_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True) 

        return os.path.join(figures_dir, path)

    @staticmethod
    def get_top_n_results(strat):
        results = strat.results
        table = []

        for strat_index, strat_results in results.items():
            strat_obj = strat.strategies[strat_index]

            # just log the top n results according to the top metric in the table for particular strategy
            for i, top_result in enumerate(nlargest(strat_obj.top_n, strat_results, key=lambda r: r["metrics"][strat_obj.top_metric])):
                row_name = strat_obj.name

                # add top name only if we keep track of multiple top results
                if strat_obj.top_n > 1:
                    row_name = f"{row_name}-Top{i+1}"

                row = {"Opt-Mode": row_name}

                row.update(top_result.get("hyperparameters", {}))
                row.update(top_result.get("metrics", {}))

                table.append(row)
        
        return table

    # Function to gauge pareto frontier
    @staticmethod
    def is_pareto_efficient(points, maximize):
        data = points.copy()
        for i, m in enumerate(maximize):
            if m:
                data[:, i] = -data[:, i]
        n = data.shape[0]
        is_efficient = np.ones(n, dtype=bool)
        for i, c in enumerate(data):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(data[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient

    

# if __name__ == "__main__":
#     # Your data (as a Python list of dicts)
#     data = [
#         {
#             "iteration": 1,
#             "metrics": {
#                 "accuracy": 0.2371,
#                 "ece": 0.0963,
#                 "ape": 1.7475,
#                 "flops": 1917120,
#                 "score": -0.54116058430776
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.4,
#                 "p_rate": 0.95,
#                 "num_bayes_layer": 3,
#                 "scale_factor": 0.6
#             }
#         },
#         {
#             "iteration": 2,
#             "metrics": {
#                 "accuracy": 0.1786,
#                 "ece": 0.0174,
#                 "ape": 1.9989,
#                 "flops": 3931170,
#                 "score": 0.049543874625775336
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.95,
#                 "p_rate": 0.7,
#                 "num_bayes_layer": 1,
#                 "scale_factor": 0.9
#             }
#         },
#         {
#             "iteration": 3,
#             "metrics": {
#                 "accuracy": 0.5984,
#                 "ece": 0.0336,
#                 "ape": 1.6632,
#                 "flops": 2506662,
#                 "score": 0.031364363946693244
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.2,
#                 "p_rate": 0.1,
#                 "num_bayes_layer": 2,
#                 "scale_factor": 0.7
#             }
#         },
#         {
#             "iteration": 4,
#             "metrics": {
#                 "accuracy": 0.2098,
#                 "ece": 0.0558,
#                 "ape": 2.0483,
#                 "flops": 452435,
#                 "score": -0.09194228433872143
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.45,
#                 "p_rate": 0.3,
#                 "num_bayes_layer": 1,
#                 "scale_factor": 0.25
#             }
#         },
#         {
#             "iteration": 5,
#             "metrics": {
#                 "accuracy": 0.1756,
#                 "ece": 0.0206,
#                 "ape": 1.8578,
#                 "flops": 4336525,
#                 "score": -0.020782943634756024
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.95,
#                 "p_rate": 0.7,
#                 "num_bayes_layer": 1,
#                 "scale_factor": 0.95
#             }
#         }
#     ]

#     data2 = [
#         {
#             "iteration": 1,
#             "metrics": {
#                 "accuracy": 0.2371,
#                 "ece": 0.0963,
#                 "ape": 1.7475,
#                 "flops": 1917120,
#                 "score": 0.08987163008371235
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.4,
#                 "p_rate": 0.95,
#                 "num_bayes_layer": 3,
#                 "scale_factor": 0.6
#             }
#         },
#         {
#             "iteration": 2,
#             "metrics": {
#                 "accuracy": 0.1786,
#                 "ece": 0.0174,
#                 "ape": 1.9989,
#                 "flops": 3931170,
#                 "score": -0.04873389196493308
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.95,
#                 "p_rate": 0.7,
#                 "num_bayes_layer": 1,
#                 "scale_factor": 0.9
#             }
#         },
#         {
#             "iteration": 3,
#             "metrics": {
#                 "accuracy": 0.5984,
#                 "ece": 0.0336,
#                 "ape": 1.6632,
#                 "flops": 2506662,
#                 "score": 0.3359844627309281
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.2,
#                 "p_rate": 0.1,
#                 "num_bayes_layer": 2,
#                 "scale_factor": 0.7
#             }
#         },
#         {
#             "iteration": 4,
#             "metrics": {
#                 "accuracy": 0.7402,
#                 "ece": 0.0244,
#                 "ape": 1.7673,
#                 "flops": 1644813,
#                 "score": 0.4837559904213181
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.05,
#                 "p_rate": 0.05,
#                 "num_bayes_layer": 2,
#                 "scale_factor": 0.55
#             }
#         },
#         {
#             "iteration": 5,
#             "metrics": {
#                 "accuracy": 0.1976,
#                 "ece": 0.0713,
#                 "ape": 2.2282,
#                 "flops": 51575,
#                 "score": 0.1472824965843925
#             },
#             "hyperparameters": {
#                 "dropout_rate": 0.05,
#                 "p_rate": 0.05,
#                 "num_bayes_layer": 3,
#                 "scale_factor": 0.05
#             }
#         }
#     ]

#     data_dict = {0: data}
#     data_dict[1] = data2

#     table = []

#     strategies = [
#         {
#                 "top_n": 3,
#                 "top_metric": "score",
#                 "name": "Opt-Balance"
#         }, {
#                 "top_n": 1,
#                 "top_metric": "flops",
#                 "name": "Opt-Efficiency"
#             }
#     ]

#     for strat_index, strat_results in data_dict.items():
#         strat_obj = strategies[strat_index]

#         # just log the top n results according to the top metric in the table for particular strategy
#         for i, top_result in enumerate(nlargest(strat_obj["top_n"], strat_results, key=lambda r: r["metrics"][strat_obj["top_metric"]])):
#             row_name = strat_obj["name"]

#             # add top name only if we keep track of multiple top results
#             if strat_obj["top_n"] > 1:
#                 row_name = f"{row_name}-Top{i+1}"

#             row = {"Opt-Mode": row_name}

#             row.update(top_result.get("hyperparameters", {}))
#             row.update(top_result.get("metrics", {}))

#             table.append(row)



#     plot_pareto_3d(data_dict, table)





    
