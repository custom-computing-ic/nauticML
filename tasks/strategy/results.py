from nautic import taskx

from heapq import nlargest
import pandas as pd
import os

class Results:
    @taskx
    def create_result_table(ctx):
        strat = ctx.strategy
        log = ctx.log

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

        log.artifact(key='strategy-top-results-table', table=table)

        # save table as markdown in figures subfolder of save dir
        df = pd.DataFrame(table)

        columns = ["Opt-Mode"] + [c for c in df.columns if c != "Opt-Mode"]
        df = df[columns]
        md_df = df.to_markdown(index=False)

        log.info("\nTop strategy results:\n%s", md_df)

        figures_dir = os.path.join(strat.save_dir.get(), "figures")
        os.makedirs(figures_dir, exist_ok=True) 

        md_path = os.path.join(figures_dir, "strategy_top_results_table.md")
        with open(md_path, "w") as f:
            f.write("# Top Strategy Results\n\n")
            f.write(md_df)

    @taskx
    def create_pareto_figures(ctx):
        pass

    @taskx
    def create_correlation_matrix(ctx):
        pass