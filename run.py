import os
import warnings

os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Pytorch converter is not enabled.*", category=UserWarning)

from nautic import flowx, Context
from pathlib import Path

#os.environ["PREFECT_API_URL"] = "http://work.local:4200/api"
os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

# absolute path for config
BASE_DIR = Path(__file__).resolve().parent
ctx = Context.create(str(BASE_DIR / "config/config_lenet.yaml"),
                     log_level="INFO",
                     disable_nautic=False)

#@flowx(name="Bayesian Optimization Flow", flow_run_name = "flow-{run_id}")

@flowx(name="Bayesian Optimization Flow")
def perform_optimization(ctx):
    engine = ctx.engine
    log = ctx.log

    engine.strategy.initialise_strategies()

    while not ctx.strategy.terminate_strategies:
        log.info(f"We are using the strategy: {ctx.strategy.curr_strategy}")

        engine.keras.initialize_experiment()
        engine.keras.get_dataset()

        engine.dse.initialise_bayesian_opt()

        while True:
            # think about triggers
            engine.dse.bayesian_opt()
            if ctx.bayes_opt.terminate:
                engine.strategy.save_results()
                engine.strategy.next_strategy()
                break
                
            engine.keras.get_model()
            engine.keras.trust.build_bayesian_model()
            engine.keras.train_model()
            engine.keras.eval()
        
        best_summary = max(ctx.bayes_opt.summary, key=lambda x: x["metrics"]["score"])

        params = best_summary["hyperparameters"]
        log.info(
        f"""Final parameters:
                droupout rate: {params["dropout_rate"]}
                p rate: {params["p_rate"]}
                scale factor: {params["scale_factor"]}
                num bayes later: {params["num_bayes_layer"]}""")

        metrics = best_summary["metrics"]
        log.info(
        f"""With performance metrics:
                ece: {metrics["ece"]}
                ape: {metrics["ape"]}
                accuracy: {metrics["accuracy"]}
                flops: {metrics["flops"]}""")
    
    engine.strategy.create_result_table()
    engine.strategy.create_4d_pareto_figures()
    engine.strategy.create_correlation_matrix()

    log.info("Finished with all the strategies for Bayesian Optimisation")

if __name__ == "__main__":
    perform_optimization(ctx)

