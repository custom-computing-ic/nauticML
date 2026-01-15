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

    engine.keras.initialize_experiment()
    engine.strategy.initialise_strategies()

    while not ctx.strategy.terminate_strategies:
        log.info(f"We are using the strategy: {ctx.strategy.curr_strategy}")

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
        
        engine.dse.show_best_parameters()
    
    engine.strategy.create_result_figures()
    log.info("Finished with all the strategies for Bayesian Optimisation")

if __name__ == "__main__":
    perform_optimization(ctx)

