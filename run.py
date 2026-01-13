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
ctx = Context.create(str(BASE_DIR / "config.yaml"),
                     log_level="INFO",
                     disable_nautic=False)

#@flowx(name="Bayesian Optimization Flow", flow_run_name = "flow-{run_id}")

@flowx(name="Bayesian Optimization Flow")
def perform_optimization(ctx):
    engine = ctx.engine
    log = ctx.log

    engine.keras.initialize_experiment()
    engine.keras.get_dataset()

    while True:
        # think about triggers
        engine.dse.bayesian_opt()
        if ctx.bayes_opt.terminate:
            break
            
        engine.keras.get_model()
        engine.keras.trust.build_bayesian_model()
        engine.keras.train_model()
        engine.keras.eval()
    
    engine.dse.create_pareto_figures()
     
    best_summary = max(ctx.bayes_opt.summary, key=lambda x: x["score"])

    log.info(f"""
          Final parameters:
                droupout rate: {best_summary["dropout_rate"]}
                p rate: {best_summary["p_rate"]}
                scale factor: {best_summary["scale_factor"]}
                num bayes later: {best_summary["num_bayes_layer"]}""")

    log.info(f"""
          With performance metrics:
            ece: {ctx.eval.ece}
            ape: {ctx.eval.ape}
            accuracy: {ctx.eval.accuracy}
            flops: {ctx.eval.flops}""")

if __name__ == "__main__":
    perform_optimization(ctx)

