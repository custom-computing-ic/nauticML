import os
import warnings
os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Pytorch converter is not enabled.*", category=UserWarning)

from nautic import flowx, Context

#os.environ["PREFECT_API_URL"] = "http://work.local:4200/api"
os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

ctx = Context.create("config.yaml",
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
        engine.dse.bayesian_opt()
        if ctx.bayes_opt.terminate:
            break
        engine.keras.get_model()
        engine.keras.trust.build_bayesian_model()
        engine.keras.train_model()
        engine.keras.eval()
        #print("Accuracy:", ctx.eval.accuracy)
        #engine.keras.trust.eval()




ctx.eval.ece = 10
ctx.eval.ape = 20

ctx.eval.flop = 1000


perform_optimization(ctx)







