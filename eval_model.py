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
ctx = Context.create(str(BASE_DIR / "config/config_eval.yaml"),
                     log_level="INFO",
                     disable_nautic=False)

@flowx(name="Evalutation Flow")
def perform_optimization(ctx):
    engine = ctx.engine
    log = ctx.log

    engine.keras.initialize_experiment()

    engine.keras.get_dataset()
    engine.keras.get_model()

    engine.keras.trust.build_bayesian_model()

    engine.keras.train_model()
    engine.keras.eval.eval()
    
    metrics = ctx.eval
    params = ctx.model
    
    log.info(
    f"""Final parameters:
            droupout rate: {params.dropout_rate}
            p rate: {params.p_rate}
            scale factor: {params.scale_factor}
            num bayes later: {params.num_bayes_layer}""")

    log.info(
        f"""With performance metrics:
                ece: {metrics.ece}
                ape: {metrics.ape}
                accuracy: {metrics.accuracy}
                flops: {metrics.flops}
                power: {metrics.power}
                energy: {metrics.energy}""")
    
    log.info("Finished evluating model ")

if __name__ == "__main__":
    perform_optimization(ctx)

