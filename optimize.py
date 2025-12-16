
import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from prefect.artifacts import create_table_artifact, \
                              create_progress_artifact,\
                              update_progress_artifact

from logic.converter.keras.model_builder import build_uncertainty_model
from logic.pipeline import get_dataset, get_model, \
                           prune_model, \
                           train_model, \
                           eval_trust

from prefect import flow, task

@task
def bayesian_opt(cfg):
    if cfg.bayes_opt.log is None:
        cfg.bayes_opt.log = pd.DataFrame(columns=cfg.bayes_opt.log_cols)
        cfg.bayes_opt.iteration = 0
        cfg.bayes_opt.pbounds = pbounds= {
            "dropout_rate": (0, len(cfg.search_space.dropout_rate_list) - 0.001),
            "p_rate": (0, len(cfg.search_space.p_rate_list) - 0.001),
            "num_bayes_layer": (0, len(cfg.search_space.num_bayes_layer_list) - 0.001),
            "scale_factor": (0, len(cfg.search_space.scale_factor_list) - 0.001)
        }


        cfg.bayes_opt.optimizer = optimizer = BayesianOptimization(
            f = None,
            pbounds=pbounds,
            random_state=cfg.output.seed,
            allow_duplicate_points=True
        )

        # Initial random points
        for _ in range(1):
            cfg.bayes_opt.tune_params = dict(zip(pbounds.keys(), optimizer._space.random_sample()))
    else:
        cfg.bayes_opt.iteration += 1
        optimizer = cfg.bayes_opt.optimizer
        optimizer.register(params=cfg.bayes_opt.tune_params, target=cfg.bayes_opt.score)
        #utility = UtilityFunction(kind="ucb", kappa=2.576, xi=0.0)
        cfg.bayes_opt.tune_params = optimizer.suggest()

    tune_params = cfg.bayes_opt.tune_params
    cfg.model.dropout_rate = cfg.search_space.dropout_rate_list[int(tune_params["dropout_rate"])]
    cfg.model.p_rate = cfg.search_space.p_rate_list[int(tune_params["p_rate"])]
    cfg.model.num_bayes_layer =  cfg.search_space.num_bayes_layer_list[int(tune_params["num_bayes_layer"])]
    cfg.model.scale_factor = cfg.search_space.scale_factor_list[int(tune_params["scale_factor"])]


    # Create a table artifact
    create_table_artifact(
        key=f"bayes-iteration-{cfg.bayes_opt.iteration}",
        table=[
            {
                "Iteration": cfg.bayes_opt.iteration,
                "Previous Score": round(cfg.bayes_opt.score, 4) if cfg.bayes_opt.score is not None else "N/A",
                "Dropout Rate": cfg.model.dropout_rate,
                "P Rate": cfg.model.p_rate,
                "Bayes Layers": cfg.model.num_bayes_layer,
                "Scale Factor": cfg.model.scale_factor
            }
        ],
        description="Bayesian Optimization Step Summary"
    )
    return cfg




@flow(name="Bayesian Optimization Flow")
def perform_optimization(rg):

    rg = get_dataset(rg)
    os.makedirs(rg.output.save_dir, exist_ok=True)

    # Store iteration results here
    iteration_summary = []

    for iter in range(rg.bayes_opt.max_iterations+1):
        rg.bayes_opt.iteration = iter
        rg = bayesian_opt(rg)

        rg = get_model(rg)
        rg = prune_model(rg)
        rg = build_uncertainty_model(rg)
        rg = train_model(rg)

        rg = eval_trust(rg)

        rg.bayes_opt.score = 123

        iteration_summary.append({
                "Iteration": rg.bayes_opt.iteration,
                "Previous Score": round(rg.bayes_opt.score, 4)
                                  if rg.bayes_opt.score is not None else "N/A",
                "Dropout Rate": rg.model.dropout_rate,
                "P Rate": rg.model.p_rate,
                "Bayes Layers": rg.model.num_bayes_layer,
                "Scale Factor": rg.model.scale_factor
            })
        create_table_artifact(
            key=f"xuxu",
            table=iteration_summary,
            description="Summary of all Bayesian Optimization iterations"
        )


    # Create final artifact table
    create_table_artifact(
        key="bayesian-optimization-results",
        table=iteration_summary,
        description="Summary of all Bayesian Optimization iterations"
    )


        #     # flops = get_flops(model, batch_size=1)
        #     # print(f"FLOPS: {flops / 10 ** 6:.03} M")




    #     # print(f"Pruning rate: {cfg.model.p_rate}")
    #     umodel = build_uncertainty_model(cfg, pruned_model)
    #     # flops = int(flops * (1 - cfg.model.p_rate))

    #     # print(f"FLOPS after prune: {flops / 10 ** 6:.03} M")

    #     trained_model = train_model(cfg, umodel, dataset)

    #     accuracy, ece, ape = eval_model(cfg, trained_model)

    #     # print("Full dataset, Accuracy Keras:  {}, ECE Keras {}, aPE Keras {}".format(accuracy, ece, ape))


    #     # if cfg.model.name == "lenet" and accuracy < 0.95:
    #     #     score = -sys.maxsize
    #     # elif cfg.model.name == "resnet" and accuracy < 0.85:
    #     #     score = -sys.maxsize
    #     # else:
    #     #     score = compute_score(accuracy, ece, ape, flops, cfg)

    #     score = 0
    #     flops = 0

    # log_entry = {
    #     "iteration": iter,
    #     "dropout_rate": cfg.model.dropout_rate,
    #     "p_rate": cfg.model.p_rate,
    #     "num_bayes_layer": cfg.model.num_bayes_layer,
    #     "scale_factor": cfg.model.scale_factor,
    #     "accuracy":accuracy,
    #     "flops": flops,
    #     "ece": ece,
    #     "ape": ape,
    #     "score": score
    # }

    # print("bayesian_output_data:\n", log_entry)

    # cfg.bayes_log.loc[len(bayes_log)] = log_entry
    # log_path = os.path.join(cfg.output.save_dir, f"bayesian_opt_iter{iteration}.csv")
    # cfg.bayes_log.to_csv(log_path, index=False)

    # final_log_path = os.path.join(cfg.output.save_dir, "bayesian_opt_final.csv")
    # cfg.bayes_opt.log.to_csv(final_log_path, index=False)
    # return cfg.bayes_opt.log, optimizer.max




def report_results(bayes_log: pd.DataFrame, best_result: dict):
    print("📊 Final Bayesian Optimization Log:")
    print(bayes_log.tail(3))
    print("🏆 Best Result:")
    print(best_result)



def initialize_experiment(rg):
    os.makedirs(rg.output.save_dir, exist_ok=True)
    rg.output.save_dir = os.path.abspath(rg.output.save_dir)
    rg.output.ckpt_pathname = os.path.join(rg.output.save_dir, "best_chkp.tf")


    seed = rg.output.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    configure_gpus(rg.training.gpus)

    return rg

def optimization_flow(rg):
    rg = initialize_experiment(rg)
    rg = perform_optimization(rg)
    #report_results(bayes_log, best_result)