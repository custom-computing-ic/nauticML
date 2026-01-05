import pandas as pd
from bayes_opt import BayesianOptimization
from nautic import taskx

#TODO: get the best params from the bo engine to display (using logs),
# understand when the summary acc gets added to the thingo - after score is calculated, so should we also update the summary with the score and all the metrics logged for further reference?

class BayesOpt:
    @taskx
    def bayesian_opt(ctx):
        bo = ctx.bayes_opt
        bo.score = None
        log = ctx.log

        if bo.engine is None:
            bo.iteration = 0
            bo.summary = []

            pbounds = { }
            tune_vals = { }
            tune_space = { }
            bo.control.params = { }
            
            for key in bo.tunable.model_fields:
                opt = getattr(bo.tunable, key)
                if isinstance(opt.space, list):
                    pbounds[key] = (0, len(opt.space) - 0.001)
                else:
                    raise ValueError(f"Unsupported space type for {key}: {type(opt.space)}")

                tune_vals[key] = opt.value
                tune_space[key] = opt.space
                
            bo.control.params['values'] = tune_vals
            bo.control.params['space'] = tune_space

            bo.control.metrics = {}
            metrics_values = {}
            for key in bo.metrics.model_fields:
                metrics_values[key] = getattr(bo.metrics, key)
            bo.control.metrics['values'] = metrics_values

            score_weights = { }
            for key in bo.score_weights.model_fields:
                score_weights[key] = getattr(bo.score_weights, key)
            bo.control.metrics['score_weights'] = score_weights

            bo.engine = BayesianOptimization(
                f = None,
                pbounds=pbounds,
                random_state=bo.seed.get(),
                allow_duplicate_points=True
            )

            # Initial random points
            for _ in range(1):
                bo.control.suggests = dict(zip(pbounds.keys(),
                                               bo.engine._space.random_sample()))
        else:
            engine = bo.engine

            score = 0
            for key in bo.control.metrics['values']:
                metric_value = bo.control.metrics['values'][key].get()
                base_value = bo.control.metrics['score_weights'][key].base
                weight_value = bo.control.metrics['score_weights'][key].weight

                score += float(metric_value / base_value) * float(weight_value)

            bo.score = score

            # record a summary for this bo iteration (can extract to function)
            summary = { 'iteration': bo.iteration, 'score': round(score, 4) }

            # set the parameters for other tasks
            for key, value in bo.control.suggests.items():
                idx = int(value)
                metric_val = bo.control.params['space'][key][idx]
                summary[key] = metric_val
                
            bo.summary.append(summary)
            log.artifact(key='bayes-iteration-summary',
                        table=bo.summary)

            engine.register(params=bo.control.suggests,
                            target=bo.score)

            bo.iteration += 1
            bo.control.suggests = bo.engine.suggest()

        bo.terminate = not (bo.iteration < bo.num_iter)

        metric_values = {}

        for key, value in bo.control.suggests.items():
            idx = int(value)
            metric_val = bo.control.params['space'][key][idx]
            metric_values[key] = metric_val   

            bo.control.params['values'][key].set(metric_val)

        # TODO: make programmatic like above or move to another step
        ctx.model.dropout_rate = metric_values["dropout_rate"]
        ctx.model.scale_factor = metric_values["scale_factor"]
        ctx.model.p_rate = metric_values["p_rate"]
        ctx.model.num_bayes_layer = metric_values["num_bayes_layer"]

        # cfg.model.dropout_rate = cfg.search_space.dropout_rate_list[int(tune_params["dropout_rate"])]
        # cfg.model.p_rate = cfg.search_space.p_rate_list[int(tune_params["p_rate"])]
        # cfg.model.num_bayes_layer =  cfg.search_space.num_bayes_layer_list[int(tune_params["num_bayes_layer"])]
        # cfg.model.scale_factor = cfg.search_space.scale_factor_list[int(tune_params["scale_factor"])]


        # # Create a table artifact
        # create_table_artifact(
        #     key=f"bayes-iteration-{cfg.bayes_opt.iteration}",
        #     table=[
        #         {
        #             "Iteration": cfg.bayes_opt.iteration,
        #             "Previous Score": round(cfg.bayes_opt.score, 4) if cfg.bayes_opt.score is not None else "N/A",
        #             "Dropout Rate": cfg.model.dropout_rate,
        #             "P Rate": cfg.model.p_rate,
        #             "Bayes Layers": cfg.model.num_bayes_layer,
        #             "Scale Factor": cfg.model.scale_factor
        #         }
        #     ],
        #     description="Bayesian Optimization Step Summary"
        # )
        # return cfg

