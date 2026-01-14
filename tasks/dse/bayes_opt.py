import pandas as pd
from bayes_opt import BayesianOptimization
from nautic import taskx

class BayesOpt:

    @taskx
    def initialise_bayesian_opt(ctx):
        bo = ctx.bayes_opt
        bo.score = None
        
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

        bo.engine = BayesianOptimization(
            f = None,
            pbounds=pbounds,
            random_state=bo.seed.get(),
            allow_duplicate_points=True
        )

        bo.curr_strategy = ctx.strategy.strategies[ctx.strategy.curr_strategy]

    @taskx
    def bayesian_opt(ctx):
        bo = ctx.bayes_opt
        engine = bo.engine

        bo.score = None
        log = ctx.log

        if bo.iteration == 0:
            # Initial random points
            for _ in range(1):
                bo.control.suggests = dict(zip(bo.tunable.model_fields,
                                                engine._space.random_sample()))

        else:
            BayesOpt.record_iteration(bo, engine, log)
            bo.control.suggests = bo.engine.suggest()

        bo.iteration += 1
        bo.terminate = not (bo.iteration <= bo.num_iter)

        metric_values = BayesOpt.suggest_to_values(bo)

        # update the refs to reflect the new values
        for key, value in metric_values.items():
            bo.control.params['values'][key].set(value)

    # Records current iteration suggest into a summary
    @staticmethod
    def record_iteration(bo, engine, log):
        score = 0
        summary = {
            'iteration': bo.iteration
        }

        for metric in bo.metrics.model_fields:
            metric_value = getattr(bo.metrics, metric).get()
            curr_metric_params = getattr(bo.curr_strategy, metric)

            score += float(metric_value / curr_metric_params.base) * float(curr_metric_params.weight)
            summary[metric] = round(metric_value, 4)

        bo.score = score

        summary['score'] = score
        summary.update(BayesOpt.suggest_to_values(bo))
            
        bo.summary.append(summary)
        log.artifact(key='bayes-iteration-summary',
                    table=bo.summary)

        engine.register(params=bo.control.suggests,
                        target=bo.score)

    # This method converts the suggest into a dictionary of hyper-parameters
    @staticmethod
    def suggest_to_values(bo):
        hyperparams = {}

        for key, value in bo.control.suggests.items():
            idx = int(value)
            hyperparams[key] = bo.control.params['space'][key][idx]

        return hyperparams       
       
       
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

