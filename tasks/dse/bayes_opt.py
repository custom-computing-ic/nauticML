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

            # NOTE: could make AliasRef accept dictionaries instead
            # set the actual strategy dict for each strategy string reference inside of bo.strategies
            bo.strategies = [getattr(ctx, s) for s in bo.strategies]

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
            curr_strategy = bo.strategies[bo.curr_strategy]

            score = 0
            summary = {
                'iteration': bo.iteration
            }
            
            for metric in bo.metrics.model_fields:
                metric_value = getattr(bo.metrics, metric).get()
                curr_metric_params = getattr(curr_strategy, metric)

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

            bo.iteration += 1
            bo.control.suggests = bo.engine.suggest()

        bo.terminate = not (bo.iteration < bo.num_iter)

        metric_values = BayesOpt.suggest_to_values(bo)

        # update the refs to reflect the new values
        for key, value in metric_values.items():
            bo.control.params['values'][key].set(value)

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

