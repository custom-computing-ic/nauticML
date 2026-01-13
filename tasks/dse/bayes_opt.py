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
            curr_strategy = bo.strategies[bo.curr_strategy].get()

            score = 0
            for metric in bo.control.metrics['values']:
                metric_value = bo.control.metrics['values'][metric].get()

                base_value = curr_strategy[metric]
                weight_value = bo.control.metrics['score_base'][metric]

                score += float(metric_value / base_value) * float(weight_value)

            bo.score = score

            # record a summary for this bo iteration (can extract to function)
            summary = { 
                'iteration': bo.iteration, 
                'score': round(score, 4),
                'accuracy': round(bo.metrics.accuracy.get(), 4),
                'ece': round(bo.metrics.ece.get(), 4),
                'ape': round(bo.metrics.aPE.get(), 4),
                'flops': round(bo.metrics.FLOP.get(), 4)
            }
            
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

    @staticmethod
    def suggest_to_values(bo):
        metric_values = {}

        for key, value in bo.control.suggests.items():
            idx = int(value)
            metric_val = bo.control.params['space'][key][idx]
            metric_values[key] = metric_val   

        return metric_values       
       
       
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

