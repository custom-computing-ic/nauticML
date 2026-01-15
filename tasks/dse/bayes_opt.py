from bayes_opt import BayesianOptimization
from nautic import taskx
from tasks.strategy.strategy import Strategy



class BayesOpt:

    SCORE_PENALTY = -1e6

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

        bo.curr_strategy = Strategy.get_curr_strategy_object(ctx)

    @taskx
    def bayesian_opt(ctx):
        bo = ctx.bayes_opt
        engine = bo.engine

        bo.score = None
        log = ctx.log

        if bo.iteration == 0:
            # Initial random points, use rng to make deterministic
            rng = bo.engine._random_state                      

            for _ in range(1):
                bo.control.suggests = dict(
                zip(
                    bo.tunable.model_fields,
                    rng.uniform(
                        bo.engine._space.bounds[:, 0],
                        bo.engine._space.bounds[:, 1]
                    )
                )
            )
                
        else:
            BayesOpt.record_iteration(bo, engine, log)
            bo.control.suggests = bo.engine.suggest()

        bo.iteration += 1
        bo.terminate = not (bo.iteration <= bo.num_iter)

        metric_values = BayesOpt.suggest_to_values(bo)

        # update the refs to reflect the new values
        for key, value in metric_values.items():
            bo.control.params['values'][key].set(value)

    @taskx
    def show_best_parameters(ctx):
        bo = ctx.bayes_opt
        log = ctx.log
        best_summary = max(bo.summary, key=lambda x: x["metrics"]["score"])

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

    # Records current iteration suggest into a summary
    @staticmethod
    def record_iteration(bo, engine, log):
        score = 0
        summary = {
            'iteration': bo.iteration
        }

        is_penalised = False
        metric_values = {}
        for metric in bo.metrics.model_fields:
            metric_value = getattr(bo.metrics, metric).get()
            metric_values[metric] = round(metric_value, 4)

            curr_metric_params = getattr(bo.curr_strategy, metric)
 
            # If we don't satisfy the minimum or maximum constraints, we have the worst possible score
            if "min" in curr_metric_params.model_fields.keys():
                is_penalised = metric_value < curr_metric_params.min
            
            if "max" in curr_metric_params.model_fields.keys():
                is_penalised = metric_value > curr_metric_params.max


            score += float(metric_value / curr_metric_params.base) * float(curr_metric_params.weight)

        if is_penalised:
            score = BayesOpt.SCORE_PENALTY

        metric_values["score"] = score
        summary["metrics"] = metric_values
        
        bo.score = score

        summary["hyperparameters"] = BayesOpt.suggest_to_values(bo)
            
        bo.summary.append(summary)
        log.artifact(key=f'bayes-iteration-{bo.iteration}-summary-strategy-{bo.curr_strategy.name}'.lower(),
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