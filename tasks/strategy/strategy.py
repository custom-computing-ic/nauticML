from nautic import taskx

class Strategy:

    @taskx
    def initialise_strategies(ctx):
        strat = ctx.strategy
        # set the actual strategy dict for each strategy string reference inside of bo.strategies
        # NOTE: could make AliasRef accept dictionaries instead
        strat.strategies = [getattr(ctx, s) for s in strat.strategies]

        strat.curr_strategy = 0
        strat.terminate_strategies = Strategy.terminate_strategy(strat)
        strat.results = {}
    
    @taskx
    def next_strategy(ctx):
        strat = ctx.strategy
        if not Strategy.terminate_strategy(strat):
            strat.curr_strategy += 1
        
        strat.terminate_strategies = Strategy.terminate_strategy(strat)
    
    @taskx
    def save_results(ctx):
        strat = ctx.strategy
        strat.results[strat.curr_strategy] = strat.curr_results.get()

    @staticmethod
    def terminate_strategy(strat):
        return strat.curr_strategy >= len(strat.strategies)
    
    @staticmethod
    def get_curr_strategy_object(ctx):
        return ctx.strategy.strategies[ctx.strategy.curr_strategy]