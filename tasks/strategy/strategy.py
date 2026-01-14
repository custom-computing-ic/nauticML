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
    
    @taskx
    def next_strategy(ctx):
        strat = ctx.strategy
        if not Strategy.terminate_strategy(strat):
            strat.curr_strategy += 1
        
        strat.terminate_strategies = Strategy.terminate_strategy(strat)
    
    @staticmethod
    def terminate_strategy(strat):
        return strat.curr_strategy >= len(strat.strategies)