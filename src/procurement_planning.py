import numpy as np
import itertools
import networkx as nx
import statsmodels.api as sm


def optimal_procurement(supply_cost, x_max, demand_quantity, holding_cost, backlogging_cost, xh_0=0, xh_n=0):
    """Calculates optimal procurement planning.

    Arguments:
    supply_cost -- Supply cost at each time period
    x_max -- Maximum supply quantity at each time period.
    demand_quantity -- Demand quantity at each time period
    holding_cost -- Holding cost.
    backlogging_cost -- Backlogging cost.
    xh_0 -- Initial inventory.
    x_hn -- Final inventory target.
    """

    G = nx.DiGraph()
    n = len(supply_cost)
    total_demand = np.sum(demand_quantity)
    for t in range(n):
        G.add_edge("source", t, {'capacity': x_max[t], 'weight': supply_cost[t]})
        G.add_edge(t, "sink", {'capacity': demand_quantity[t], 'weight': 0})
        G.add_edge(t, t + 1, {'capacity': np.inf, 'weight': holding_cost})
        G.add_edge(t + 1, t, {'capacity': np.inf, 'weight': backlogging_cost})

    G.add_edge("source", -1, {'capacity': xh_0, 'weight': 0})
    G.add_edge(-1, 0, {'capacity': xh_0, 'weight': 0})

    G.add_edge(n, "sink", {'capacity': xh_n, 'weight': 0})

    nx.set_node_attributes(G, "demand", {"source": -total_demand, "sink": total_demand})

    cost, mincost_flow = nx.capacity_scaling(G)
    return cost, np.array([mincost_flow['source'][t] for t in range(n)])

ss = np.array([ 16.97,  17.27,  16.84,  18.27,  19.43]),
dd = np.array([3465466011, 3493734498, 3547479479, 3660968521, 3669918842])

optimal_procurement(ss, [1e13]*5, dd, 10, 1)

def find_ARIMA_pdq(ts):
    aic = 1e9
    model = None
    model_p = None
    for p, d, q in itertools.product(range(5), range(3), range(5)):
        try:
            aa = sm.tsa.ARIMA(ts, (p, d, q)).fit()
            if aa.aic < aic:
                model = aa
                model_p = (p, d, q)
                aic = aa.aic
        except Exception:
            pass
    return model, model_p


def autoForecast(ts, time_steps):
    """Atomatically forecast time series.

    It fits ARIMA(p,d,q) model to the time series. Model selection is using smallest AIC.

    Arguments:
    ts -- time series to be forecasted
    time_steps -- number of time steps to be forecasted
    """
    ts = np.array(ts, dtype=np.float32)
    model, (p, d, q) = find_ARIMA_pdq(ts)
    return model.forecast(time_steps), (p, d, q)


def optimal_procurement_with_history(supply_cost, x_max, demand_quantity, holding_cost, backlogging_cost, **kwargs):
    """Calculates optimal procurement planning.

    This function automatically forecast future supply_cost and demand_quantity quantity by given data. Number of entries in x_max determine how many time steps into future to forecast.

    Arguments:
    supply_cost -- Supply cost history at each time period
    x_max -- Maximum supply quantity at each future.
    demand_quantity -- Demand quantity history at each time period
    holding_cost -- Holding cost.
    backlogging_cost -- Backlogging cost.
    xh_0 -- Initial inventory.
    x_hn -- Final inventory target.
    """

    n = len(np.array(x_max))
    forecast, _ = autoForecast(supply_cost, n)
    s = forecast[0]
    forecast, _ = autoForecast(demand_quantity, n)
    d = forecast[0]

    return optimal_procurement(s, x_max, d, holding_cost, backlogging_cost, **kwargs)
