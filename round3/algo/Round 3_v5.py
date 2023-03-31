import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from typing import Any
from datamodel import Listing, OrderDepth, Trade, TradingState, Order, ProsperityEncoder, Symbol
import statistics
import json


def perals_fv(bid_ask_hystory: pd.DataFrame) -> Tuple[float, float]:
    prices = bid_ask_hystory.price
    volumes = bid_ask_hystory.volume.abs()
    av_mean_price = sum(prices * volumes) / (volumes.sum())
    av_variance = sum(prices ** 2 * volumes) / (volumes.sum()) - av_mean_price ** 2
    ask_price = av_mean_price + 0.4 * np.sqrt(av_variance)
    bid_price = av_mean_price - 0.4 * np.sqrt(av_variance)
    return ask_price, bid_price


def PEARLS(state: TradingState, result: Dict[str, List[Order]]):
    product = "PEARLS"
    # Retrieve the Order Depth containing all the market BUY and SELL orders for product
    order_depth: OrderDepth = state.order_depths[product]
    # Create a DataFrame to store trades of other partecipants
    df_trades = pd.DataFrame(columns=['price', 'volume'])
    # Retrieve the trades that other market participants have done since the last TradingState came in for
    # product
    if product in state.market_trades.keys():
        mkt_trades = state.market_trades[product]
        if len(mkt_trades) > 0:
            for item in mkt_trades:
            # Filling the DataFrame
                new_row = pd.Series([item.price, item.quantity], index=['price', 'volume'])
                df_trades = df_trades.append(new_row, ignore_index=True)

    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []

    # Create a DataFrame to store all the market BUY and SELL orders for product
    df_bid_ask = pd.DataFrame(columns=['price', 'volume'])

    if len(order_depth.sell_orders) > 0:
        # Filling the DataFrame
        for order_price in order_depth.sell_orders.keys():
            new_row = pd.Series([order_price, order_depth.sell_orders[order_price]],
                                index=['price', 'volume'])
            df_bid_ask = df_bid_ask.append(new_row, ignore_index=True)

        for order_price in order_depth.buy_orders.keys():
            new_row = pd.Series([order_price, order_depth.buy_orders[order_price]],
                                index=['price', 'volume'])
            df_bid_ask = df_bid_ask.append(new_row, ignore_index=True)


    acceptable_price_ask, acceptable_price_bid = perals_fv(df_bid_ask)

    if len(order_depth.sell_orders) > 0:  # favorable to buy
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask]
        while best_ask < acceptable_price_bid and (
                (product in state.position.keys() and state.position[product] < 20) or (
                product not in state.position.keys())):
            # In the case in which there is no position for prod
            if product not in state.position.keys():
                if -order_depth.sell_orders[best_ask] <= 20:
                    # If the quantity is lower than the position limit
                    logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask, -order_depth.sell_orders[
                        best_ask]))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position[product] = -order_depth.sell_orders[best_ask]
                else:
                    # If the quantity is higher than the position limit, buy the position limit
                    logger.print("BUY", str(20) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask, 20))  # buy 20 on the ask
                    state.position[product] = 20

            # In the case in which there is a position for prod
            if product in state.position.keys():
                if -order_depth.sell_orders[best_ask] + state.position[product] <= 20:
                    # In the case the position + the ask_volume is higher than the position limit
                    logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                    # Update the orders list
                    orders.append(
                        Order(product, best_ask, -order_depth.sell_orders[best_ask]))  # buy on the ask
                    # Update the position
                    state.position[product] += -order_depth.sell_orders[best_ask]
                else:
                    # In the case the position + the ask_volume is lower than the position limit
                    logger.print("BUY", str(20 - state.position[product]) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask,
                                        20 - state.position[product]))  # buy 20 - current pos on the ask
                    # Update the position
                    state.position[product] = 20

            order_depth.sell_orders[best_ask] += orders[-1].quantity

    if len(order_depth.buy_orders) > 0:  # favorable to buyPEARLS_Trade.py
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        while best_bid > acceptable_price_ask and (
                (product in state.position.keys() and state.position[product] > -20) or (
                product not in state.position.keys())):
            # In the case in which there is a position for prod
            if product in state.position.keys():  # we have position and if we buy, we are stll within min pos
                if -order_depth.buy_orders[best_bid] + state.position[product] >= -20:
                    # In the case the position + the bid_volume is lower than the position limit
                    logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                    # Update the position
                    state.position[product] -= order_depth.buy_orders[best_bid]
                else:
                    # In the case the position + the bid_volume is higer than the position limit
                    logger.print("SELL", str(-20 - state.position[product]) + "x", best_bid)
                    # Update the orders list
                    orders.append(
                        Order(product, best_bid,
                              -20 - state.position[product]))  # sell on bid to max out pos
                    # Update the position
                    state.position[product] = -20

            # In the case in which there is no position for prod
            if product not in state.position.keys():
                if order_depth.buy_orders[best_bid] < 20:  # if small bid and we dont have position
                    # If the quantity is lower than the position limit
                    logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                    # Update the position #sell on bid
                    state.position[product] = -order_depth.buy_orders[best_bid]

                else:
                    # If the quantity is higher than the position limit, sell the position limit
                    logger.print("SELL", str(20) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -20))  # sell 20 on bid
                    # Update the position
                    state.position[product] = -20

            order_depth.buy_orders[best_bid] += orders[-1].quantity
            # Update the result with the list of orders for prod
    result[product] = orders


def bananas_fv(trades: pd.DataFrame, bid_ask_hystory: pd.DataFrame) -> Tuple[float, float]:
    prices = bid_ask_hystory.price
    volume = bid_ask_hystory.volume

    x = np.array(prices).astype(str).astype(float)
    y = np.array(volume).astype(str).astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    ask_price = -intercept / slope
    bid_price = -intercept / slope

    return ask_price, bid_price


def BANANAS(state: TradingState, result: Dict[str, List[Order]]):
    product = "BANANAS"
    # Retrieve the Order Depth containing all the market BUY and SELL orders for product
    order_depth: OrderDepth = state.order_depths[product]
    # Create a DataFrame to store trades of other partecipants
    df_trades = pd.DataFrame(columns=['price', 'volume'])
    # Retrieve the trades that other market participants have done since the last TradingState came in for
    # product

    if product in state.market_trades.keys():
        mkt_trades = state.market_trades[product]
        if len(mkt_trades) > 0:
            for item in mkt_trades:
            # Filling the DataFrame
                new_row = pd.Series([item.price, item.quantity], index=['price', 'volume'])
                df_trades = df_trades.append(new_row, ignore_index=True)

    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []

    # Create a DataFrame to store all the market BUY and SELL orders for product
    df_bid_ask = pd.DataFrame(columns=['price', 'volume'])
    

    if len(order_depth.sell_orders) > 0:
        # Filling the DataFrame
        for order_price in order_depth.sell_orders.keys():
            new_row = pd.Series([order_price, order_depth.sell_orders[order_price]],
                                index=['price', 'volume'])
            df_bid_ask = df_bid_ask.append(new_row, ignore_index=True)

        for order_price in order_depth.buy_orders.keys():
            new_row = pd.Series([order_price, order_depth.buy_orders[order_price]],
                                index=['price', 'volume'])
            df_bid_ask = df_bid_ask.append(new_row, ignore_index=True)

    

    acceptable_price_ask, acceptable_price_bid = bananas_fv(df_trades, df_bid_ask)

    if len(order_depth.sell_orders) > 0:  # favorable to buy
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask]
        while best_ask < acceptable_price_bid and (
                (product in state.position.keys() and state.position[product] < 20) or (
                product not in state.position.keys())):
            # In the case in which there is no position for prod
            if product not in state.position.keys():
                if -order_depth.sell_orders[best_ask] <= 20:
                    # If the quantity is lower than the position limit
                    logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask, -order_depth.sell_orders[
                        best_ask]))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position[product] = -order_depth.sell_orders[best_ask]
                else:
                    # If the quantity is higher than the position limit, buy the position limit
                    logger.print("BUY", str(20) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask, 20))  # buy 20 on the ask
                    state.position[product] = 20

            # In the case in which there is a position for prod
            if product in state.position.keys():
                if -order_depth.sell_orders[best_ask] + state.position[product] <= 20:
                    # In the case the position + the ask_volume is higher than the position limit
                    logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                    # Update the orders list
                    orders.append(
                        Order(product, best_ask, -order_depth.sell_orders[best_ask]))  # buy on the ask
                    # Update the position
                    state.position[product] += -order_depth.sell_orders[best_ask]
                else:
                    # In the case the position + the ask_volume is lower than the position limit
                    logger.print("BUY", str(20 - state.position[product]) + "x", best_ask)
                    # Update the orders list
                    orders.append(Order(product, best_ask,
                                        20 - state.position[product]))  # buy 20 - current pos on the ask
                    # Update the position
                    state.position[product] = 20

            order_depth.sell_orders[best_ask] += orders[-1].quantity
            if order_depth.sell_orders[best_ask] == 0:
                order_depth.sell_orders.pop(best_ask)

            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]

    if len(order_depth.buy_orders) > 0:  # favorable to buyPEARLS_Trade.py
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        while best_bid > acceptable_price_ask and (
                (product in state.position.keys() and state.position[product] > -20) or (
                product not in state.position.keys())):
            # In the case in which there is a position for prod
            if product in state.position.keys():  # we have position and if we buy, we are stll within min pos
                if -order_depth.buy_orders[best_bid] + state.position[product] >= -20:
                    # In the case the position + the bid_volume is lower than the position limit
                    logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                    # Update the position
                    state.position[product] -= order_depth.buy_orders[best_bid]
                else:
                    # In the case the position + the bid_volume is higer than the position limit
                    logger.print("SELL", str(-20 - state.position[product]) + "x", best_bid)
                    # Update the orders list
                    orders.append(
                        Order(product, best_bid,
                              -20 - state.position[product]))  # sell on bid to max out pos
                    # Update the position
                    state.position[product] = -20

            # In the case in which there is no position for prod
            if product not in state.position.keys():
                if order_depth.buy_orders[best_bid] < 20:  # if small bid and we dont have position
                    # If the quantity is lower than the position limit
                    logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                    # Update the position #sell on bid
                    state.position[product] = -order_depth.buy_orders[best_bid]

                else:
                    # If the quantity is higher than the position limit, sell the position limit
                    logger.print("SELL", str(20) + "x", best_bid)
                    # Update the orders list
                    orders.append(Order(product, best_bid, -20))  # sell 20 on bid
                    # Update the position
                    state.position[product] = -20

            order_depth.buy_orders[best_bid] += orders[-1].quantity
            if order_depth.buy_orders[best_bid] == 0:
                order_depth.buy_orders.pop(best_bid)

            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            # Update the result with the list of orders for prod
    result[product] = orders

def berries(state: TradingState, result: Dict[str, List[Order]]):
    # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
    for product in state.order_depths.keys():
            if product=='BERRIES':
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders_b: list[Order] = []

                # Define a fair value for the PEARLS.
                #acceptable_price=9998
                
                trades=order_depth
                dict_bid={}
                for key in trades.buy_orders.keys():
                    
                    dict_bid[key]=trades.buy_orders[key]
                for key in trades.sell_orders.keys():
                    if key not in dict_bid.keys():
                        dict_bid[key]=abs(trades.sell_orders[key])
                    if key in dict_bid.keys():
                        dict_bid[key]+=abs(trades.sell_orders[key])
                bid=0
                y=0
                for price in dict_bid.keys():
                    bid+=price*dict_bid[price]
                    y+=dict_bid[price]
                bid/=y
                acceptable_price = bid
                acceptable_price_bid=bid-2
                acceptable_price_ask=bid+2
                if len(Trader.bids)>7:

                    SAA=Trader.bids[len(Trader.bids)-7:len(Trader.bids)]
                    SMA=statistics.mean(SAA)
                    acceptable_price=SMA
                    std=statistics.stdev(SAA)
                    acceptable_price_bid=acceptable_price-0.6*std
                    acceptable_price_ask=acceptable_price+0.6*std

                Trader.bids.append(bid)
                if len(order_depth.sell_orders) > 0:  # favorable to buy
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    while best_ask < acceptable_price_bid and (
                            (product in state.position.keys() and state.position[product] < 250) or (
                            product not in state.position.keys())):
                        # In the case in which there is no position for prod
                        if product not in state.position.keys():
                            if -order_depth.sell_orders[best_ask] <= 250:
                                # If the quantity is lower than the position limit
                                logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                                # Update the orders list
                                orders_b.append(Order(product, best_ask, -order_depth.sell_orders[
                                    best_ask]))  # buy the ask if the ask is small and we dont have the product
                                # Update the position
                                state.position[product] = -order_depth.sell_orders[best_ask]
                            else:
                                # If the quantity is higher than the position limit, buy the position limit
                                logger.print("BUY", str(250) + "x", best_ask)
                                # Update the orders list
                                orders_b.append(Order(product, best_ask, 250))  # buy 20 on the ask
                                state.position[product] = 250

                        # In the case in which there is a position for prod
                        if product in state.position.keys():
                            if -order_depth.sell_orders[best_ask] + state.position[product] <= 250:
                                # In the case the position + the ask_volume is higher than the position limit
                                logger.print("BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                                # Update the orders list
                                orders_b.append(
                                    Order(product, best_ask, -order_depth.sell_orders[best_ask]))  # buy on the ask
                                # Update the position
                                state.position[product] += -order_depth.sell_orders[best_ask]
                            else:
                                # In the case the position + the ask_volume is lower than the position limit
                                logger.print("BUY", str(250 - state.position[product]) + "x", best_ask)
                                # Update the orders list
                                orders_b.append(Order(product, best_ask,
                                                    250 - state.position[product]))  # buy 20 - current pos on the ask
                                # Update the position
                                state.position[product] = 250

                        order_depth.sell_orders[best_ask] += orders_b[-1].quantity

                if len(order_depth.buy_orders) > 0:  # favorable to buyPEARLS_Trade.py
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    while best_bid > acceptable_price_ask and (
                            (product in state.position.keys() and state.position[product] > -250) or (
                            product not in state.position.keys())):
                        # In the case in which there is a position for prod
                        if product in state.position.keys():  # we have position and if we buy, we are stll within min pos
                            if -order_depth.buy_orders[best_bid] + state.position[product] >= -250:
                                # In the case the position + the bid_volume is lower than the position limit
                                logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                                # Update the orders list
                                orders_b.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                                # Update the position
                                state.position[product] -= order_depth.buy_orders[best_bid]
                            else:
                                # In the case the position + the bid_volume is higer than the position limit
                                logger.print("SELL", str(-250 - state.position[product]) + "x", best_bid)
                                # Update the orders list
                                orders_b.append(
                                    Order(product, best_bid,
                                          -250 - state.position[product]))  # sell on bid to max out pos
                                # Update the position
                                state.position[product] = -250

                        # In the case in which there is no position for prod
                        if product not in state.position.keys():
                            if order_depth.buy_orders[best_bid] < 250:  # if small bid and we dont have position
                                # If the quantity is lower than the position limit
                                logger.print("SELL", str(order_depth.buy_orders[best_bid]) + "x", best_bid)
                                # Update the orders list
                                orders_b.append(Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                                # Update the position #sell on bid
                                state.position[product] = -order_depth.buy_orders[best_bid]

                            else:
                                # If the quantity is higher than the position limit, sell the position limit
                                logger.print("SELL", str(250) + "x", best_bid)
                                # Update the orders list
                                orders_b.append(Order(product, best_bid, -250))  # sell 20 on bid
                                # Update the position
                                state.position[product] = -250

                        order_depth.buy_orders[best_bid] += orders_b[-1].quantity
    result["BERRIES"]=orders_b

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""


logger = Logger()

class Trader:
    bid=0
    bids=[]

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}

        PEARLS(state, result)
        BANANAS(state, result)
        
        berries(state, result)

        logger.flush(state, result)

        return result