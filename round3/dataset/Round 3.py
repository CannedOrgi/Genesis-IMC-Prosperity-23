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
    # Retrieve the trades that other market participants have done since the last TradingState came in for
    # product
    if product in state.market_trades.keys():
        mkt_trades = state.market_trades[product]

    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []

    # Create a DataFrame to store all the market BUY and SELL orders for product
    df_bid_ask = pd.DataFrame(columns=['price', 'volume'])
    # Create a DataFrame to store trades of other partecipants
    df_trades = pd.DataFrame(columns=['price', 'volume'])

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

    if len(mkt_trades) > 0:
        for item in mkt_trades:
            # Filling the DataFrame
            new_row = pd.Series([item.price, item.quantity], index=['price', 'volume'])
            df_trades = df_trades.append(new_row, ignore_index=True)

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
    # Retrieve the trades that other market participants have done since the last TradingState came in for
    # product
    mkt_trades = state.market_trades[product]

    # Initialize the list of Orders to be sent as an empty list
    orders: list[Order] = []

    # Create a DataFrame to store all the market BUY and SELL orders for product
    df_bid_ask = pd.DataFrame(columns=['price', 'volume'])
    # Create a DataFrame to store trades of other partecipants
    df_trades = pd.DataFrame(columns=['price', 'volume'])

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

    if len(mkt_trades) > 0:
        for item in mkt_trades:
            # Filling the DataFrame
            new_row = pd.Series([item.price, item.quantity], index=['price', 'volume'])
            df_trades = df_trades.append(new_row, ignore_index=True)

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


def COCONUTS_PINACOLADA(state: TradingState, result: Dict[str, List[Order]]):
    # Retrieve the Order Depth containing all the market BUY and SELL orders for product
    order_depth_c: OrderDepth = state.order_depths["COCONUTS"]
    order_depth_pc: OrderDepth = state.order_depths["PINA_COLADAS"]

    # Retrieve the trades that other market participants have done since the last TradingState came
    # in for product
    mkt_trades_c = state.market_trades["COCONUTS"]
    mkt_trades_pc = state.market_trades["PINA_COLADAS"]

    # Initialize the list of Orders to be sent as an empty list
    orders_c: list[Order] = []
    orders_pc: list[Order] = []

    # Create a DataFrame to store all the market BUY and SELL orders for product
    df_prices = pd.DataFrame(columns=['price_coco', 'price_pina'])
    # Create a DataFrame to store trades of other partecipants
    df_trades = pd.DataFrame(columns=['price_coc', 'price_pina'])

    if len(order_depth_c.sell_orders) > 0 and len(order_depth_pc.sell_orders) > 0:
        # Filling the DataFrame
        l = min(len(order_depth_c.sell_orders), len(order_depth_pc.sell_orders))
        prices_c = [item for item in order_depth_c.sell_orders.keys()]
        prices_pc = [item for item in order_depth_pc.sell_orders.keys()]

        for i in range(0, l):
            new_row = pd.Series([float(prices_c[i]), float(prices_pc[i])],
                                index=['price_coco', 'price_pina'])
            df_prices = df_prices.append(new_row, ignore_index=True)

    if len(order_depth_c.buy_orders) > 0 and len(order_depth_pc.buy_orders) > 0:
        # Filling the DataFrame
        l = min(len(order_depth_c.buy_orders), len(order_depth_pc.buy_orders))
        prices_c = [item for item in order_depth_c.buy_orders.keys()]
        prices_pc = [item for item in order_depth_pc.buy_orders.keys()]

        for i in range(0, l):
            new_row = pd.Series([float(prices_c[i]), float(prices_pc[i])],
                                index=['price_coco', 'price_pina'])
            df_prices = df_prices.append(new_row, ignore_index=True)

    # Compute the relation between the prices
    df_prices["diff"]=abs(df_prices["price_coco"]-df_prices["price_pina"])
    mean_dif = df_prices["diff"].mean()
    std_dif = df_prices["diff"].std()

    best_ask_pc = min(order_depth_pc.sell_orders.keys())
    best_ask_c = min(order_depth_c.sell_orders.keys())
    best_ask_volume_pc = order_depth_pc.sell_orders[best_ask_pc]
    best_ask_volume_c = order_depth_c.sell_orders[best_ask_c]

    best_bid_pc = min(order_depth_pc.buy_orders.keys())
    best_bid_c = min(order_depth_c.buy_orders.keys())
    best_bid_volume_pc = order_depth_pc.buy_orders[best_bid_pc]
    best_bid_volume_c = order_depth_c.buy_orders[best_bid_c]

    while best_ask_pc - best_bid_c > mean_dif and state.position["PINA_COLADAS"] < 300 and state.position["COCONUTS"] > -600:
        if ("PINA_COLADAS" in state.position.keys() and state.position["PINA_COLADAS"] < 300) or (
                "PINA_COLADAS" not in state.position.keys()):
            if ("COCONUTS" in state.position.keys() and state.position["COCONUTS"] > -600) or (
                    "COCONUTS" not in state.position.keys()):
                if best_ask_volume_pc < best_bid_volume_c:
                    logger.print("BUY", str(-order_depth_pc.sell_orders[best_ask_pc]) + "x", best_ask_pc)
                    logger.print("SELL", str(order_depth_c.buy_orders[best_bid_c]) + "x", best_bid_c)
                    # Update the orders list
                    orders_pc.append(Order("PINA_COLADAS", best_ask_pc, -order_depth_pc.sell_orders[
                        best_ask_pc]))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position["PINA_COLADAS"] = -order_depth_pc.sell_orders[best_ask_pc]
                    # Update the orders list
                    orders_c.append(
                        Order("COCONUTS", best_bid_c, -order_depth_c.buy_orders[best_bid_c]))
                    # Update the position
                    state.position["COCONUTS"] -= order_depth_c.buy_orders[best_bid_c]
                if best_ask_volume_pc > best_bid_volume_c:
                    best_ask_volume_pc = best_bid_volume_c
                    logger.print("BUY", str(-best_ask_volume_pc) + "x", best_ask_pc)
                    logger.print("SELL", str(order_depth_c.buy_orders[best_bid_c]) + "x", best_bid_c)
                    # Update the orders list
                    orders_pc.append(Order("PINA_COLADAS", best_ask_pc, -best_ask_volume_pc))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position["PINA_COLADAS"] = -best_ask_volume_pc
                    # Update the orders list
                    orders_c.append(
                        Order("COCONUTS", best_bid_c, -order_depth_c.buy_orders[best_bid_c]))
                    # Update the position
                    state.position["COCONUTS"] -= order_depth_c.buy_orders[best_bid_c]

        order_depth_pc.sell_orders[best_ask_pc] += orders_pc[-1].quantity
        order_depth_c.buy_orders[best_bid_c] += orders_c[-1].quantity
        if order_depth_pc.sell_orders[best_ask_pc] == 0:
            order_depth_pc.sell_orders.pop(best_ask_pc)

        if order_depth_c.buy_orders[best_bid_c] == 0:
            order_depth_c.buy_orders.pop(best_bid_c)

        best_ask_pc = min(order_depth_pc.sell_orders.keys())
        best_bid_c = min(order_depth_c.buy_orders.keys())
        best_ask_volume_pc = order_depth_pc.sell_orders[best_ask_pc]
        best_bid_volume_c = order_depth_c.buy_orders[best_bid_c]

    while abs(best_ask_c - best_bid_pc) > mean_dif and state.position["PINA_COLADAS"] > -300 and state.position["COCONUTS"] < 600:
        if ("PINA_COLADAS" in state.position.keys() and state.position["PINA_COLADAS"] > -300) or (
                "PINA_COLADAS" not in state.position.keys()):
            if ("COCONUTS" in state.position.keys() and state.position["COCONUTS"] < 600) or (
                    "COCONUTS" not in state.position.keys()):
                if best_ask_volume_c > best_bid_volume_pc:
                    logger.print("BUY", str(-order_depth_c.buy_orders[best_ask_c]) + "x", best_ask_c)
                    logger.print("SELL", str(order_depth_pc.sell_orders[best_bid_pc]) + "x", best_bid_pc)
                    # Update the orders list
                    orders_pc.append(Order("PINA_COLADAS", best_bid_pc, -order_depth_pc.buy_orders[
                        best_bid_pc]))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position["PINA_COLADAS"] = -order_depth_pc.buy_orders[best_bid_pc]
                    # Update the orders list
                    orders_c.append(
                        Order("COCONUTS", best_ask_c, -order_depth_c.sell_orders[best_ask_c]))
                    # Update the position
                    state.position["COCONUTS"] -= order_depth_c.sell_orders[best_ask_c]
                if  best_ask_volume_c < best_bid_volume_pc:
                    best_bid_volume_pc = best_ask_volume_c
                    logger.print("BUY", str(-best_ask_volume_c) + "x", best_ask_c)
                    logger.print("SELL", str(order_depth_pc.sell_orders[best_bid_pc]) + "x", best_bid_pc)
                    # Update the orders list
                    orders_pc.append(Order("PINA_COLADAS", best_bid_pc, -best_bid_volume_pc))  # buy the ask if the ask is small and we dont have the product
                    # Update the position
                    state.position["PINA_COLADAS"] = -best_bid_volume_pc
                    # Update the orders list
                    orders_c.append(
                        Order("COCONUTS", best_ask_c, -order_depth_c.sell_orders[best_ask_c]))
                    # Update the position
                    state.position["COCONUTS"] -= order_depth_c.sell_orders[best_ask_c]

        order_depth_pc.buy_orders[best_bid_pc] += orders_pc[-1].quantity
        order_depth_c.sell_orders[best_ask_c] += orders_c[-1].quantity
        if order_depth_pc.buy_orders[best_bid_pc] == 0:
            order_depth_pc.buy_orders.pop(best_bid_pc)

        if order_depth_c.sell_orders[best_ask_c] == 0:
            order_depth_c.sell_orders.pop(best_ask_c)

        best_ask_c = min(order_depth_c.sell_orders.keys())
        best_bid_pc = best_bid_pc = min(order_depth_pc.buy_orders.keys())
        best_ask_volume_c = order_depth_c.sell_orders[best_ask_c]
        best_bid_volume_pc = order_depth_pc.buy_orders[best_bid_pc]




    # Update the result with the list of orders for prod
    result["PINA_COLADAS"] = orders_pc
    result["COCONUTS"] = orders_c

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}

        PEARLS(state, result)
        BANANAS(state, result)
        COCONUTS_PINACOLADA(state, result)

        logger.flush(state, result)

        return result