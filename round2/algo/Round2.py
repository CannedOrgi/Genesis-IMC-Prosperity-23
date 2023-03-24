import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datamodel import Listing, OrderDepth, Trade, TradingState, Order
import statistics


def perals_fv(bid_ask_hystory: pd.DataFrame) -> Tuple[float, float]:
    prices = bid_ask_hystory.price
    volumes = bid_ask_hystory.volume.abs()
    av_mean_price = sum(prices * volumes) / (volumes.sum())
    av_std_price = sum(prices ** 2 * volumes) / \
        (volumes.sum()) - av_mean_price ** 2
    ask_price = av_mean_price + 0.4 * np.sqrt(av_std_price)
    bid_price = av_mean_price - 0.4 * np.sqrt(av_std_price)
    return ask_price, bid_price


def bananas_fv(trades: pd.DataFrame, bid_ask_hystory: pd.DataFrame) -> Tuple[float, float]:
    prices = bid_ask_hystory.price
    volumes = bid_ask_hystory.volume
    if len(trades) > 0:
        long_mean = trades.price.mean()
    else:
        long_mean = prices.mean()
    ratio = prices / long_mean
    ratio = ratio.sort_values(ascending=True)

    quantiles = statistics.quantiles(ratio, n=11)  # n=11/7
    bid_price = quantiles[0]
    ask_price = quantiles[-1]
    ask_price = ask_price * long_mean
    bid_price = bid_price * long_mean

    return ask_price, bid_price


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:

        # Initialize the method output dict as an empty dict
        result = {}
        check = True
        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
            if product == "BANANAS" or product == "PEARLS":
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
                        df_bid_ask = df_bid_ask.append(
                            new_row, ignore_index=True)

                    for order_price in order_depth.buy_orders.keys():
                        new_row = pd.Series([order_price, order_depth.buy_orders[order_price]],
                                            index=['price', 'volume'])
                        df_bid_ask = df_bid_ask.append(
                            new_row, ignore_index=True)

                if len(mkt_trades) > 0:
                    for item in mkt_trades:
                        # Filling the DataFrame
                        new_row = pd.Series([item.price, item.quantity], index=[
                                            'price', 'volume'])
                        df_trades = df_trades.append(
                            new_row, ignore_index=True)

                if product == "PEARLS":
                    acceptable_price_ask, acceptable_price_bid = perals_fv(
                        df_bid_ask)

                if product == "BANANAS":
                    acceptable_price_ask, acceptable_price_bid = bananas_fv(
                        df_trades, df_bid_ask)

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
                                print(
                                    "BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                                # Update the orders list
                                orders.append(Order(product, best_ask, -order_depth.sell_orders[
                                    best_ask]))  # buy the ask if the ask is small and we dont have the product
                                # Update the position
                                state.position[product] = - \
                                    order_depth.sell_orders[best_ask]
                            else:
                                # If the quantity is higher than the position limit, buy the position limit
                                print("BUY", str(20) + "x", best_ask)
                                # Update the orders list
                                # buy 20 on the ask
                                orders.append(Order(product, best_ask, 20))
                                state.position[product] = 20

                        # In the case in which there is a position for prod
                        if product in state.position.keys():
                            if -order_depth.sell_orders[best_ask] + state.position[product] <= 20:
                                # In the case the position + the ask_volume is higher than the position limit
                                print(
                                    "BUY", str(-order_depth.sell_orders[best_ask]) + "x", best_ask)
                                # Update the orders list
                                orders.append(
                                    Order(product, best_ask, -order_depth.sell_orders[best_ask]))  # buy on the ask
                                # Update the position
                                state.position[product] += - \
                                    order_depth.sell_orders[best_ask]
                            else:
                                # In the case the position + the ask_volume is lower than the position limit
                                print("BUY", str(
                                    20 - state.position[product]) + "x", best_ask)
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

                if len(order_depth.buy_orders) > 0:  # favorable to buy
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    while best_bid > acceptable_price_ask and (
                            (product in state.position.keys() and state.position[product] > -20) or (
                            product not in state.position.keys())):
                        # In the case in which there is a position for prod
                        if product in state.position.keys():  # we have position and if we buy, we are stll within min pos
                            if -order_depth.buy_orders[best_bid] + state.position[product] >= -20:
                                # In the case the position + the bid_volume is lower than the position limit
                                print("SELL", str(
                                    order_depth.buy_orders[best_bid]) + "x", best_bid)
                                # Update the orders list
                                orders.append(
                                    Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                                # Update the position
                                state.position[product] -= order_depth.buy_orders[best_bid]
                            else:
                                # In the case the position + the bid_volume is higer than the position limit
                                print(
                                    "SELL", str(-20 - state.position[product]) + "x", best_bid)
                                # Update the orders list
                                orders.append(
                                    Order(product, best_bid,
                                          -20 - state.position[product]))  # sell on bid to max out pos
                                # Update the position
                                state.position[product] = -20

                        # In the case in which there is no position for prod
                        if product not in state.position.keys():
                            # if small bid and we dont have position
                            if order_depth.buy_orders[best_bid] < 20:
                                # If the quantity is lower than the position limit
                                print("SELL", str(
                                    order_depth.buy_orders[best_bid]) + "x", best_bid)
                                # Update the orders list
                                orders.append(
                                    Order(product, best_bid, -order_depth.buy_orders[best_bid]))
                                # Update the position #sell on bid
                                state.position[product] = - \
                                    order_depth.buy_orders[best_bid]

                            else:
                                # If the quantity is higher than the position limit, sell the position limit
                                print("SELL", str(20) + "x", best_bid)
                                # Update the orders list
                                # sell 20 on bid
                                orders.append(Order(product, best_bid, -20))
                                # Update the position
                                state.position[product] = -20

                        order_depth.buy_orders[best_bid] += orders[-1].quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            order_depth.buy_orders.pop(best_bid)

                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                        # Update the result with the list of orders for prod
                result[product] = orders

            else:
                if check:
                    check = False
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
                    df_prices = pd.DataFrame(
                        columns=['price_coco', 'price_pina'])
                    # Create a DataFrame to store trades of other partecipants
                    df_trades = pd.DataFrame(
                        columns=['price_coc', 'price_pina'])

                    if len(order_depth_c.sell_orders) > 0 and len(order_depth_pc.sell_orders) > 0:
                        # Filling the DataFrame
                        l = min(len(order_depth_c.sell_orders),
                                len(order_depth_pc.sell_orders))
                        prices_c = [
                            item for item in order_depth_c.sell_orders.keys()]
                        prices_pc = [
                            item for item in order_depth_pc.sell_orders.keys()]

                        for i in range(0, l):
                            new_row = pd.Series([prices_c[i], prices_pc[i]],
                                                index=['price_coc', 'price_pina'])
                            df_prices = df_prices.append(
                                new_row, ignore_index=True)

                    if len(order_depth_c.buy_orders) > 0 and len(order_depth_pc.buy_orders) > 0:
                        # Filling the DataFrame
                        l = min(len(order_depth_c.buy_orders),
                                len(order_depth_pc.buy_orders))
                        prices_c = [
                            item for item in order_depth_c.buy_orders.keys()]
                        prices_pc = [
                            item for item in order_depth_pc.buy_orders.keys()]

                        for i in range(0, l):
                            new_row = pd.Series([prices_c[i], prices_pc[i]],
                                                index=['price_coc', 'price_pina'])
                            df_prices = df_prices.append(
                                new_row, ignore_index=True)

                    # Compute the relation between the prices

                    x = np.array(df_prices.price_coc).astype(str).astype(float)
                    y = np.array(df_prices.price_pina).astype(
                        str).astype(float)
                    slope, intercept = np.polyfit(x, y, 1)

                    unitary_c_to_pc = slope + intercept
                    unitary_pc_to_c = -slope / intercept

                    # favorable to buy
                    if len(order_depth_c.sell_orders) > 0 and len(order_depth_pc.sell_orders) > 0:
                        best_ask_c = min(order_depth_c.sell_orders.keys())
                        best_ask_pc = min(order_depth_pc.sell_orders.keys())

                        if best_ask_pc < intercept + slope * best_ask_c:
                            if ("PINA_COLADAS" in state.position.keys() and state.position["PINA_COLADAS"] < 300) or (
                                    "PINA_COLADAS" not in state.position.keys()):
                                if "PINA_COLADAS" not in state.position.keys():
                                    if -order_depth_pc.sell_orders[best_ask_pc] <= 300:
                                        # If the quantity is lower than the position limit
                                        print(
                                            "BUY", str(-order_depth_pc.sell_orders[best_ask_pc]) + "x", best_ask_pc)
                                        # Update the orders list
                                        orders_pc.append(Order("PINA_COLADAS", best_ask_pc, -order_depth_pc.sell_orders[
                                            best_ask_pc]))  # buy the ask if the ask is small and we dont have the product
                                        # Update the position
                                        state.position["PINA_COLADAS"] = - \
                                            order_depth_pc.sell_orders[best_ask_pc]
                                    else:
                                        # If the quantity is higher than the position limit, buy the position limit
                                        print("BUY", str(300) +
                                              "x", best_ask_pc)
                                        # Update the orders list
                                        # buy 20 on the ask
                                        orders_pc.append(
                                            Order("PINA_COLADAS", best_ask_pc, 300))
                                        state.position[product] = 300
                                        # In the case in which there is a position for prod
                                    if "PINA_COLADAS" in state.position.keys():
                                        if -order_depth_pc.sell_orders[best_ask_pc] + state.position[
                                                "PINA_COLADAS"] <= 300:
                                            # In the case the position + the ask_volume is higher than the position limit
                                            print("BUY", str(-order_depth_pc.sell_orders[best_ask_pc]) + "x",
                                                  best_ask_pc)
                                            # Update the orders list
                                            orders_pc.append(
                                                Order("PINA_COLADAS", best_ask_pc,
                                                      -order_depth_pc.sell_orders["PINA_COLADAS"]))  # buy on the ask
                                            # Update the position
                                            state.position["PINA_COLADAS"] += - \
                                                order_depth_pc.sell_orders[best_ask_pc]
                                        else:
                                            # In the case the position + the ask_volume is lower than the position limit
                                            print("BUY", str(
                                                300 - state.position["PINA_COLADAS"]) + "x", best_ask_pc)
                                            # Update the orders list
                                            orders_pc.append(Order("PINA_COLADAS", best_ask_pc,
                                                                   300 - state.position[
                                                                       "PINA_COLADAS"]))  # buy 20 - current pos on the ask
                                            # Update the position
                                            state.position["PINA_COLADAS"] = 300

                    # favorable to buy
                    if len(order_depth_c.buy_orders) > 0 and len(order_depth_pc.buy_orders) > 0:
                        best_bid_c = max(order_depth_c.buy_orders.keys())
                        best_bid_pc = min(order_depth_pc.buy_orders.keys())

                        if best_bid_pc < intercept + slope * best_bid_c:
                            if ("PINA_COLADAS" in state.position.keys() and state.position["PINA_COLADAS"] > -300) or (
                                    "PINA_COLADAS" not in state.position.keys()):
                                if "PINA_COLADAS" in state.position.keys():  # we have position and if we buy, we are stll
                                    # within min pos
                                    if -order_depth_pc.buy_orders[best_bid_pc] + state.position["PINA_COLADAS"] >= -300:
                                        # In the case the position + the bid_volume is lower than the position limit
                                        print("SELL", str(
                                            order_depth_pc.buy_orders[best_bid_pc]) + "x", best_bid_pc)
                                        # Update the orders list
                                        orders_pc.append(
                                            Order("PINA_COLADAS", best_bid_pc, -order_depth_pc.buy_orders[best_bid_pc]))
                                        # Update the position
                                        state.position["PINA_COLADAS"] -= order_depth_pc.buy_orders[best_bid_pc]
                                    else:
                                        # In the case the position + the bid_volume is higer than the position limit
                                        print(
                                            "SELL", str(-300 - state.position["PINA_COLADAS"]) + "x", best_bid_pc)
                                        # Update the orders list
                                        orders_pc.append(
                                            Order("PINA_COLADAS", best_bid_pc,
                                                  -300 - state.position["PINA_COLADAS"]))  # sell on bid to max out pos
                                        # Update the position
                                        state.position["PINA_COLADAS"] = -300

                                # In the case in which there is no position for prod
                                if "PINA_COLADAS" not in state.position.keys():
                                    if order_depth_pc.buy_orders[
                                            best_bid_pc] < 300:  # if small bid and we dont have position
                                        # If the quantity is lower than the position limit
                                        print("SELL", str(
                                            order_depth_pc.buy_orders[best_bid_pc]) + "x", best_bid_pc)
                                        # Update the orders list
                                        orders_pc.append(
                                            Order("PINA_COLADAS", best_bid_pc, -order_depth_pc.buy_orders[best_bid_pc]))
                                        # Update the position #sell on bid
                                        state.position["PINA_COLADAS"] = - \
                                            order_depth_pc.buy_orders[best_bid_pc]

                                    else:
                                        # If the quantity is higher than the position limit, sell the position limit
                                        print("SELL", str(20) +
                                              "x", best_bid_pc)
                                        # Update the orders list
                                        # sell 20 on bid
                                        orders_pc.append(
                                            Order("PINA_COLADAS", best_bid_pc, -300))
                                        # Update the position
                                        state.position["PINA_COLADAS"] = -300
                    # Update the result with the list of orders for prod
                    result["PINA_COLADAS"] = orders_pc

        return result
