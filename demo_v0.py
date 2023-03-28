position_limit_pearls = 20
position_limit_bananas = 20

position_limit_coconuts = 600
position_limit_pina = 300

position_limit_diving = 50
position_limit_berries = 250

position_limit_baguette = 150
position_limit_dip = 300
position_limit_ukulele = 70
position_limit_picnic = 70


def price_vol_extractor(structure, df):

    if type(structure) == dict:
        for order_price in structure.keys():
            new_row = pd.Series([order_price, structure[order_price]], index=[
                                'price', 'volume'])
            # axis = 0 for rows not 1 (corrected)
            df = df.concat(new_row, axis=0, ignore_index=True)

    elif type(structure) == Trade:
        for item in structure:
            new_row = pd.Series([item.price, item.quantity],
                                index=['price', 'volume'])
            df = df.concat(new_row, axis=0, ignore_index=True)

    else:
        raise RaiseError(f"Unknown structure detected: {structure}")

    return df


# r1
def pearls_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    # mean reversion based strategy
    prices = df_bid_ask.price
    volumes = df_bid_ask.volume.abs()

    avg_mean_price = sum(prices * volumes) / (volumes.sum())
    avg_variance = sum(prices ** 2 * volumes) / \
        (volumes.sum()) - av_mean_price ** 2

    # 0.4?
    acceptable_ask_price = avg_mean_price + 0.4 * np.sqrt(avg_variance)
    acceptable_bid_price = avg_mean_price - 0.4 * np.sqrt(avg_variance)

    return acceptable_ask_price, acceptable_bid_price


def bananas_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    # strategy?
    prices = df_bid_ask.price
    volumes = df_bid_ask.volume

    x = np.array(prices).astype(str).astype(float)
    y = np.array(volumes).astype(str).astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    # why are both the same?
    acceptable_ask_price = -intercept / slope
    acceptable_bid_price = -intercept / slope

    return acceptable_ask_price, acceptable_bid_price

# r2


def pina_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def coconuts_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass

# r3


def diving_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def berries_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass

# r4


def baguette_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def dip_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def ukulele_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def picnic_fv(df_bid_ask: pd.DataFrame, df_trades: pd.DataFrame):
    pass


def fair_value(product, df_bid_ask):

    # r1
    if product == 'PEARLS':
        acceptable_price_ask, acceptable_price_bid = pearls_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_pearls
    elif product == 'BANANAS':
        acceptable_price_ask, acceptable_price_bid = bannas_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_bananas

    # r2
    elif product == 'PINA_COLADAS':
        acceptable_price_ask, acceptable_price_bid = pina_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_pina
    elif product == 'COCONUTS':
        acceptable_price_ask, acceptable_price_bid = coconuts_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_coconuts

    # r3
    elif product == 'DIVING_GEAR':
        acceptable_price_ask, acceptable_price_bid = diving_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_diving
    elif product == 'BERRIES':
        acceptable_price_ask, acceptable_price_bid = berris_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_berries

    # r4
    elif product == 'BAGUETTE':
        acceptable_price_ask, acceptable_price_bid = baguette_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_beaguette
    elif product == 'DIP':
        acceptable_price_ask, acceptable_price_bid = dip_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_dip
    elif product == 'UKULELE':
        acceptable_price_ask, acceptable_price_bid = ukulele_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_ukulele
    elif product == 'PICNIC_BASKET':
        acceptable_price_ask, acceptable_price_bid = picnic_fv(
            df_bid_ask, df_trades)
        position_limit = position_limit_picnic

    else:
        raise ValueError("Unknown product")

    return (acceptable_price_ask, acceptable_price_bid, position_limit)


class Trader:

    def __init__(self):
        pass

    def run(self, state: TradingState) -> Dict[str, List[Order]]:

        output_orders = {}

        for product in state.order_depths[product]:

            order_depth = state.order_depths[product]
            market_trades = state.market_trades[product]

            # contains both buys and sells (not same as positionless)
            orders: List[Order] = []

            # creating dataframes to store TradingState price-volume data
            df_bid_ask = pd.DataFrame(columns=[
                                      'price_' + product.split('_').lower(), 'volume_' + product.split('_').lower()])
            df_trades = pd.DataFrame(columns=[
                                     'price_' + product.split('_').lower(), 'volume_' + product.split('_').lower()])

            # filling dataframe with latest TradingState price-volume data
            try:
                df_bid_ask = price_vol_extractor(
                    order_depth.buy_order, df_bid_ask)
                df_bid_ask = price_vol_extractor(
                    order_depth.sell_orders, df_bid_ask)
            except:
                raise RaiseError("Insufficient size of the dictionaries")

            try:
                df_bid_ask = price_vol_extractor(market_trades, df_bid_ask)
            except:
                raise RaiseError("Insufficient size of the list")

            # fair price according to corresponding tradingstate
            try:
                acceptable_price_ask, acceptable_price_bid, position_limit = fair_value(
                    product, df_bid_ask, df_trades)
            except:
                raise RaiseError(
                    "Pricing could not be done due to unknown product")

            # favorable to buy
            try:
                # finding the cheapest option to buy
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = - \
                    order_depth.sell_orders[best_ask]  # as qty -ve

                # checking for feasibility of price and position limit
                # !!!!!!!!!!!!!!!!!!!!(while or if)
                if ((best_ask < acceptable_price_bid)):

                    # if we DON'T hold the product previously in any amount
                    if (product not in state.position.keys()):
                        old_positon = 0
                    else:
                        old_positon = state.position[product]

                    if (np.absolute(best_ask_volume + old_positon) <= position_limit):
                        # placing full buy order
                        print("BUY", str(best_ask_volume) + 'x', best_ask)
                        orders.append(
                            Order(product, best_ask, best_ask_volume))
                        state.position[product] = best_ask_volume + old_positon
                    else:
                        print(
                            f"Position limit exceeded for buying full order of {product}")
                        # placing partial buy order
                        print("BUY", str(position_limit -
                              old_positon) + 'x', best_ask)
                        orders.append(
                            Order(product, best_ask, position_limit - old_positon))
                        state.position[product] = position_limit

                    # updating the order depth for sells(as reduced from open market)
                    order_depth.sell_orders[best_ask] += orders[-1].quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        order_depth.sell_orders.pop(best_ask)

                    #!!!!!!!!!! don't we need to update the order depth for buys
                else:
                    print('Market is selling at a premium')

            except:
                raise RaiseError("No sell orders in the market currently")

            # favorable to sell
            try:
                # finding the cheapest option to buy
                best_bid = max(order_depths.buy_orders.keys())
                # as qty +ve
                best_bid_volume = order_depths.buy_orders[best_bid]

                # checking for feasibility of price and position limit
                # !!!!!!!!!!!!!!!!!!!! (while or if)
                if ((best_bid > acceptable_price_ask)):

                    # if we DON'T hold the product previously in any amount
                    if (product not in state.position.keys()):
                        old_positon = 0
                    else:
                        old_positon = state.position[product]

                    # -best_bid_volume as being sold (np.absolute(check)), check qty sold/bought
                    if (np.absolute(-best_bid_volume + old_positon) <= position_limit):
                        # placing full sell order
                        print("SELL", str(best_bid_volume) + 'x', best_bid)
                        orders.append(
                            Order(product, best_bid, -(best_bid_volume)))  # as qty -ve for sell
                        state.position[product] = - \
                            best_bid_volume + old_positon
                    else:
                        print(
                            f"Position limit exceeded for selling full order of {product}")
                        # placing parital buy order
                        print("SELL", str(position_limit -
                              old_positon) + 'x', best_ask)
                        orders.append(
                            Order(product, best_ask, -(position_limit - old_positon)))
                        state.position[product] = -position_limit

                # updating the order depth for sells(as reduced from open market)
                order_depth.buy_orders[best_bid] += orders[-1].quantity
                if order_depth.buy_orders[best_bid] == 0:
                    order_depth.buy_orders.pop(best_bid)

                #!!!!!!!!!! don't we need to update the order depth for buys
                else:
                    print('Market is buying at a premium')
            except:
                raise RaiseError("No buy orders in the market currently")

            output_orders[product] = orders

        return output_orders
